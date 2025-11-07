import argparse
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Utility: camera & rays
# -------------------------------
def make_pinhole_rays(H, W, fov_deg=50.0, cam_origin=(0, 0, 4.0), lookat=(0, 0, 0)):
    """
    Returns per-pixel ray origins (H*W,3) and directions (H*W,3)
    """
    fx = 0.5 * W / math.tan(0.5 * math.radians(fov_deg))
    fy = 0.5 * H / math.tan(0.5 * math.radians(fov_deg))
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    # pixel centers
    x = (i + 0.5 - W * 0.5) / fx
    y = -(j + 0.5 - H * 0.5) / fy
    dirs = torch.stack([x, y, -torch.ones_like(x)], dim=-1)  # camera looks -z
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)

    # camera-to-world (simple: place cam at origin, look at center)
    o = torch.tensor(cam_origin, dtype=torch.float32)
    # Build basis (z forward)
    z = F.normalize(torch.tensor(lookat, dtype=torch.float32) - o, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    xaxis = F.normalize(torch.linalg.cross(up, z), dim=0)
    yaxis = torch.linalg.cross(z, xaxis)
    R = torch.stack([xaxis, yaxis, z], dim=1)  # 3x3

    dirs = dirs.reshape(-1, 3) @ R.T
    origins = o.expand_as(dirs)
    return origins, F.normalize(dirs, dim=-1)


# -------------------------------
# Implicit field + color
# -------------------------------
class BlobbyGyroid(nn.Module):
    """
    F(p,t) = G(p+W) + κ*log(1+M(p+W)) + H(p+W,t) - b
    ρ = softplus(-(F)/δ) ; color uses normal n = ∇F/||∇F||
    """

    def __init__(self, J=4, N=16, K=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        # --- Warp W(p,t) = Σ a_j ⊙ sin(B_j p + ω_j t + φ_j)
        self.J = J
        self.warp_a = nn.Parameter(
            torch.randn(J, 3, generator=g) * 0.2
        )  # amplitudes (3,)
        self.warp_B = nn.Parameter(torch.randn(J, 3, 3, generator=g) * 0.7)  # 3x3
        self.warp_omega = nn.Parameter(torch.randn(J, generator=g) * 0.5)  # scalar
        self.warp_phi = nn.Parameter(torch.randn(J, 3, generator=g))  # phase per-axis

        # --- Gyroid core
        self.omega = nn.Parameter(torch.tensor([1.6, 1.7, 1.8]))  # per-axis freq
        self.phi = nn.Parameter(torch.zeros(3))
        self.alpha = nn.Parameter(torch.tensor(0.55))

        # --- Metaballs
        self.N = N
        self.mb_w = nn.Parameter(torch.ones(N) * 0.6)
        self.mb_beta_raw = nn.Parameter(torch.randn(N) * 0.0)  # beta = softplus + floor
        self.mb_cbar = nn.Parameter(torch.randn(N, 3, generator=g))
        self.mb_u = nn.Parameter(torch.randn(N, 3, generator=g) * 0.2)
        self.mb_nu = nn.Parameter(torch.abs(torch.randn(N, generator=g)) * 0.6 + 0.2)
        self.mb_psi = nn.Parameter(torch.randn(N, generator=g))

        # --- Harmonics
        self.K = K
        self.h_s = nn.Parameter(torch.randn(K) * 0.1)
        self.h_k = nn.Parameter(torch.randn(K, 3))
        self.h_w = nn.Parameter(torch.randn(K) * 0.7)
        self.h_zeta = nn.Parameter(torch.randn(K))

        # --- Globals
        self.kappa = nn.Parameter(torch.tensor(0.7))
        self.bias_b = nn.Parameter(torch.tensor(0.0))
        self.delta_raw = nn.Parameter(torch.tensor(-2.2))  # δ = softplus -> around 0.1

        # --- Color params (glass + emissive)
        self.q0 = nn.Parameter(torch.tensor([0.6, 0.8, 1.0]))  # base tint
        self.q1 = nn.Parameter(torch.tensor(0.5))
        self.Q2 = nn.Parameter(torch.randn(3, 3) * 0.1)
        self.q3 = nn.Parameter(torch.tensor(0.6))
        self.eta = nn.Parameter(torch.tensor(2.5))
        self.w_vec = nn.Parameter(torch.randn(3))
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.light_dir = nn.Parameter(
            F.normalize(torch.tensor([0.3, 0.8, 0.4]), dim=0), requires_grad=False
        )

    # --- Pieces -------------------------------------------------
    def warp(self, p, t):
        # p: (...,3), t: scalar or (...,1)
        # broadcast t to (...,1)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=p.dtype, device=p.device)
        t = t.expand(p.shape[:-1] + (1,))
        W = 0.0
        for j in range(self.J):
            arg = (p @ self.warp_B[j].T) + self.warp_omega[j] * t + self.warp_phi[j]
            W = W + self.warp_a[j] * torch.sin(arg)
        return W

    def gyroid(self, q):
        # G(q)
        x, y, z = q.unbind(-1)
        wx, wy, wz = self.omega
        phix, phiy, phiz = self.phi
        Gq = (
            torch.sin(wx * x + phix) * torch.cos(wy * y + phiy)
            + torch.sin(wy * y + phiy) * torch.cos(wz * z + phiz)
            + torch.sin(wz * z + phiz) * torch.cos(wx * x + phix)
            - self.alpha
        )
        return Gq

    def metaballs(self, q, t):
        # q: (...,3), t: (...,1)
        q_ = q.unsqueeze(-2)  # (...,1,3)
        # Compute center positions: c = mb_cbar + mb_u * sin(mb_nu * t + mb_psi)
        # mb_cbar: (N,3), mb_u: (N,3), mb_nu: (N,), mb_psi: (N,)
        # t: (batch,1) - need to broadcast properly
        # Result should be: (batch, N, 3)
        c = self.mb_cbar[None, :, :] + self.mb_u[None, :, :] * torch.sin(
            self.mb_nu[None, :, None] * t[:, None, :] + self.mb_psi[None, :, None]
        )
        beta = F.softplus(self.mb_beta_raw) + 0.3
        dist2 = ((q_ - c) ** 2).sum(-1)  # (...,N)
        M = torch.sum(self.mb_w * torch.exp(-beta * dist2), dim=-1)
        return M

    def harmonics(self, q, t):
        # Σ s_k sin(k·q + ω t + ζ)
        s = 0.0
        for k in range(self.K):
            s = s + self.h_s[k] * torch.sin(
                (q @ self.h_k[k]) + self.h_w[k] * t.squeeze(-1) + self.h_zeta[k]
            )
        return s

    def field_F(self, p, t):
        q = p + self.warp(p, t)
        Gq = self.gyroid(q)
        Mq = self.metaballs(q, t)
        Hq = self.harmonics(q, t)
        Fp = Gq + self.kappa * torch.log1p(Mq) + Hq - self.bias_b
        return Fp, q

    def density_and_color(self, p, t, need_normals=True):
        # Make sure p requires grad for normals
        if need_normals and not p.requires_grad:
            p = p.clone().detach().requires_grad_(True)

        Fval, q = self.field_F(p, t)
        delta = F.softplus(self.delta_raw) + 1e-3  # > 0
        sigma = F.softplus(-(Fval) / delta)  # density

        if need_normals:
            grad = torch.autograd.grad(
                Fval,
                p,
                grad_outputs=torch.ones_like(Fval),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            n = F.normalize(grad, dim=-1)
        else:
            n = torch.zeros_like(p)

        # Simple differentiable shading
        ndotl = n @ self.light_dir
        emissive = torch.sin(self.eta * (q @ self.w_vec) + self.zeta * t.squeeze(-1))
        rgb = torch.sigmoid(
            self.q0
            + self.q1 * ndotl.unsqueeze(-1)
            + (n @ self.Q2.T)
            + self.q3 * emissive.unsqueeze(-1)
        )
        return sigma.unsqueeze(-1), rgb, Fval, n


# -------------------------------
# Volume rendering (NeRF-style)
# -------------------------------
def render(
    model, rays_o, rays_d, t_scalar, near=0.0, far=8.0, n_samples=96, stratified=True
):
    """
    rays_o/d: (N,3). Returns rgb (N,3), aux dict
    """
    device = rays_o.device
    N = rays_o.shape[0]
    t = torch.full((N, 1), float(t_scalar), device=device)

    # Sample points along rays
    z_vals = torch.linspace(near, far, n_samples, device=device)
    if stratified:
        mids = 0.5 * (z_vals[:-1] + z_vals[1:])
        upper = torch.cat([mids, z_vals[-1:]], 0)
        lower = torch.cat([z_vals[:1], mids], 0)
        z_vals = lower + (upper - lower) * torch.rand((N, n_samples), device=device)
    else:
        z_vals = z_vals.expand(N, n_samples)

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]  # (N,S,3)
    pts = pts.reshape(-1, 3)
    t_rep = t.repeat_interleave(n_samples, dim=0)  # (N*S,1)

    sigma, rgb, Fval, n = model.density_and_color(pts, t_rep, need_normals=True)
    sigma = sigma.reshape(N, n_samples, 1)
    rgb = rgb.reshape(N, n_samples, 3)

    # Volumetric integration
    dists = torch.cat(
        [z_vals[:, 1:] - z_vals[:, :-1], torch.full((N, 1), 1e10, device=device)], dim=1
    )  # (N,S)
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (N,S)
    # cumulative product for transmittance
    T = torch.cumprod(
        torch.cat([torch.ones((N, 1), device=device), 1.0 - alpha + 1e-10], dim=1),
        dim=1,
    )[:, :-1]
    weights = alpha * T  # (N,S)
    comp_rgb = torch.sum(weights[..., None] * rgb, dim=1)  # (N,3)
    depth = torch.sum(weights * z_vals, dim=1)  # (N,)
    return comp_rgb, {"weights": weights, "depth": depth}


# -------------------------------
# Video loader
# -------------------------------
def load_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Could not open {path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).float() / 255.0)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    vid = torch.stack(frames, dim=0)  # T,H,W,3
    return vid, fps


# -------------------------------
# Training
# -------------------------------
def train_on_video(
    mp4_path,
    iters=4000,
    rays_per_step=2048,
    samples_per_ray=96,
    lr=2e-3,
    fov=50.0,
    eikonal_w=0.05,
    device="cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    video, fps = load_video(mp4_path)
    T, H, W, _ = video.shape
    print(f"Loaded video: {T} frames, {W}x{H}, fps={fps:.2f}")
    # Normalize time to seconds
    times = torch.arange(T, dtype=torch.float32) / float(fps)

    # Camera rays (fixed camera assumption)
    rays_o_full, rays_d_full = make_pinhole_rays(H, W, fov_deg=fov)
    rays_o_full = rays_o_full.to(device)
    rays_d_full = rays_d_full.to(device)

    model = BlobbyGyroid(J=4, N=16, K=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Precompute pixel coords for random sampling
    all_pix = torch.stack(
        torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1
    ).reshape(-1, 2)

    for it in range(1, iters + 1):
        if it % 10 == 0 or it == 1:
            print("starting iteration", it)
        model.train()
        # Random frame/time
        fidx = np.random.randint(0, T)
        t_now = times[fidx].item()
        # Random pixels
        sel = torch.randint(0, H * W, (rays_per_step,))
        pix = all_pix[sel]
        target = video[fidx, pix[:, 0], pix[:, 1]].to(device)  # (N,3)

        rays_o = rays_o_full[sel]
        rays_d = rays_d_full[sel]

        pred_rgb, aux = render(
            model,
            rays_o,
            rays_d,
            t_now,
            near=0.0,
            far=8.0,
            n_samples=samples_per_ray,
            stratified=True,
        )

        loss = F.mse_loss(pred_rgb, target)

        # Optional Eikonal: sample random points in volume
        if eikonal_w > 0.0:
            P = 4096
            p_rand = (torch.rand(P, 3, device=device) - 0.5) * 6.0  # within [-3,3]^3
            p_rand.requires_grad_(True)
            t_rand = torch.full((P, 1), t_now, device=device)
            Fval, _ = model.field_F(p_rand, t_rand)
            grad = torch.autograd.grad(
                Fval, p_rand, torch.ones_like(Fval), create_graph=True
            )[0]
            eik = ((grad.norm(dim=-1) - 1.0) ** 2).mean()
            loss = loss + eikonal_w * eik

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 100 == 0:
            print(f"[{it:05d}] loss={loss.item():.6f}")
            render_full_video(
                model,
                (H, W),
                (rays_o_full, rays_d_full),
                times,
                out_path=f"intermediate_{it:05d}.mp4",
                n_samples=samples_per_ray,
                device=device,
            )

    return model, (H, W), (rays_o_full, rays_d_full), times


# -------------------------------
# Rendering a reconstructed video
# -------------------------------
def render_full_video(
    model, size_hw, rays_all, times, out_path="recon.mp4", n_samples=128, device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    H, W = size_hw
    rays_o_full, rays_d_full = rays_all
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, max(1.0, len(times) / len(times)), (W, H)
    )
    for ti, t_now in enumerate(times.tolist()):
        rgb_img = torch.zeros(H * W, 3, device=device)
        chunk = 8192
        for s in range(0, H * W, chunk):
            e = min(s + chunk, H * W)
            with torch.set_grad_enabled(True):
                rgb, _ = render(
                    model,
                    rays_o_full[s:e],
                    rays_d_full[s:e],
                    t_now,
                    near=0.0,
                    far=8.0,
                    n_samples=n_samples,
                    stratified=False,
                )
            rgb_img[s:e] = rgb.clamp(0, 1).detach()
        rgb_img = rgb_img.reshape(H, W, 3).cpu().numpy()
        bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        print(f"rendered frame {ti + 1}/{len(times)}")
    writer.release()
    print(f"Wrote {out_path}")


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="path to .mp4")
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fov", type=float, default=50.0)
    args = parser.parse_args()

    model, size_hw, rays_all, times = train_on_video(
        args.video, iters=args.iters, device=args.device, fov=args.fov
    )

    # optional: render reconstruction at original frame times
    render_full_video(
        model,
        size_hw,
        rays_all,
        times,
        out_path="reconstruction.mp4",
        device=args.device,
    )
