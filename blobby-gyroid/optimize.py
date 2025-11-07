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
        self.alpha = nn.Parameter(torch.tensor(3.0))  # large alpha to make G≈0 on average

        # --- Metaballs
        self.N = N
        self.mb_w = nn.Parameter(torch.ones(N) * 1.5)  # stronger individual weights
        self.mb_beta_raw = nn.Parameter(torch.randn(N) * 0.3)  # vary sharpness
        self.mb_cbar = nn.Parameter(torch.randn(N, 3, generator=g) * 0.8)  # tight in [-0.8,0.8]^3 near origin
        self.mb_u = nn.Parameter(torch.randn(N, 3, generator=g) * 0.3)  # moderate motion
        self.mb_nu = nn.Parameter(torch.abs(torch.randn(N, generator=g)) * 1.0 + 0.5)  # faster movement
        self.mb_psi = nn.Parameter(torch.randn(N, generator=g) * 3.14159)  # random phase

        # --- Harmonics
        self.K = K
        self.h_s = nn.Parameter(torch.randn(K) * 0.01)  # much smaller contribution
        self.h_k = nn.Parameter(torch.randn(K, 3))
        self.h_w = nn.Parameter(torch.randn(K) * 0.7)
        self.h_zeta = nn.Parameter(torch.randn(K))

        # --- Globals
        self.kappa = nn.Parameter(torch.tensor(3.0))  # strong metaball influence
        self.bias_b = nn.Parameter(torch.tensor(0.5))  # positive bias to offset metaball contribution
        self.delta_raw = nn.Parameter(torch.tensor(-2.5))  # δ = softplus -> around 0.08, much sharper

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
        # Reduce gyroid contribution significantly - focus on metaballs
        Fp = 0.1 * Gq + self.kappa * torch.log1p(Mq) + Hq - self.bias_b
        return Fp, q

    def positional_encoding(self, x, L=10):
        """Encode position with sinusoidal functions for high-frequency details."""
        freq = 2.0 ** torch.linspace(0, L-1, L, device=x.device)
        # x: (N, 3), freq: (L,)
        # result: (N, 3*2*L) = (N, 6*L)
        encoded = []
        for f in freq:
            encoded.append(torch.sin(2 * math.pi * f * x))
            encoded.append(torch.cos(2 * math.pi * f * x))
        return torch.cat(encoded, dim=-1)
    
    def density_and_color(self, p, t, need_normals=True):
        # Make sure p requires grad for normals
        if need_normals and not p.requires_grad:
            p = p.clone().detach().requires_grad_(True)
        
        # Apply positional encoding to enable high-frequency learning
        p_encoded = self.positional_encoding(p, L=6)  # Start with L=6 for reasonable freq

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

        # Spatially-varying color with shading
        ndotl = n @ self.light_dir
        emissive = torch.sin(self.eta * (q @ self.w_vec) + self.zeta * t.squeeze(-1))
        
        # Add position-dependent color variation
        q_color = torch.sin(q * 2.0)  # spatial frequency color modulation
        
        # Make color brighter in high-density regions
        density_boost = torch.clamp(sigma.squeeze() * 5.0, 0, 1).unsqueeze(-1)
        
        rgb = torch.sigmoid(
            self.q0
            + self.q1 * ndotl.unsqueeze(-1)
            + (n @ self.Q2.T)
            + self.q3 * emissive.unsqueeze(-1)
            + q_color * 0.3  # add spatial color variation
            + density_boost * 0.5  # brighten high-density areas
        )
        return sigma.unsqueeze(-1), rgb, Fval, n


# -------------------------------
# Volume rendering (NeRF-style)
# -------------------------------
def render(
    model, rays_o, rays_d, t_scalar, near=0.0, far=8.0, n_samples=96, stratified=True,
    visualize_density=False, visualize_F=False
):
    """
    rays_o/d: (N,3). Returns rgb (N,3), aux dict
    visualize_density: if True, render density as grayscale instead of color
    visualize_F: if True, render signed distance field F as color (red=positive, blue=negative)
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
    Fval_reshaped = Fval.reshape(N, n_samples)
    
    # Override colors for visualization modes
    if visualize_density:
        # Visualize density as grayscale
        density_viz = sigma.squeeze(-1).unsqueeze(-1).expand(-1, -1, 3)
        rgb = torch.clamp(density_viz * 10.0, 0, 1)  # scale up for visibility
    elif visualize_F:
        # Visualize F field: red for positive (outside), blue for negative (inside)
        F_normalized = torch.tanh(Fval_reshaped * 2.0)  # [-1, 1]
        rgb = torch.zeros_like(rgb)
        rgb[:, :, 0] = torch.clamp(F_normalized, 0, 1)   # red for positive
        rgb[:, :, 2] = torch.clamp(-F_normalized, 0, 1)  # blue for negative

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
    
    # Add dark blue background to match typical video style
    acc_alpha = torch.sum(weights, dim=1, keepdim=True)  # (N,1)
    bg_color = torch.tensor([0.15, 0.27, 0.33], device=comp_rgb.device).expand_as(comp_rgb)
    comp_rgb = comp_rgb + (1.0 - acc_alpha) * bg_color
    
    depth = torch.sum(weights * z_vals, dim=1)  # (N,)
    return comp_rgb, {"weights": weights, "depth": depth, "acc_alpha": acc_alpha, "sigma": sigma}


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
    samples_per_ray=64,
    lr=5e-3,
    fov=50.0,
    eikonal_w=0.01,
    device="mps",
):
    if isinstance(device, str):
        if device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")
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
        
        # Sparsity loss: encourage most space to be empty
        # This prevents the model from learning uniform density everywhere
        sparsity_loss = aux['sigma'].mean()
        loss = loss + 0.01 * sparsity_loss
        
        # Temporal variation loss: encourage output to change over time
        # Sample the same pixels at two different times to ensure temporal dynamics
        if it % 2 == 0:  # Every other iteration
            t_other_idx = np.random.randint(0, T)
            t_other = times[t_other_idx].item()
            # Only add temporal loss if times are different
            if abs(t_other - t_now) > 0.1:  # at least 0.1 sec apart
                with torch.no_grad():
                    target_other = video[t_other_idx, pix[:, 0], pix[:, 1]].to(device)
                
                pred_rgb_other, _ = render(
                    model, rays_o, rays_d, t_other,
                    near=0.0, far=8.0, n_samples=samples_per_ray, stratified=True
                )
                
                # Temporal consistency: different times should produce different outputs
                # But the difference should match the target difference
                pred_temporal_diff = (pred_rgb_other - pred_rgb).abs().mean()
                target_temporal_diff = (target_other - target).abs().mean()
                
                # Encourage predicted temporal variation to match target variation
                temporal_loss = (pred_temporal_diff - target_temporal_diff).abs()
                loss = loss + 0.1 * temporal_loss

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
            with torch.no_grad():
                avg_weights = aux['weights'].mean().item()
                max_weights = aux['weights'].max().item()
                acc_alpha_val = aux['acc_alpha'].mean().item()
                
                # Measure actual visual quality on predicted pixels
                pred_std = pred_rgb.std().item()
                target_std = target.std().item()
                contrast_ratio = pred_std / (target_std + 1e-8)
                
                # Measure temporal variation by rendering at t=0 and t=max
                t_start = times[0].item()
                t_end = times[-1].item()
                rgb_start, _ = render(model, rays_o[:512], rays_d[:512], t_start, 
                                     near=0.0, far=8.0, n_samples=32, stratified=False)
                rgb_end, _ = render(model, rays_o[:512], rays_d[:512], t_end,
                                   near=0.0, far=8.0, n_samples=32, stratified=False)
                temporal_var = (rgb_end - rgb_start).abs().mean().item()
                
            print(f"[{it:05d}] loss={loss.item():.6f} | contrast={contrast_ratio:.1%} | temp_var={temporal_var:.5f}")
            
            # Save a single frame preview at higher res for debugging
            if it % 500 == 0:
                with torch.set_grad_enabled(True):
                    preview_H, preview_W = int(H * 0.25), int(W * 0.25)
                    preview_rays_o, preview_rays_d = make_pinhole_rays(preview_H, preview_W, fov_deg=fov)
                    preview_rays_o = preview_rays_o.to(device)
                    preview_rays_d = preview_rays_d.to(device)
                    
                    preview_rgb = torch.zeros(preview_H * preview_W, 3, device=device)
                    chunk = 8192
                    for s in range(0, preview_H * preview_W, chunk):
                        e = min(s + chunk, preview_H * preview_W)
                        rgb_chunk, _ = render(
                            model, preview_rays_o[s:e], preview_rays_d[s:e], 
                            times[0].item(), near=0.0, far=8.0, n_samples=32, stratified=False
                        )
                        preview_rgb[s:e] = rgb_chunk.clamp(0, 1).detach()
                    
                    preview_img = preview_rgb.reshape(preview_H, preview_W, 3).cpu().numpy()
                    preview_bgr = cv2.cvtColor((preview_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"frame_{it:05d}.png", preview_bgr)
                    print(f"  -> Saved frame_{it:05d}.png")
            
            render_full_video(
                model,
                (H, W),
                (rays_o_full, rays_d_full),
                times,
                out_path=f"intermediate_{it:05d}.mp4",
                n_samples=samples_per_ray,
                device=device,
                preview_scale=0.1,
                max_frames=30,
                preview_samples=16,
            )

    return model, (H, W), (rays_o_full, rays_d_full), times


# -------------------------------
# Rendering a reconstructed video
# -------------------------------
def render_full_video(
    model, size_hw, rays_all, times, out_path="recon.mp4", n_samples=128, device="mps",
    preview_scale=1.0, max_frames=None, preview_samples=None
):
    if isinstance(device, str):
        if device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    model.eval()
    H, W = size_hw
    
    # Apply preview scale to resolution
    if preview_scale < 1.0:
        H_render = int(H * preview_scale)
        W_render = int(W * preview_scale)
        rays_o_preview, rays_d_preview = make_pinhole_rays(H_render, W_render, fov_deg=50.0)
        rays_o_preview = rays_o_preview.to(device)
        rays_d_preview = rays_d_preview.to(device)
    else:
        H_render, W_render = H, W
        rays_o_preview, rays_d_preview = rays_all
        # Ensure rays are on correct device
        rays_o_preview = rays_o_preview.to(device)
        rays_d_preview = rays_d_preview.to(device)
    
    # Use preview samples if specified
    render_samples = preview_samples if preview_samples is not None else n_samples
    
    # Limit frames if requested
    times_to_render = times if max_frames is None else times[:max_frames]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, max(1.0, len(times_to_render) / len(times_to_render)), (W_render, H_render)
    )
    print("Rendering video...")
    for ti, t_now in enumerate(times_to_render.tolist()):
        rgb_img = torch.zeros(H_render * W_render, 3, device=device)
        chunk = 8192
        for s in range(0, H_render * W_render, chunk):
            e = min(s + chunk, H_render * W_render)
            with torch.set_grad_enabled(True):
                rgb, _ = render(
                    model,
                    rays_o_preview[s:e],
                    rays_d_preview[s:e],
                    t_now,
                    near=0.0,
                    far=8.0,
                    n_samples=render_samples,
                    stratified=False,
                )
            rgb_img[s:e] = rgb.clamp(0, 1).detach()
        rgb_img = rgb_img.reshape(H_render, W_render, 3).cpu().numpy()
        bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        #print(f"rendered frame {ti + 1}/{len(times_to_render)}")
    writer.release()
    print(f"Wrote {out_path}")


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="path to .mp4")
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument("--render-final", action="store_true", help="Render final high-quality video (slow)")
    parser.add_argument("--final-scale", type=float, default=0.5, help="Resolution scale for final render")
    parser.add_argument("--final-samples", type=int, default=64, help="Samples per ray for final render")
    args = parser.parse_args()

    model, size_hw, rays_all, times = train_on_video(
        args.video, iters=args.iters, device=args.device, fov=args.fov
    )

    if args.render_final:
        print(f"\n{'='*60}")
        print(f"Rendering final video at {args.final_scale*100:.0f}% resolution with {args.final_samples} samples/ray...")
        print(f"This may take a while. Use --final-scale and --final-samples to adjust quality/speed.")
        print(f"{'='*60}\n")
        render_full_video(
            model,
            size_hw,
            rays_all,
            times,
            out_path="reconstruction.mp4",
            device=args.device,
            preview_scale=args.final_scale,
            preview_samples=args.final_samples,
        )
    else:
        print(f"\nTraining complete! Use --render-final to render a high-quality video.")
        print(f"Intermediate videos were saved as intermediate_XXXXX.mp4")
