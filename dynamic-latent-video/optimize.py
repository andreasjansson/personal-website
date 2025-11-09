# optimize.py
import math, cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Video I/O
# ----------------------------
def load_video(path, resize_width=320, max_frames=None):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"open {path} failed"
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if resize_width and w != resize_width:
            r = resize_width / float(w)
            frame = cv2.resize(frame, (resize_width, int(round(h * r))), cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).float() / 255.0)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    vid = torch.stack(frames, 0)  # T,H,W,3
    return vid, float(fps)


def save_video(tensor_T_H_W_3, path, fps):
    T, H, W, _ = tensor_T_H_W_3.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        bgr = cv2.cvtColor(
            (tensor_T_H_W_3[t].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        vw.write(bgr)
    vw.release()


def make_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2)  # (HW,2)


# ----------------------------
# SIREN decoder g
# ----------------------------
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def init_siren_(lin, first=False, w0=30.0, c=6.0):
    with torch.no_grad():
        if first:
            lin.weight.uniform_(-1 / lin.in_features, 1 / lin.in_features)
        else:
            lin.weight.uniform_(
                -math.sqrt(c / lin.in_features) / w0,
                math.sqrt(c / lin.in_features) / w0,
            )


class SIRENDecoder(nn.Module):
    """
    g(x,y; z_t) -> RGB.  Coordinates + FiLM from z_t.
    """

    def __init__(self, dim_z=64, hidden=128, layers=5, w0=30.0):
        super().__init__()
        self.w0 = w0
        self.first = nn.Linear(2, hidden)
        init_siren_(self.first, first=True)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers - 2)]
        )
        for h in self.hiddens:
            init_siren_(h, w0=w0)
        self.to_rgb = nn.Linear(hidden, 3)
        nn.init.xavier_uniform_(self.to_rgb.weight)

        # FiLM from latent z: produce per-layer scale/shift
        self.affines = nn.ModuleList(
            [nn.Linear(dim_z, 2 * hidden) for _ in range(layers - 1)]
        )
        for a in self.affines:
            nn.init.xavier_uniform_(a.weight)
        self.act = Sine(w0)

    def forward(self, coords, z):
        if z.dim() == 1:
            z = z[None, :]
        B = coords.shape[0]
        x = self.first(coords)
        gamma, beta = self.affines[0](z).chunk(2, -1)
        x = self.act(x * gamma + beta)
        for i, layer in enumerate(self.hiddens, start=1):
            x = layer(x)
            gamma, beta = self.affines[i](z).chunk(2, -1)
            x = self.act(x * gamma + beta)
        rgb = torch.sigmoid(self.to_rgb(x))
        return rgb


# ----------------------------
# Structured dynamics f  (port-Hamiltonian step)
# z_{t+1} = z_t + dt * ( J(z) ∇H(z)  - R(z) ∇D(z) + u(z) )
# J is skew-symmetric; R is PSD; H,D >= 0.  u pumps energy toward a target.
# ----------------------------
class PortHamiltonianDynamics(nn.Module):
    def __init__(self, dim=64, hidden=128, dt=0.1, energy_target=1.0):
        super().__init__()
        self.dt_logit = nn.Parameter(torch.tensor(0.0))
        self.dt_max = dt

        # H(z), D(z) positive scalars
        self.Hnet = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.Dnet = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Skew-symmetric field J(z): produce A(z), take A - A^T, apply to grad H
        self.Jgen = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim * dim)
        )
        # Dissipation R(z): produce B(z), use R = B B^T (PSD)
        self.Bgen = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim * dim)
        )

        # tiny control input u(z) to regulate energy around target
        self.unet = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim)
        )
        self.energy_target = nn.Parameter(
            torch.tensor(energy_target), requires_grad=False
        )

    def forward(self, z):
        if self.training:
            z = z.requires_grad_(True)

            # energies
            H = F.softplus(self.Hnet(z)).squeeze(-1)
            D = F.softplus(self.Dnet(z)).squeeze(-1)

            gH = torch.autograd.grad(H.sum(), z, create_graph=True)[0]  # ∇H
            gD = torch.autograd.grad(D.sum(), z, create_graph=True)[0]  # ∇D
        else:
            with torch.enable_grad():
                z_temp = z.requires_grad_(True)
                H = F.softplus(self.Hnet(z_temp)).squeeze(-1)
                D = F.softplus(self.Dnet(z_temp)).squeeze(-1)
                gH = torch.autograd.grad(H.sum(), z_temp, create_graph=False)[0]
                gD = torch.autograd.grad(D.sum(), z_temp, create_graph=False)[0]
            z = z_temp.detach()

        dim = z.shape[-1]
        Jflat = self.Jgen(z)  # (B, dim*dim)
        A = Jflat.view(-1, dim, dim)
        J = A - A.transpose(1, 2)  # skew-symmetric

        Bflat = self.Bgen(z).view(-1, dim, dim)
        R = torch.bmm(Bflat, Bflat.transpose(1, 2))  # PSD

        cons = torch.bmm(J, gH.unsqueeze(-1)).squeeze(-1)  # conservative flow
        diss = -torch.bmm(R, gD.unsqueeze(-1)).squeeze(-1)  # dissipative
        # energy regulation (push toward target)
        u = self.unet(z)
        pump = (self.energy_target - H).unsqueeze(-1) * u  # small control

        dt = torch.sigmoid(self.dt_logit) * self.dt_max
        dz = cons + diss + 0.1 * pump
        return (z + dt * dz).detach() if not self.training else (z + dt * dz)


# ----------------------------
# Wrapper (learn z0, f, g)
# ----------------------------
class LatentVideo(nn.Module):
    def __init__(self, H, W, dim_z=64, decoder_hidden=128, decoder_layers=5, dynamics_hidden=128):
        super().__init__()
        self.f = PortHamiltonianDynamics(dim=dim_z, hidden=dynamics_hidden)
        self.g = SIRENDecoder(dim_z=dim_z, hidden=decoder_hidden, layers=decoder_layers)
        self.z0 = nn.Parameter(torch.zeros(dim_z))
        self.H, self.W = H, W
        self.coords = None

    def rollout(self, T, with_detach=False):
        zs = []
        z = self.z0.unsqueeze(0)  # add batch dim
        for _ in range(T):
            zs.append(z.squeeze(0))
            z = self.f(z)
            if with_detach:
                z = z.detach()
        return torch.stack(zs, 0)  # (T,D)

    def decode_frames(self, zs, chunk=65536, device=None):
        T, D = zs.shape
        H, W = self.H, self.W
        if self.coords is None or self.coords.device != zs.device:
            self.coords = make_grid(H, W, zs.device)
        X = []
        for t in range(T):
            rgb = []
            for s in range(0, H * W, chunk):
                e = min(s + chunk, H * W)
                rgb.append(self.g(self.coords[s:e], zs[t].unsqueeze(0)))
            X.append(torch.cat(rgb, 0).reshape(H, W, 3))
        return torch.stack(X, 0)  # (T,H,W,3)

    def decode_frames_at_resolution(self, zs, H, W, chunk=65536):
        T, D = zs.shape
        coords = make_grid(H, W, zs.device)
        X = []
        for t in range(T):
            rgb = []
            for s in range(0, H * W, chunk):
                e = min(s + chunk, H * W)
                rgb.append(self.g(coords[s:e], zs[t].unsqueeze(0)))
            X.append(torch.cat(rgb, 0).reshape(H, W, 3))
        return torch.stack(X, 0)  # (T,H,W,3)


# ----------------------------
# Loss helpers
# ----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def frame_mse(a, b):
    return ((a - b) ** 2).mean()


# ----------------------------
# Training loop
# ----------------------------
def train(
    mp4_path,
    iters=4000,
    resize_width=80,
    latent_dim=64,
    offset_n=8,
    pixels_per_step=8192,
    lr=2e-3,
    device="mps",
    output_width=None,
    output_height=None,
    decoder_hidden=128,
    decoder_layers=5,
    dynamics_hidden=128,
):
    if device == "cuda":
        device = torch.device(device if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device)

    Y, fps = load_video(mp4_path, resize_width=resize_width)
    T, H, W, _ = Y.shape
    Y = Y.to(device)
    
    # If offset_n not explicitly set, use T for long-term stability
    if offset_n is None:
        offset_n = T
    
    model = LatentVideo(
        H, W, 
        dim_z=latent_dim, 
        decoder_hidden=decoder_hidden, 
        decoder_layers=decoder_layers,
        dynamics_hidden=dynamics_hidden
    ).to(device)

    # Calculate output resolution with aspect ratio handling
    if output_width is None and output_height is None:
        out_H, out_W = H, W
    elif output_width is not None and output_height is None:
        out_W = output_width
        out_H = int(round(out_W * H / W))
    elif output_width is None and output_height is not None:
        out_H = output_height
        out_W = int(round(out_H * W / H))
    else:
        out_W, out_H = output_width, output_height
        expected_H = int(round(out_W * H / W))
        if abs(out_H - expected_H) > 1:
            print(f"Warning: Output aspect ratio {out_W}x{out_H} doesn't match training {W}x{H}")

    print(f"\n=== Model Architecture ===")
    print(f"Training: {T} frames @ {H}x{W}")
    print(f"Output: {T} frames @ {out_H}x{out_W}")
    print(f"Latent dim (Z): {latent_dim}")
    print(f"Offset n: {offset_n} (matching X[t] and X[t+{2*offset_n}] to Y[t])")
    print(f"Decoder g (X): {count_parameters(model.g):,} parameters")
    print(f"Dynamics f (Z): {count_parameters(model.f):,} parameters")
    print(f"Total: {count_parameters(model):,} parameters")
    print(f"Device: {device}\n")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    HW = H * W
    grid = make_grid(H, W, device)
    flatY = Y.reshape(T, HW, 3)

    for it in range(1, iters + 1):
        model.train()
        # Roll out a bit beyond training so X[t+2n] exists
        T_need = min(T + 2 * offset_n, T + 2 * offset_n)
        zs = model.rollout(T_need)  # (T_need, D)

        # sample times ensuring t+2n < T_need and t < T
        t = torch.randint(0, T - 2 * offset_n, (1,), device=device).item()
        idx = torch.randint(0, HW, (pixels_per_step,), device=device)

        # decode only needed frames and pixels
        def decode_pixels(t_idx, z_t):
            # chunked decode to (B,3)
            out = []
            c = 32768
            for s in range(0, pixels_per_step, c):
                e = min(s + c, pixels_per_step)
                out.append(model.g(grid[idx[s:e]], z_t.unsqueeze(0)))
            return torch.cat(out, 0)

        x_t = decode_pixels(t, zs[t])
        x_t2n = decode_pixels(t + 2 * offset_n, zs[t + 2 * offset_n])
        y_t = flatY[t, idx]
        loss_mse_t = frame_mse(x_t, y_t)
        loss_mse_t2n = frame_mse(x_t2n, y_t)
        loss_rec = loss_mse_t + loss_mse_t2n

        # small latent magnitude regularizer
        loss_lat = zs.pow(2).mean() * 1e-4

        loss = loss_rec + loss_lat
        opt.zero_grad(set_to_none=True)
        loss.backward()
        
        if it % 200 == 0:
            # Collect gradient statistics
            grad_stats = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    component = name.split('.')[0]  # f, g, or z0
                    if component not in grad_stats:
                        grad_stats[component] = []
                    grad_stats[component].append(param.grad.abs().mean().item())
            
            print(
                f"[{it:05d}] total={loss.item():.6f} mse_t={loss_mse_t.item():.6f} mse_t+2n={loss_mse_t2n.item():.6f} rec={loss_rec.item():.6f}"
            )
            print(f"        Gradient stats:")
            for comp in ['z0', 'f', 'g']:
                if comp in grad_stats:
                    grads = grad_stats[comp]
                    print(f"          {comp}: mean={sum(grads)/len(grads):.6e}, max={max(grads):.6e}, min={min(grads):.6e}")
            print(f"        z0 vector: mean={model.z0.data.mean().item():.6f}, std={model.z0.data.std().item():.6f}, "
                  f"max={model.z0.data.max().item():.6f}, min={model.z0.data.min().item():.6f}")
        else:
            print(
                f"[{it:05d}] total={loss.item():.6f} mse_t={loss_mse_t.item():.6f} mse_t+2n={loss_mse_t2n.item():.6f} rec={loss_rec.item():.6f}"
            )
        
        opt.step()

        if it % 200 == 0:
            model.eval()
            with torch.no_grad():
                zs_render = model.rollout(T * 3, with_detach=True)
                X_render = model.decode_frames_at_resolution(zs_render, out_H, out_W)
                save_video(X_render, f"checkpoint_iter_{it:05d}.mp4", fps)
                print(f"Saved checkpoint_iter_{it:05d}.mp4 (3x duration: {T*3} frames)")
            model.train()

    return model, fps, T, out_H, out_W


# ----------------------------
# Reconstruct training window and extrapolate
# ----------------------------
@torch.no_grad()
def render_model(model, fps, T_train, T_extra=120, out_path="reconstruction.mp4", out_H=None, out_W=None):
    model.eval()
    zs = model.rollout(T_train + T_extra, with_detach=True)
    if out_H is not None and out_W is not None:
        X = model.decode_frames_at_resolution(zs, out_H, out_W)
    else:
        X = model.decode_frames(zs)
    save_video(X[:T_train], out_path.replace(".mp4", "_fit.mp4"), fps)
    save_video(X[T_train:], out_path.replace(".mp4", "_extrap.mp4"), fps)
    print(
        "wrote:",
        out_path.replace(".mp4", "_fit.mp4"),
        "and",
        out_path.replace(".mp4", "_extrap.mp4"),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Train a latent video model with port-Hamiltonian dynamics and SIREN decoder"
    )
    
    # Input/Output
    p.add_argument("--video", default="input.mp4",
                   help="Path to input video file (default: input.mp4)")
    p.add_argument("--resize_width", type=int, default=80,
                   help="Width to resize input video for training. Height is computed to maintain aspect ratio (default: 80)")
    p.add_argument("--output-width", type=int, default=None,
                   help="Width of output video. If not specified, uses training resolution. SIREN can render at any resolution (default: None)")
    p.add_argument("--output-height", type=int, default=None,
                   help="Height of output video. If only width specified, height computed to maintain aspect ratio (default: None)")
    
    # Training
    p.add_argument("--iters", type=int, default=6000,
                   help="Number of training iterations (default: 6000)")
    p.add_argument("--lr", type=float, default=2e-3,
                   help="Learning rate for Adam optimizer (default: 0.002)")
    p.add_argument("--offset_n", type=int, default=None,
                   help="Frame offset for stability loss: match X[t] and X[t+2n] to same target Y[t]. Default: T (video length) for long-term stability")
    p.add_argument("--device", default="mps",
                   help="Device to train on: 'mps', 'cuda', or 'cpu' (default: mps)")
    
    # Architecture - Latent space
    p.add_argument("--latent_dim", type=int, default=64,
                   help="Dimension of latent vector Z (default: 64)")
    
    # Architecture - Decoder (g: Z -> X)
    p.add_argument("--decoder-hidden", type=int, default=128,
                   help="Hidden size for SIREN decoder network g (default: 128)")
    p.add_argument("--decoder-layers", type=int, default=5,
                   help="Number of layers in SIREN decoder network g (default: 5)")
    
    # Architecture - Dynamics (f: Z -> Z)
    p.add_argument("--dynamics-hidden", type=int, default=128,
                   help="Hidden size for port-Hamiltonian dynamics network f (default: 128)")
    
    args = p.parse_args()

    model, fps, T, out_H, out_W = train(
        args.video,
        iters=args.iters,
        resize_width=args.resize_width,
        latent_dim=args.latent_dim,
        offset_n=args.offset_n,
        lr=args.lr,
        device=args.device,
        output_width=args.output_width,
        output_height=args.output_height,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        dynamics_hidden=args.dynamics_hidden,
    )
    render_model(model, fps, T, T_extra=2*T, out_path="reconstruction.mp4", out_H=out_H, out_W=out_W)
