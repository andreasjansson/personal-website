"""
Training script for the Evolving Field System.
Uses the same rendering and training infrastructure as optimize.py.
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evolving_field import EvolvingFieldSystem
from metaballs_fixed import FixedMetaballs
from optimize import load_video, make_pinhole_rays


def render_video(model, H, W, times, device, out_path, preview_scale=0.1, n_samples=16):
    """Render a quick preview video."""
    model.eval()
    
    H_render = int(H * preview_scale)
    W_render = int(W * preview_scale)
    
    rays_o, rays_d = make_pinhole_rays(H_render, W_render, fov_deg=50.0)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 10.0, (W_render, H_render))
    
    for ti, t_now in enumerate(times.tolist()):
        rgb_img = torch.zeros(H_render * W_render, 3, device=device)
        chunk = 4096
        for s in range(0, H_render * W_render, chunk):
            e = min(s + chunk, H_render * W_render)
            with torch.set_grad_enabled(True):  # Need grad for model dynamics
                rgb, _ = render(
                    model, rays_o[s:e], rays_d[s:e], t_now,
                    near=0.0, far=8.0, n_samples=n_samples, stratified=False
                )
            rgb_img[s:e] = rgb.clamp(0, 1).detach()
        
        rgb_img = rgb_img.reshape(H_render, W_render, 3).cpu().numpy()
        bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    
    writer.release()
    print(f"Wrote {out_path}")
    model.train()

def render(model, rays_o, rays_d, t_scalar, near=0.0, far=8.0, n_samples=64, stratified=True):
    """Volume rendering for evolving field system."""
    device = rays_o.device
    N = rays_o.shape[0]

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

    sigma, rgb = model(pts, t_scalar)
    sigma = sigma.reshape(N, n_samples, 1)
    rgb = rgb.reshape(N, n_samples, 3)

    # Volume rendering
    dists = torch.cat(
        [z_vals[:, 1:] - z_vals[:, :-1], torch.full((N, 1), 1e10, device=device)], dim=1
    )
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
    T = torch.cumprod(
        torch.cat([torch.ones((N, 1), device=device), 1.0 - alpha + 1e-10], dim=1),
        dim=1,
    )[:, :-1]
    weights = alpha * T
    comp_rgb = torch.sum(weights[..., None] * rgb, dim=1)
    
    # Dark background
    acc_alpha = torch.sum(weights, dim=1, keepdim=True)
    bg_color = torch.tensor([0.15, 0.27, 0.33], device=device).expand_as(comp_rgb)
    comp_rgb = comp_rgb + (1.0 - acc_alpha) * bg_color
    
    return comp_rgb, {"weights": weights, "sigma": sigma, "acc_alpha": acc_alpha}


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load video
    video, fps = load_video(args.video)
    T, H, W, _ = video.shape
    print(f"Loaded video: {T} frames, {W}x{H}, fps={fps:.2f}")
    times = torch.arange(T, dtype=torch.float32) / float(fps)
    
    # Create model
    if args.model_type == 'evolving':
        model = EvolvingFieldSystem(
            n_blobs=args.n_blobs,
            hidden_dynamics=args.hidden_dynamics,
            hidden_query=args.hidden_query,
            hidden_summary=args.hidden_summary,
            pos_enc_L=args.pos_enc_L,
        ).to(device)
    elif args.model_type == 'metaballs':
        model = FixedMetaballs(
            n_balls=args.n_blobs,
            pos_enc_L=args.pos_enc_L,
            hidden=args.hidden_query if args.hidden_query > 0 else 64,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Camera rays
    rays_o_full, rays_d_full = make_pinhole_rays(H, W, fov_deg=args.fov)
    rays_o_full = rays_o_full.to(device)
    rays_d_full = rays_d_full.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Pixel coords for sampling
    all_pix = torch.stack(
        torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij"), dim=-1
    ).reshape(-1, 2)
    
    for it in range(1, args.iters + 1):
        model.train()
        
        # Random frame and pixels
        fidx = np.random.randint(0, T)
        t_now = times[fidx].item()
        sel = torch.randint(0, H * W, (args.rays_per_step,))
        pix = all_pix[sel]
        target = video[fidx, pix[:, 0], pix[:, 1]].to(device)
        
        rays_o = rays_o_full[sel]
        rays_d = rays_d_full[sel]
        
        # Render
        pred_rgb, aux = render(
            model, rays_o, rays_d, t_now,
            near=0.0, far=8.0, n_samples=args.samples_per_ray, stratified=True
        )
        
        loss = F.mse_loss(pred_rgb, target)
        
        # Sparsity loss
        sparsity_loss = aux['sigma'].mean()
        loss = loss + 0.01 * sparsity_loss
        
        # Temporal variation loss - compare two different times
        t_other_idx = (fidx + np.random.randint(1, max(2, T//4))) % T
        t_other = times[t_other_idx].item()
        with torch.no_grad():
            target_other = video[t_other_idx, pix[:, 0], pix[:, 1]].to(device)
        
        pred_rgb_other, _ = render(
            model, rays_o, rays_d, t_other,
            near=0.0, far=8.0, n_samples=args.samples_per_ray, stratified=True
        )
        
        pred_temporal_diff = (pred_rgb_other - pred_rgb).abs().mean()
        target_temporal_diff = (target_other - target).abs().mean()
        temporal_loss = (pred_temporal_diff - target_temporal_diff).abs()
        loss = loss + 0.5 * temporal_loss
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if it % 100 == 0:
            with torch.no_grad():
                pred_std = pred_rgb.std().item()
                target_std = target.std().item()
                contrast_ratio = pred_std / (target_std + 1e-8)
            print(f"[{it:05d}] loss={loss.item():.6f} | contrast={contrast_ratio:.1%} | temp_diff={pred_temporal_diff.item():.5f}")
            
            # Render intermediate video
            print("Rendering video...")
            render_video(model, H, W, times[:30], device, f"intermediate_{it:05d}.mp4")
        
        # Save checkpoint
        if it % 500 == 0:
            torch.save({
                'iteration': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_{it:05d}.pt')
            print(f"  Saved checkpoint_{it:05d}.pt")
    
    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--n-blobs", type=int, default=4, help="Number of evolving blobs")
    parser.add_argument("--hidden-dynamics", type=int, default=0, help="Hidden size for dynamics net (0=disabled)")
    parser.add_argument("--hidden-query", type=int, default=0, help="Hidden size for query net (0=minimal)")
    parser.add_argument("--hidden-summary", type=int, default=0, help="Hidden size for blob summary (0=use raw)")
    parser.add_argument("--pos-enc-L", type=int, default=2, help="Positional encoding frequency levels")
    parser.add_argument("--rays-per-step", type=int, default=2048)
    parser.add_argument("--samples-per-ray", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--fov", type=float, default=50.0)
    args = parser.parse_args()
    
    model = train(args)
