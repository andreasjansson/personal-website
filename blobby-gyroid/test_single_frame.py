import torch
import cv2
import numpy as np
from optimize import BlobbyGyroid, make_pinhole_rays, render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a simple model
model = BlobbyGyroid(J=4, N=16, K=8).to(device)
model.eval()

# Create camera rays
H, W = 256, 256
rays_o, rays_d = make_pinhole_rays(H, W, fov_deg=50.0)
rays_o = rays_o.to(device)
rays_d = rays_d.to(device)

# Render at t=0
rgb_img = torch.zeros(H * W, 3, device=device)
chunk = 4096
t_now = 0.0

with torch.set_grad_enabled(True):
    for s in range(0, H * W, chunk):
        e = min(s + chunk, H * W)
        rgb, aux = render(
            model, rays_o[s:e], rays_d[s:e], t_now,
            near=0.0, far=8.0, n_samples=64, stratified=False
        )
        rgb_img[s:e] = rgb.clamp(0, 1).detach()
        if s == 0:
            print(f"Sample stats:")
            print(f"  weights: avg={aux['weights'].mean().item():.4f}, max={aux['weights'].max().item():.4f}")
            print(f"  alpha: {aux['acc_alpha'].mean().item():.4f}")
            print(f"  rgb sample: {rgb[0].detach().cpu().numpy()}")

rgb_img = rgb_img.reshape(H, W, 3).cpu().numpy()
bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("test_render.png", bgr)
print(f"Wrote test_render.png")
