import torch
import cv2
import numpy as np
from optimize import BlobbyGyroid, make_pinhole_rays, render

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Create a model
model = BlobbyGyroid(J=4, N=16, K=8).to(device)
model.eval()

# Create camera rays
H, W = 360, 360
rays_o, rays_d = make_pinhole_rays(H, W, fov_deg=50.0)
rays_o = rays_o.to(device)
rays_d = rays_d.to(device)

t_now = 0.0

# Render normal RGB
print("Rendering RGB...")
rgb_img = torch.zeros(H * W, 3, device=device)
chunk = 4096
with torch.set_grad_enabled(True):
    for s in range(0, H * W, chunk):
        e = min(s + chunk, H * W)
        rgb, aux = render(
            model, rays_o[s:e], rays_d[s:e], t_now,
            near=0.0, far=8.0, n_samples=64, stratified=False,
            visualize_density=False, visualize_F=False
        )
        rgb_img[s:e] = rgb.clamp(0, 1).detach()

rgb_img = rgb_img.reshape(H, W, 3).cpu().numpy()
bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("debug_rgb.png", bgr)
print("Wrote debug_rgb.png")

# Render density visualization
print("Rendering density...")
density_img = torch.zeros(H * W, 3, device=device)
with torch.set_grad_enabled(True):
    for s in range(0, H * W, chunk):
        e = min(s + chunk, H * W)
        rgb, aux = render(
            model, rays_o[s:e], rays_d[s:e], t_now,
            near=0.0, far=8.0, n_samples=64, stratified=False,
            visualize_density=True
        )
        density_img[s:e] = rgb.clamp(0, 1).detach()

density_img = density_img.reshape(H, W, 3).cpu().numpy()
bgr_density = cv2.cvtColor((density_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("debug_density.png", bgr_density)
print("Wrote debug_density.png")

# Render F field visualization
print("Rendering F field...")
f_img = torch.zeros(H * W, 3, device=device)
with torch.set_grad_enabled(True):
    for s in range(0, H * W, chunk):
        e = min(s + chunk, H * W)
        rgb, aux = render(
            model, rays_o[s:e], rays_d[s:e], t_now,
            near=0.0, far=8.0, n_samples=64, stratified=False,
            visualize_F=True
        )
        f_img[s:e] = rgb.clamp(0, 1).detach()

f_img = f_img.reshape(H, W, 3).cpu().numpy()
bgr_f = cv2.cvtColor((f_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite("debug_F.png", bgr_f)
print("Wrote debug_F.png")

print("\nVisualization guide:")
print("- debug_rgb.png: Normal rendered output with colors")
print("- debug_density.png: Density field (brighter = higher density)")
print("- debug_F.png: Signed distance field (red = outside/positive, blue = inside/negative)")
