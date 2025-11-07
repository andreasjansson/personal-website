import torch
import numpy as np
from optimize import BlobbyGyroid

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model = BlobbyGyroid(J=4, N=16, K=8).to(device)
model.eval()

# Sample points in a grid through the viewing volume
print("\nSampling points in viewing volume (z=0 to z=4)...")
N = 1000
pts = torch.randn(N, 3, device=device) * 2.0  # points in [-2,2]^3
t = torch.zeros(N, 1, device=device)

with torch.no_grad():
    # Check each component
    W = model.warp(pts, t)
    q = pts + W
    G = model.gyroid(q)
    M = model.metaballs(q, t)
    H = model.harmonics(q, t)
    F, _ = model.field_F(pts, t)
    
    delta = torch.nn.functional.softplus(model.delta_raw) + 1e-3
    sigma = torch.nn.functional.softplus(-F / delta)

print(f"\nField component statistics:")
print(f"Warp W:      mean={W.mean().item():.4f}, std={W.std().item():.4f}, range=[{W.min().item():.4f}, {W.max().item():.4f}]")
print(f"Gyroid G:    mean={G.mean().item():.4f}, std={G.std().item():.4f}, range=[{G.min().item():.4f}, {G.max().item():.4f}]")
print(f"Metaballs M: mean={M.mean().item():.4f}, std={M.std().item():.4f}, range=[{M.min().item():.4f}, {M.max().item():.4f}]")
print(f"Harmonics H: mean={H.mean().item():.4f}, std={H.std().item():.4f}, range=[{H.min().item():.4f}, {H.max().item():.4f}]")
print(f"κ*log(1+M): mean={(model.kappa * torch.log1p(M)).mean().item():.4f}, std={(model.kappa * torch.log1p(M)).std().item():.4f}")
print(f"\nFinal F:     mean={F.mean().item():.4f}, std={F.std().item():.4f}, range=[{F.min().item():.4f}, {F.max().item():.4f}]")
print(f"Delta δ:     {delta.item():.6f}")
print(f"Density σ:   mean={sigma.mean().item():.4f}, std={sigma.std().item():.4f}, range=[{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
print(f"bias_b:      {model.bias_b.item():.4f}")
print(f"kappa κ:     {model.kappa.item():.4f}")

print(f"\nMetaball centers (mb_cbar):")
print(f"  range: [{model.mb_cbar.min().item():.4f}, {model.mb_cbar.max().item():.4f}]")
print(f"  mean position: {model.mb_cbar.mean(dim=0).cpu().numpy()}")

print(f"\nInterpretation:")
if F.mean() > 1.0:
    print("  ⚠️  F is very positive (outside surface everywhere) - density will be low everywhere")
elif F.mean() < -1.0:
    print("  ⚠️  F is very negative (inside surface everywhere) - density will be high everywhere")
else:
    print("  ✓  F values look reasonable")

if sigma.std() < 0.1:
    print("  ⚠️  Density has very low variation - will produce uniform appearance")
else:
    print(f"  ✓  Density has good variation (std={sigma.std().item():.4f})")
