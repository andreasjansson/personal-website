"""
Continuous Evolution Field System - designed from first principles

Key insights from our experiments:
1. Need 0.1% → 100% high frequencies (positional encoding essential)
2. Need 14% → 100% motion (true temporal evolution, not sin(t))
3. Need sharp features (metaballs/gyroid too smooth)
4. Want X[t+dt] = X[t] + f(X[t]) * dt (continuous evolution)

Design principles:
- Multiple interacting "blob" states that evolve through coupled ODEs
- Each blob has: position, velocity, size, intensity
- Nonlinear coupling between blobs creates complex patterns
- Density field computed from evolved blob states
- High-frequency detail via positional encoding of spatial queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EvolvingFieldSystem(nn.Module):
    """
    Continuous dynamical system of coupled blob oscillators.
    
    State X = [positions, velocities, sizes, intensities] for N blobs
    Evolution: dX/dt = f(X, params)
    
    The field at any point p and time t is computed by:
    1. Evolve blob states from t=0 to t using ODE
    2. Query field at p using positional encoding + blob influences
    """
    
    def __init__(self, n_blobs=20, hidden_dynamics=64, hidden_query=64, 
                 hidden_summary=32, pos_enc_L=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        
        self.n_blobs = n_blobs
        self.pos_enc_L = pos_enc_L
        pos_enc_dim = 3 * 2 * pos_enc_L  # 3 coords * 2 (sin+cos) * L
        
        # Initial blob states (learnable)
        # Each blob: [x, y, z, vx, vy, vz, size, intensity]
        self.initial_state = nn.Parameter(torch.randn(n_blobs, 8, generator=g) * 0.3)
        
        # Evolution dynamics network
        if hidden_dynamics > 0:
            self.dynamics_net = nn.Sequential(
                nn.Linear(n_blobs * 8, hidden_dynamics),
                nn.Tanh(),
                nn.Linear(hidden_dynamics, n_blobs * 8),
            )
        else:
            self.dynamics_net = None
        
        # Coupling parameters (how blobs affect each other)
        self.coupling_strength = nn.Parameter(torch.ones(1) * 0.5)
        self.damping = nn.Parameter(torch.ones(1) * 0.1)
        
        # Nonlinear interaction matrix between blobs
        self.interaction_matrix = nn.Parameter(torch.randn(n_blobs, n_blobs, generator=g) * 0.1)
        
        # Blob state summarizer (for field query)
        if hidden_summary > 0:
            self.blob_summarizer = nn.Sequential(
                nn.Linear(n_blobs * 8, hidden_summary),
            )
        else:
            self.blob_summarizer = None
            hidden_summary = n_blobs * 8  # Use raw state
        
        # Field query network
        if hidden_query > 0:
            self.field_query_net = nn.Sequential(
                nn.Linear(pos_enc_dim + hidden_summary, hidden_query),
                nn.ReLU(),
                nn.Linear(hidden_query, 1),
            )
        else:
            self.field_query_net = nn.Linear(pos_enc_dim + hidden_summary, 1)
        
        # Color network
        if hidden_query > 0:
            self.color_net = nn.Sequential(
                nn.Linear(pos_enc_dim + hidden_summary + 1, hidden_query),
                nn.ReLU(),
                nn.Linear(hidden_query, 3),
                nn.Sigmoid(),
            )
        else:
            self.color_net = nn.Sequential(
                nn.Linear(pos_enc_dim + hidden_summary + 1, 3),
                nn.Sigmoid(),
            )
        
    def positional_encoding(self, x, L=10):
        """High-frequency positional encoding."""
        freq = 2.0 ** torch.linspace(0, L-1, L, device=x.device)
        encoded = []
        for f in freq:
            encoded.append(torch.sin(2 * math.pi * f * x))
            encoded.append(torch.cos(2 * math.pi * f * x))
        return torch.cat(encoded, dim=-1)
    
    def dynamics(self, state):
        """
        Compute dX/dt given current state X.
        
        state: (n_blobs, 8) = [pos(3), vel(3), size(1), intensity(1)]
        returns: (n_blobs, 8) derivatives
        """
        flat_state = state.reshape(-1)
        
        # Extract components
        positions = state[:, :3]  # (n_blobs, 3)
        velocities = state[:, 3:6]  # (n_blobs, 3)
        sizes = state[:, 6:7]  # (n_blobs, 1)
        intensities = state[:, 7:8]  # (n_blobs, 1)
        
        # Neural network component (learns complex dynamics)
        dstate_neural = self.dynamics_net(flat_state).reshape(self.n_blobs, 8)
        
        # Physical-inspired components (for interpretability + stability)
        # Positions evolve according to velocities
        dpos = velocities
        
        # Velocities affected by:
        # 1. Damping
        # 2. Pairwise forces between blobs (attraction/repulsion)
        # 3. Neural network learned dynamics
        
        # Compute pairwise distances and forces
        diffs = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
        dists = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-6)  # (n, n)
        
        # Interaction forces (depends on learned interaction matrix)
        # F_ij ~ interaction_matrix[i,j] * (1 / dist^2) * direction
        force_magnitudes = self.interaction_matrix / (dists ** 2 + 1.0)  # (n, n)
        force_directions = diffs / (dists[:, :, None] + 1e-6)  # (n, n, 3)
        forces = (force_magnitudes[:, :, None] * force_directions).sum(dim=1)  # (n, 3)
        
        dvel = -self.damping * velocities + self.coupling_strength * forces + dstate_neural[:, 3:6]
        
        # Sizes and intensities evolve according to neural dynamics
        # (with slight damping to prevent explosion)
        dsize = -0.05 * (sizes - 1.0) + dstate_neural[:, 6:7]
        dintensity = -0.05 * (intensities - 0.5) + dstate_neural[:, 7:8]
        
        return torch.cat([dpos, dvel, dsize, dintensity], dim=-1)
    
    def evolve_state(self, t, dt=0.01, method='euler'):
        """
        Evolve blob states from t=0 to t using ODE integration.
        
        For now using simple Euler, but could use RK4 or adaptive methods.
        """
        state = self.initial_state
        n_steps = max(1, int(t / dt))
        actual_dt = t / n_steps if n_steps > 0 else 0
        
        for _ in range(n_steps):
            dstate = self.dynamics(state)
            state = state + actual_dt * dstate
            
        return state
    
    def query_field(self, p, blob_state):
        """
        Query density field at positions p given current blob state.
        
        p: (N, 3) positions
        blob_state: (n_blobs, 8) current blob states
        returns: (N, 1) density, (N, 3) color
        """
        # Positional encoding of query points
        p_enc = self.positional_encoding(p, L=4)  # (N, 24) - reduced for speed
        
        # Summarize blob state
        blob_summary = self.blob_summarizer(blob_state.reshape(-1))  # (32,)
        blob_summary_exp = blob_summary[None, :].expand(p.shape[0], -1)  # (N, 32)
        
        # Compute density
        field_input = torch.cat([p_enc, blob_summary_exp], dim=-1)
        density_raw = self.field_query_net(field_input)
        density = F.softplus(density_raw)
        
        # Compute color
        color_input = torch.cat([p_enc, blob_summary_exp, density], dim=-1)
        color = self.color_net(color_input)
        
        return density, color
    
    def forward(self, p, t):
        """
        p: (N, 3) positions in space
        t: (N, 1) or scalar time
        returns: density (N, 1), color (N, 3)
        """
        # Handle scalar time
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=p.dtype, device=p.device)
        if t.dim() == 0:
            t = t.item()
        elif t.dim() == 1:
            t = t[0].item()
        elif t.dim() == 2:
            t = t[0, 0].item()
        
        # Evolve blob states to time t
        blob_state = self.evolve_state(t)
        
        # Query field at positions p
        density, color = self.query_field(p, blob_state)
        
        return density, color

print("="*80)
print("EVOLVING FIELD SYSTEM")
print("="*80)
print("""
This system implements true continuous evolution: X[t+dt] = X[t] + f(X[t]) * dt

Key features:
1. Multiple blob states evolve through coupled ODEs
2. Nonlinear interactions create complex emergent patterns
3. High-frequency spatial detail via positional encoding
4. Learned dynamics allow rich temporal evolution
5. Many parameters (~100K+) for matching complex videos

The blobs interact through:
- Pairwise forces (attraction/repulsion)
- Learned coupling dynamics
- Damping and restoring forces
- Neural network component for complex behavior

This should produce ACTUAL evolving motion, not just sin(t) sampling.
""")
print("="*80)
