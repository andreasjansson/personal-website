"""
Fixed Metaball model incorporating all our learnings:

1. Positional encoding for high-frequency spatial detail
2. Rich temporal evolution (not just sin(t), but coupled oscillators)
3. Many metaballs with complex interactions
4. Direct spatial query network for sharpness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FixedMetaballs(nn.Module):
    def __init__(self, n_balls=20, pos_enc_L=6, hidden=128):
        super().__init__()
        
        self.n_balls = n_balls
        self.pos_enc_L = pos_enc_L
        pos_enc_dim = 3 * 2 * pos_enc_L
        
        # Each metaball has:
        # - Base position (3)
        # - Oscillation amplitudes (3) 
        # - Frequencies for position (3)
        # - Phase offsets (3)
        # - Size oscillation params (amplitude, freq, phase)
        # - Intensity oscillation params (amplitude, freq, phase)
        
        self.ball_base_pos = nn.Parameter(torch.randn(n_balls, 3) * 1.5)
        self.ball_osc_amp = nn.Parameter(torch.randn(n_balls, 3) * 0.3)
        self.ball_freq = nn.Parameter(torch.abs(torch.randn(n_balls, 3)) * 2.0 + 0.5)
        self.ball_phase = nn.Parameter(torch.randn(n_balls, 3) * math.pi)
        
        # Coupling between balls (like coupled oscillators)
        self.coupling = nn.Parameter(torch.randn(n_balls, n_balls) * 0.1)
        
        # Size evolution
        self.ball_size_base = nn.Parameter(torch.ones(n_balls) * 0.8)
        self.ball_size_amp = nn.Parameter(torch.randn(n_balls) * 0.2)
        self.ball_size_freq = nn.Parameter(torch.abs(torch.randn(n_balls)) * 1.0 + 0.3)
        self.ball_size_phase = nn.Parameter(torch.randn(n_balls) * math.pi)
        
        # Intensity
        self.ball_intensity_base = nn.Parameter(torch.ones(n_balls) * 1.0)
        self.ball_intensity_amp = nn.Parameter(torch.randn(n_balls) * 0.3)
        self.ball_intensity_freq = nn.Parameter(torch.abs(torch.randn(n_balls)) * 1.5 + 0.4)
        self.ball_intensity_phase = nn.Parameter(torch.randn(n_balls) * math.pi)
        
        # Field query network: pos_enc(p) + ball_positions + ball_sizes -> density
        # This adds high-frequency spatial detail
        self.density_net = nn.Sequential(
            nn.Linear(pos_enc_dim + n_balls * 3 + n_balls, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        
        # Color network
        self.color_net = nn.Sequential(
            nn.Linear(pos_enc_dim + n_balls * 3 + n_balls + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )
        
    def positional_encoding(self, x, L=6):
        """Positional encoding for high-frequency detail."""
        freq = 2.0 ** torch.linspace(0, L-1, L, device=x.device)
        encoded = []
        for f in freq:
            encoded.append(torch.sin(2 * math.pi * f * x))
            encoded.append(torch.cos(2 * math.pi * f * x))
        return torch.cat(encoded, dim=-1)
    
    def get_ball_positions(self, t):
        """
        Get metaball positions at time t.
        Each ball oscillates with its own frequency + coupling effects.
        """
        # Base oscillation
        positions = self.ball_base_pos + self.ball_osc_amp * torch.sin(
            self.ball_freq * t + self.ball_phase
        )
        
        # Add coupling effects (balls influence each other's motion)
        # This creates more complex patterns than independent oscillation
        coupling_effect = torch.matmul(self.coupling, positions)
        positions = positions + 0.1 * coupling_effect
        
        return positions  # (n_balls, 3)
    
    def get_ball_sizes(self, t):
        """Get metaball sizes at time t."""
        sizes = self.ball_size_base + self.ball_size_amp * torch.sin(
            self.ball_size_freq * t + self.ball_size_phase
        )
        return F.softplus(sizes)  # (n_balls,)
    
    def get_ball_intensities(self, t):
        """Get metaball intensities at time t."""
        intensities = self.ball_intensity_base + self.ball_intensity_amp * torch.sin(
            self.ball_intensity_freq * t + self.ball_intensity_phase
        )
        return torch.sigmoid(intensities)  # (n_balls,)
    
    def forward(self, p, t):
        """
        p: (N, 3) positions
        t: scalar or (N, 1) time
        """
        # Handle time input
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=p.dtype, device=p.device)
        if t.dim() > 0:
            t = t.flatten()[0].item()
        
        # Get ball states at time t
        ball_pos = self.get_ball_positions(t)  # (n_balls, 3)
        ball_sizes = self.get_ball_sizes(t)  # (n_balls,)
        ball_intensities = self.get_ball_intensities(t)  # (n_balls,)
        
        # Positional encoding of query points
        p_enc = self.positional_encoding(p, L=self.pos_enc_L)  # (N, pos_enc_dim)
        
        # Broadcast ball info to all query points
        ball_pos_flat = ball_pos.reshape(-1)  # (n_balls*3,)
        ball_pos_exp = ball_pos_flat[None, :].expand(p.shape[0], -1)  # (N, n_balls*3)
        ball_sizes_exp = ball_sizes[None, :].expand(p.shape[0], -1)  # (N, n_balls)
        
        # Density query
        density_input = torch.cat([p_enc, ball_pos_exp, ball_sizes_exp], dim=-1)
        density_raw = self.density_net(density_input)
        density = F.softplus(density_raw)
        
        # Color query
        color_input = torch.cat([p_enc, ball_pos_exp, ball_sizes_exp, density], dim=-1)
        color = self.color_net(color_input)
        
        return density, color

print("Fixed Metaball Model")
print("- Explicit coupled oscillator dynamics (X[t] = f(t) with learnable params)")
print("- Positional encoding for high-frequency spatial detail")  
print("- Ball-ball coupling for complex motion patterns")
print("- Network adds detail on top of metaball structure")
