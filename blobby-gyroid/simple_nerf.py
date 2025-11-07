"""
Simplified NeRF model that prioritizes sharp features and temporal variation.
The original BlobbyGyroid model is too smooth - this uses pure MLP with positional encoding.
"""

import argparse
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimize import load_video, make_pinhole_rays

class SimpleNeRF(nn.Module):
    """Simple NeRF with positional encoding for sharp features."""
    
    def __init__(self):
        super().__init__()
        
        # Positional encoding: L=10 for position, L=4 for time
        # Position: 3 * 2 * 10 = 60 dimensions
        # Time: 1 * 2 * 4 = 8 dimensions
        # Total input: 60 + 8 = 68
        
        # Density network
        self.density_net = nn.Sequential(
            nn.Linear(68, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        # Color network (takes position + density feature)
        self.color_net = nn.Sequential(
            nn.Linear(68 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )
        
    def positional_encoding(self, x, L):
        """Positional encoding."""
        freq = 2.0 ** torch.linspace(0, L-1, L, device=x.device)
        encoded = []
        for f in freq:
            encoded.append(torch.sin(2 * math.pi * f * x))
            encoded.append(torch.cos(2 * math.pi * f * x))
        return torch.cat(encoded, dim=-1)
    
    def forward(self, p, t):
        """
        p: (N, 3) positions
        t: (N, 1) times
        Returns: sigma (N, 1), rgb (N, 3)
        """
        # Encode position and time
        p_enc = self.positional_encoding(p, L=10)
        t_enc = self.positional_encoding(t, L=4)
        
        # Concatenate
        x = torch.cat([p_enc, t_enc], dim=-1)
        
        # Compute density
        density_features = self.density_net[:-1](x)  # Get features before final layer
        sigma = F.softplus(self.density_net[-1](density_features))
        
        # Compute color
        color_input = torch.cat([x, density_features], dim=-1)
        rgb = self.color_net(color_input)
        
        return sigma, rgb

print("Simple NeRF model created - focuses on sharp features via positional encoding")
print("Run this with the same training loop as optimize.py")
