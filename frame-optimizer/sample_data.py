import numpy as np
import torch

def create_target_images(width=128, height=128) -> tuple[torch.Tensor, torch.Tensor]:
    """Create target images for training"""
    target1 = torch.zeros(height, width, 3)
    target2 = torch.zeros(height, width, 3)

    # Target 1: Circular pattern
    for i in range(height):
        for j in range(width):
            dist = ((i - height / 2) ** 2 + (j - width / 2) ** 2) ** 0.5
            angle = np.arctan2(i - height / 2, j - width / 2)
            if dist < height / 3:
                factor = 0.7 + 0.3 * np.sin(8 * angle)
                if dist < height / 4 * factor:
                    target1[i, j, 0] = 1.0 - 0.3 * np.sin(dist / 5)
                    target1[i, j, 1] = 0.5 + 0.2 * np.cos(dist / 7)
                else:
                    target1[i, j, 2] = 0.7 + 0.3 * np.sin(dist / 8)
            else:
                target1[i, j, 2] = 0.8
                target1[i, j, 1] = 0.2 + 0.1 * np.sin(i / 10) * np.sin(j / 10)

    # Target 2: Grid pattern
    for i in range(height):
        for j in range(width):
            pattern = np.sin(i / 10) * np.sin(j / 10) + np.sin(i / 20 + j / 15)
            if pattern > 0:
                target2[i, j, 1] = 0.7 + 0.3 * pattern
                target2[i, j, 2] = 0.5 + 0.3 * pattern
            else:
                target2[i, j, 0] = 0.7 - 0.5 * pattern
                target2[i, j, 2] = 0.3 - 0.2 * pattern

    return target1, target2
