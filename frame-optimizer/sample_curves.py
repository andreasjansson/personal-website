import torch
import numpy as np
import math


def stepped_curve(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.linspace(0, 1, n)
    y = np.zeros_like(x)

    # Target y-values for each x range
    target_values = [
        (0.0, 0.1, 0),  # x<0.1 → y≈0
        (0.1, 0.2, 2),  # 0.1≤x<0.2 → y≈3
        (0.2, 0.3, 1),  # 0.2≤x<0.3 → y≈2
        (0.3, 0.4, 0),  # 0.3≤x<0.4 → y≈1
        (0.4, 0.5, 3),  # 0.4≤x<0.5 → y≈4
        (0.5, 0.6, 1),  # 0.5≤x<0.6 → y≈2
        (0.6, 0.7, 2),  # 0.6≤x<0.7 → y≈3
        (0.7, 0.8, 0),  # 0.7≤x<0.8 → y≈1
        (0.8, 0.9, 1),  # 0.8≤x<0.9 → y≈2
        (0.9, 1.0, 0),  # 0.9≤x<1.0 → y≈0
    ]

    # Set y values based on x ranges
    for i, x_val in enumerate(x):
        # Find which range this x value falls into
        for x_min, x_max, target in target_values:
            if x_min <= x_val < x_max:
                # Scale the base function to be centered around the target
                # Map to desired range, centering around target
                y[i] = target
                break

    return torch.tensor(x), torch.tensor(y)


def smooth_curve(n=100, include_noise=False, noise_level=0.05):
    # Create evenly spaced x values from 0 to 1
    x_values = torch.linspace(0, 1, n)

    # Create a sine wave with multiple frequencies
    # Scale it to range [1, 5]
    base_signal = (
        torch.sin(x_values * 2 * np.pi) * 0.8  # Main frequency
        + torch.sin(x_values * 3 * np.pi) * 1.2  # Higher frequency component
        + torch.sin(x_values * 7 * np.pi) * 2.1  # Higher frequency component
        + torch.sin(x_values * 8 * np.pi) * 1.2  # Higher frequency component
        + torch.sin(x_values * 16 * np.pi + 0.1) * 1.2  # Higher frequency component
        + torch.tan(x_values * 1 * np.pi) * 0.1  # Lower frequency with phase shift
    )

    # Normalize to [-1, 1]
    base_signal = base_signal / base_signal.abs().max()

    # Scale to [1, 5]
    y_values = base_signal * 2.0 + 3.0

    # Add noise if requested
    if include_noise:
        amplitude = 4.0  # Range max - min
        noise = torch.randn(n) * (amplitude * noise_level)
        y_values = y_values + noise

        # Ensure values stay in [1, 5] range after noise
        y_values = torch.clamp(y_values, 1.0, 5.0)

    return x_values, y_values


# Example with a non-contiguous target curve
def random_curve(n=50, noise_level=0.2):
    """Create a sample target curve with noise and gaps"""
    # Create non-uniformly spaced x values with gaps
    np.random.seed(42)  # For reproducibility
    x_values = np.sort(np.random.rand(n) * 0.8 + 0.1)  # Range [0.1, 0.9]

    # Create clusters with gaps
    mask = np.ones_like(x_values, dtype=bool)
    # Create gap regions
    gap_regions = [(0.2, 0.25), (0.4, 0.5), (0.7, 0.75)]
    for start, end in gap_regions:
        mask = mask & ~((x_values >= start) & (x_values <= end))

    x_values = x_values[mask]

    # Create y values with multiple frequency components + noise
    base_signal = (
        np.sin(x_values * 20) * 0.5
        + np.sin(x_values * 7) * 1.0
        + np.sin(x_values * 2) * 1.5
    )

    # Add noise
    noise = np.random.randn(len(x_values)) * noise_level
    y_values = base_signal + noise

    # Manually add some outliers
    outlier_indices = np.random.choice(len(y_values), 3, replace=False)
    y_values[outlier_indices] += np.random.randn(3) * 1.0

    return torch.tensor(x_values, dtype=torch.float32), torch.tensor(
        y_values, dtype=torch.float32
    )


def generate_target_image(
    height: int = 100,
    width: int = 100,
    num_values: int = 3,
    pattern_type: str = "circles",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generate a target image with a specific pattern.

    Args:
        height: Image height
        width: Image width
        num_values: Number of distinct values (classes)
        pattern_type: Type of pattern ('circles', 'squares', 'stripes')
        device: Computation device

    Returns:
        Tensor of shape [height, width] with class indices
    """
    if pattern_type == "circles":
        # Create concentric circles
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        distance = torch.sqrt(x**2 + y**2)
        # Create rings
        target = (distance * num_values).floor().remainder(num_values).long()

    elif pattern_type == "squares":
        # Create nested squares
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        # Use max distance from center (L∞ norm)
        distance = torch.maximum(torch.abs(x), torch.abs(y))
        target = (distance * num_values * 1.5).floor().remainder(num_values).long()

    elif pattern_type == "stripes":
        # Create diagonal stripes
        y, x = torch.meshgrid(
            torch.linspace(0, num_values, height, device=device),
            torch.linspace(0, num_values, width, device=device),
            indexing="ij",
        )
        target = ((x + y) % num_values).long()

    elif pattern_type == "checkerboard":
        # Create checkerboard pattern
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        block_size = min(height, width) // (num_values * 2)
        if block_size < 1:
            block_size = 1
        target = (((y // block_size) + (x // block_size)) % num_values).long()

    else:
        # Random pattern as fallback
        target = torch.randint(0, num_values, (height, width), device=device)

    return target
