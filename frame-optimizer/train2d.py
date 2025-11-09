import time
import urllib.parse
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import requests

from fractal2d_nondiff import Fractal2DNonDiff
from fractal2d_diff import Fractal2DDiff
from sa import SimulatedAnnealing
from optimizer import Optimizer, OptimizerPart
from ppo import PPOOptimizer


def optimize_fractal2d(
    model: Fractal2DNonDiff,
    max_depth: int,
    target_image: torch.Tensor,
    num_iterations: int = 2000,
    sa_initial_temp: float = 1.0,
    sa_min_temp: float = 0.1,
    sa_num_restarts: int = 3,
    sa_step_size: float = 0.1,
    color_map: np.ndarray = None,
    plot_every_n: int = 200,
) -> Fractal2DNonDiff:
    # Move target image to device
    target_image = target_image.to(device=model.device, dtype=torch.long)

    optimizer = SimulatedAnnealing(
        model.parameters(),
        num_iterations=num_iterations,
        initial_temp=sa_initial_temp,
        num_restarts=sa_num_restarts,
        min_temp=sa_min_temp,
        step_size=sa_step_size,
    )
    # optimizer = PPOOptimizer(model.parameters(), num_iterations=num_iterations)

    best_loss = float("inf")
    best_state = {}

    def closure():
        # optimizer.zero_grad()
        predicted_image = model(max_depth)
        loss = compute_loss(predicted_image, target_image, model.num_classes)
        return loss

    def plot_fn():
        model.plot_history(
            depths=[max_depth - 3, max_depth, max_depth + 2],
            target_image=target_image,
            color_map=color_map,
        )

    times_per_iteration = []
    for i in range(num_iterations):
        t = time.time()
        loss = optimizer.step(closure, plot_fn)
        times_per_iteration.append(time.time() - t)

        model.track_iteration(
            iteration=i,
            recon_loss=loss,
        )

        if i % plot_every_n == 0 or i == num_iterations - 1:
            print(
                f"Iteration {i}/{num_iterations}, Loss: {loss.item():.6f}, time per iteration: {np.mean(times_per_iteration):.4f}s"
            )
            plot_fn()
            times_per_iteration = []

        if loss.item() < best_loss:
            print("Best loss!")
            plot_fn()
            best_loss = loss.item()

    return model


def compute_loss(predicted: torch.Tensor, target: torch.Tensor, num_classes):
    # Simple loss based on accuracy
    recon_loss = 1 - (torch.sum(predicted == target) / torch.numel(target))
    edge_loss = 0  # edge_detection_loss(predicted, target)
    count_loss = 0  # value_count_loss(predicted, target, num_classes)

    size = predicted.shape[0]
    mid_part = 0.4
    start = int(size * mid_part)
    end = int(size * (1 - mid_part))
    mid_loss = 1 - (
        torch.sum(predicted[start:end, start:end] == target[start:end, start:end])
        / torch.numel(target[start:end, start:end])
    )

    # return recon_loss * 10 + edge_loss * 5 + count_loss * 100
    return recon_loss * 10 + mid_loss * 10


def detect_edges(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape

    # Get horizontal edges (differences between rows)
    h_edges = (image[: h - 1, :] != image[1:, :]).float()

    # Get vertical edges (differences between columns)
    v_edges = (image[:, : w - 1] != image[:, 1:]).float()

    return h_edges, v_edges


def edge_detection_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Detect edges in both images
    pred_h_edges, pred_v_edges = detect_edges(predicted)
    target_h_edges, target_v_edges = detect_edges(target)

    # Calculate loss for horizontal edges
    h_edge_loss = torch.mean((pred_h_edges - target_h_edges).pow(2))

    # Calculate loss for vertical edges
    v_edge_loss = torch.mean((pred_v_edges - target_v_edges).pow(2))

    # Combine for total edge loss
    edge_loss = (h_edge_loss + v_edge_loss) / 2.0

    return edge_loss


def value_count_loss(
    predicted: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Efficiently computes loss based on distribution of classes using PyTorch's bincount.

    Args:
        predicted: The predicted image tensor
        target: The target image tensor
        num_classes: Number of possible classes
    """
    # Flatten tensors
    pred_flat = predicted.flatten().to(torch.int64)
    target_flat = target.flatten().to(torch.int64)

    # Get counts for each value using bincount
    pred_counts = torch.bincount(pred_flat, minlength=num_classes).float()
    target_counts = torch.bincount(target_flat, minlength=num_classes).float()

    # Normalize to get frequency distributions
    total_pixels = torch.numel(predicted)
    pred_freq = pred_counts / total_pixels
    target_freq = target_counts / total_pixels

    # Calculate distribution difference loss
    # Can use MSE, L1, KL divergence, or other metrics
    count_loss = torch.mean((pred_freq - target_freq).pow(2))

    return count_loss


import matplotlib.pyplot as plt


def train_fractal(
    model,
    target_image,
    num_iterations=100,
    lr=0.005,
    temperature_start=0.2,
    temperature_end=0.004,
    plot_every_n=10,
    color_map=None,
        max_depth=5,
        warmup=True,
):
    """
    Train the fractal model to match the target image.

    Args:
        model: The Fractal2DDiff model
        target_image: Target image tensor of shape [H, W] with class indices
        num_iterations: Number of training iterations
        lr: Learning rate
        temperature: Temperature for the softmax operations

    Returns:
        Trained model and loss history
    """
    print("starting training")

    # Prepare target
    target = target_image.to(model.device)
    target_flat = target.reshape(-1)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Track loss history
    loss_history = []

    # Training loop
    for i in range(num_iterations):
        progress = min(1.0, i / (0.8 * num_iterations))
        temperature = (
            temperature_start * (temperature_end / temperature_start) ** progress
        )

        optimizer.zero_grad()

        # Forward pass - use different depths for different phases of training
        if warmup:
            depth = min(max_depth, 1 + i // 30)  # Start shallow, gradually increase depth
        else:
            depth = max_depth

        output = model(max_depth=depth, temperature=temperature, hard=False)
        output_flat = output.reshape(-1, model.num_classes)

        # Calculate loss
        loss = loss_fn(output_flat, target_flat)
        loss_history.append(loss.item())

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        model.track_iteration(i, loss)
        if (i + 1) % plot_every_n == 0:
            model.plot_history(
                [depth, depth + 2],
                target_image=target_image,
                temperature=temperature,
                color_map=color_map,
            )

    return model


def optimize_fractal_function_diff(
    model: Fractal2DDiff,
    max_depth: int,
    target_image: torch.Tensor,
    num_iterations: int = 2000,
    adam_lr: float = 0.01,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_every_n: int = 50,
    show_progress: bool = True,
    color_map=None,
) -> Fractal2DDiff:
    """
    Optimize a differentiable fractal model using Adam.

    Args:
        model: The Fractal2DDiff model to optimize
        max_depth: Maximum depth to use for training
        target_image: Target image tensor of shape [height, width] with class indices
        num_iterations: Number of optimization iterations
        adam_lr: Learning rate for Adam optimizer
        temperature_start: Initial temperature for softmax
        temperature_end: Final temperature for softmax
        device: Device to use for computation
        plot_every_n: Plot every n iterations (0 to disable)
        show_progress: Whether to show progress in console

    Returns:
        The optimized model
    """
    target_image = target_image.to(device=device, dtype=torch.long)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

    # Cross entropy loss for class probabilities
    criterion = nn.CrossEntropyLoss()

    # Temperature annealing schedule
    def get_temperature(iteration):
        progress = min(1.0, iteration / (0.8 * num_iterations))
        return temperature_start * (temperature_end / temperature_start) ** progress

    best_loss = float("inf")
    best_state = {}

    for i in range(num_iterations):
        # Clear gradients
        optimizer.zero_grad()

        # Get current temperature
        temp = get_temperature(i)

        # Forward pass - get soft class predictions
        # Shape: [num_points_x, num_points_y, num_classes]
        predicted_probs = model(max_depth, temperature=temp, hard=False)

        # Reshape for cross entropy
        # From [H, W, C] to [H*W, C]
        predicted_probs_flat = predicted_probs.reshape(-1, model.num_classes)
        target_flat = target_image.reshape(-1)

        # Compute loss
        loss = criterion(predicted_probs_flat, target_flat)

        # Add regularization to encourage diverse class assignments
        class_diversity_loss = -torch.std(model.class_probs.mean(dim=0))

        # Total loss
        total_loss = loss + 0.01 * class_diversity_loss

        # Backward pass
        total_loss.backward()

        # Update weights
        optimizer.step()

        # Clamp split points to valid range
        with torch.no_grad():
            model.split_points.data.clamp_(0.1, 0.9)

        # Track history
        model.track_iteration(i, total_loss)

        # Visualize progress
        if plot_every_n > 0 and (i % plot_every_n == 0 or i == num_iterations - 1):
            if show_progress:
                print(
                    f"Iteration {i}/{num_iterations}, Loss: {total_loss.item():.6f}, Temp: {temp:.4f}"
                )

            model.plot_history(
                depths=[max_depth - 3, max_depth, max_depth + 2],
                target_image=target_image,
                temperature=temp,
                color_map=color_map,
            )

        # Save best state
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {
                name: param.detach().clone() for name, param in model.named_parameters()
            }

    # Restore best parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(best_state[name])

    return model


def load_and_quantize_image(
    image_path: str,
    num_classes: int,
    target_size: tuple[int, int] = (100, 100),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load an image (from local file or URL), resize it, and quantize to a specified number of colors.

    Args:
        image_path: Path to the image file or HTTP URL
        num_classes: Number of colors to quantize to
        target_size: Output size (width, height)
        device: Device to place tensor on

    Returns:
        Tuple of (quantized_image_tensor, color_map)
        - quantized_image_tensor: Tensor of shape [height, width] with class indices
        - color_map: Array of shape [num_classes, 3] with RGB values for each class
    """
    # Check if the image_path is a URL or local file
    is_url = bool(urllib.parse.urlparse(image_path).scheme)

    if is_url:
        # Load image from URL
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()  # Raise exception for HTTP errors
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            raise Exception(f"Failed to load image from URL: {e}")
    else:
        # Load image from local file
        try:
            img = Image.open(image_path)
        except Exception as e:
            raise Exception(f"Failed to load image from file: {e}")

    # Resize image
    img = img.resize(target_size)
    img_array = np.array(img)

    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=-1)

    # Handle RGBA images (remove alpha channel)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Extract RGB values
    pixels = img_array.reshape(-1, 3)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Reshape back to original dimensions
    quantized_img = labels.reshape(target_size[1], target_size[0])

    # Create tensor
    quantized_tensor = torch.tensor(quantized_img, device=device)

    print(f"Image quantized to {num_classes} colors.")

    return quantized_tensor, centers


class EdgeAwareLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        color_map: np.ndarray,
        edge_weight: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        A loss function that combines classification loss with edge-awareness.

        Args:
            num_classes: Number of classes in the segmentation
            color_map: Array of shape [num_classes, 3] with RGB values for each class
            edge_weight: Weight for the edge detection component
            device: Computation device
        """
        super().__init__()
        self.num_classes = num_classes
        self.color_map = torch.tensor(color_map / 255.0, device=device, dtype=torch.float32)
        self.edge_weight = edge_weight
        self.device = device
        self.classification_loss = nn.CrossEntropyLoss()

        # Create Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=device).view(1, 1, 3, 3)

        # Cache for target edge map
        self.target_edges = None

    def _compute_edge_map(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Compute edge map from RGB image.

        Args:
            rgb_image: Tensor of shape [..., 3] containing RGB values

        Returns:
            Edge map tensor
        """
        # Convert RGB to grayscale
        if rgb_image.dim() == 3:  # [H, W, 3]
            gray = (0.299 * rgb_image[..., 0] +
                    0.587 * rgb_image[..., 1] +
                    0.114 * rgb_image[..., 2])
            # Add batch and channel dimensions
            gray = gray.unsqueeze(0).unsqueeze(0)
        else:  # [B, H, W, 3]
            gray = (0.299 * rgb_image[..., 0] +
                    0.587 * rgb_image[..., 1] +
                    0.114 * rgb_image[..., 2])
            # Add channel dimension
            gray = gray.unsqueeze(1)

        # Compute edges
        edges_x = nn.functional.conv2d(gray, self.sobel_x, padding=1)
        edges_y = nn.functional.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        return edges

    def _class_indices_to_rgb(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert class indices to RGB values using the color map.

        Args:
            indices: Tensor with class indices

        Returns:
            Tensor with RGB values
        """
        # One-hot encode the indices to create masks for each class
        one_hot = nn.functional.one_hot(indices, self.num_classes).float()

        # Reshape one_hot to [*, num_classes, 1] and multiply with color_map [num_classes, 3]
        # to get [*, num_classes, 3], then sum over classes to get [*, 3]
        if indices.dim() == 2:  # [H, W]
            rgb = torch.einsum('hwc,cd->hwd', one_hot, self.color_map)
        else:  # [B, H, W]
            rgb = torch.einsum('bhwc,cd->bhwd', one_hot, self.color_map)

        return rgb

    def _probs_to_rgb(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Convert class probabilities to RGB values using the color map.

        Args:
            probs: Tensor with class probabilities [..., num_classes]

        Returns:
            Tensor with RGB values
        """
        # Multiply probabilities with color map and sum over classes
        if probs.dim() == 3:  # [H, W, C]
            rgb = torch.einsum('hwc,cd->hwd', probs, self.color_map)
        else:  # [B, H, W, C]
            rgb = torch.einsum('bhwc,cd->bhwd', probs, self.color_map)

        return rgb

    def compute_target_edges(self, target: torch.Tensor):
        """
        Compute and cache the edge map for the target image.

        Args:
            target: Target tensor with class indices [H, W] or [B, H, W]
        """
        target_rgb = self._class_indices_to_rgb(target)
        self.target_edges = self._compute_edge_map(target_rgb)

    def forward(self,
                pred_probs: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_probs: Predicted class probabilities [..., num_classes]
            target: Target class indices

        Returns:
            Tuple of (total_loss, classification_loss, edge_loss)
        """
        # Ensure target edges are computed
        if self.target_edges is None:
            self.compute_target_edges(target)

        # Compute classification loss
        target_flat = target.reshape(-1)
        pred_flat = pred_probs.reshape(-1, self.num_classes)
        cls_loss = self.classification_loss(pred_flat, target_flat)

        # Compute edge loss
        pred_rgb = self._probs_to_rgb(pred_probs)
        pred_edges = self._compute_edge_map(pred_rgb)
        edge_loss = nn.functional.mse_loss(pred_edges, self.target_edges)

        # Combined loss
        total_loss = cls_loss + self.edge_weight * edge_loss

        return total_loss, cls_loss, edge_loss
