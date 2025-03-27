import time
import urllib.parse
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch.optim import Adam
import requests

from fractal2d_nondiff import Fractal2DNonDiff
from sa import SimulatedAnnealing
from optimizer import Optimizer, OptimizerPart


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
    plot_every_n: int = 50,
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

    best_loss = float("inf")
    best_state = {}

    def closure():
        optimizer.zero_grad()
        predicted_image = model(max_depth)
        loss = compute_loss(predicted_image, target_image, model.num_values)
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
            plot_fn()
            print(
                f"Iteration {i}/{num_iterations}, Loss: {loss.item():.6f}, time per iteration: {np.mean(times_per_iteration):.4f}s"
            )
            times_per_iteration = []

        if loss.item() < best_loss:
            print("Best loss!")
            plot_fn()
            best_loss = loss.item()
            best_state = {
                "split_points_param": model.split_points_param.detach().clone(),
                "matrices_param": model.matrices_param.detach().clone(),
            }

    # Restore best parameters
    model.split_points_param.data.copy_(best_state["split_points_param"])
    model.matrices_param.data.copy_(best_state["matrices_param"])

    return model


def compute_loss(predicted: torch.Tensor, target: torch.Tensor, num_values):
    # Simple loss based on accuracy
    recon_loss = 1 - (torch.sum(predicted == target) / torch.numel(target))
    edge_loss = 0  # edge_detection_loss(predicted, target)
    count_loss = 0  # value_count_loss(predicted, target, num_values)

    size = predicted.shape[0]
    mid_part = 0.3
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
    predicted: torch.Tensor, target: torch.Tensor, num_values: int
) -> torch.Tensor:
    """
    Efficiently computes loss based on distribution of values using PyTorch's bincount.

    Args:
        predicted: The predicted image tensor
        target: The target image tensor
        num_values: Number of possible values (classes)
    """
    # Flatten tensors
    pred_flat = predicted.flatten().to(torch.int64)
    target_flat = target.flatten().to(torch.int64)

    # Get counts for each value using bincount
    pred_counts = torch.bincount(pred_flat, minlength=num_values).float()
    target_counts = torch.bincount(target_flat, minlength=num_values).float()

    # Normalize to get frequency distributions
    total_pixels = torch.numel(predicted)
    pred_freq = pred_counts / total_pixels
    target_freq = target_counts / total_pixels

    # Calculate distribution difference loss
    # Can use MSE, L1, KL divergence, or other metrics
    count_loss = torch.mean((pred_freq - target_freq).pow(2))

    return count_loss


def load_and_quantize_image(
    image_path: str,
    num_values: int,
    target_size: tuple[int, int] = (100, 100),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load an image (from local file or URL), resize it, and quantize to a specified number of colors.

    Args:
        image_path: Path to the image file or HTTP URL
        num_values: Number of colors to quantize to
        target_size: Output size (width, height)
        device: Device to place tensor on

    Returns:
        Tuple of (quantized_image_tensor, color_map)
        - quantized_image_tensor: Tensor of shape [height, width] with class indices
        - color_map: Array of shape [num_values, 3] with RGB values for each class
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
    kmeans = KMeans(n_clusters=num_values, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Reshape back to original dimensions
    quantized_img = labels.reshape(target_size[1], target_size[0])

    # Create tensor
    quantized_tensor = torch.tensor(quantized_img, device=device)

    print(f"Image quantized to {num_values} colors.")

    return quantized_tensor, centers
