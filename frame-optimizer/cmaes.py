import numpy as np
import cma
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import math
import time

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#NUM_COLOR_IDX = 3
#SPLIT_IDX = 3
#LEFT_IDX = 4
#RIGHT_IDX = 5
#NUM_PARAMS = 6
NUM_COLOR_IDX = 1
SPLIT_IDX = 1
LEFT_IDX = 2
RIGHT_IDX = 3
NUM_PARAMS = 4


# Render frame on GPU
def render_frame(frame_idx, frames, depth, max_depth, width, height):
    # Get frame parameters
    #color = frames[frame_idx, :NUM_COLOR_IDX]
    color = (frames[frame_idx, 0] > 0.5) * 1.0
    color = torch.tensor([color, color, color])
    split = frames[frame_idx, SPLIT_IDX]
    left_idx = int(frames[frame_idx, LEFT_IDX].item())
    right_idx = int(frames[frame_idx, RIGHT_IDX].item())

    # Create base image with background color
    img = color.repeat(height, width, 1).to("cuda")

    if depth == max_depth - 1:
        return img

    # Apply split based on frame type
    if frame_idx % 2 == 0:  # Vertical split
        split_x = max(1, min(width - 1, int(width * split)))
        if split_x > 0:
            left = render_frame(left_idx, frames, depth + 1, max_depth, split_x, height)
            img[:, :split_x, :] = left

        if width - split_x > 0:
            right = render_frame(
                right_idx, frames, depth + 1, max_depth, width - split_x, height
            )
            img[:, split_x:, :] = right

    else:  # Horizontal split
        split_y = max(1, min(height - 1, int(height * split)))

        if split_y > 0:
            top = render_frame(left_idx, frames, depth + 1, max_depth, width, split_y)
            img[:split_y, :, :] = top

        if height - split_y > 0:
            bottom = render_frame(
                right_idx, frames, depth + 1, max_depth, width, height - split_y
            )
            img[split_y:, :, :] = bottom

    return img


# Batch render
def render_population(frames_batch, max_depth, width, height):
    batch_size, n, _ = frames_batch.shape
    depth_images = []

    for d in range(1, max_depth + 1):  # Start from depth 1
        imgs = []
        for i in range(batch_size):
            img = render_frame(0, frames_batch[i], 0, d, width, height)
            imgs.append(img)
        depth_images.append(torch.stack(imgs))

    return depth_images


def get_initial_params(n):
    initial_params = np.random.rand(n * NUM_PARAMS)
    return initial_params


def params_batch_to_frames_batch(params_batch: np.ndarray) -> torch.Tensor:
    frames_batch = [params_to_frames(params) for params in params_batch]
    return torch.stack(frames_batch).to(device)


def params_to_frames(params: np.ndarray) -> torch.Tensor:
    n = len(params) // NUM_PARAMS
    frames = torch.tensor(params).reshape(NUM_PARAMS, n).T
    frames[:, SPLIT_IDX] = (frames[:, SPLIT_IDX] * 0.9 + 0.05).clamp(0.05, 0.95)
    frames[:, LEFT_IDX:] = (frames[:, LEFT_IDX:] * n).floor().clamp(0, n - 1)
    return frames


def objective(
    params: np.ndarray,
    target_tensor: torch.Tensor,
    max_depth: int,
    width: int,
    height: int,
):
    pop_size = len(params)  # [popsize, n*6]
    frames_batch = params_batch_to_frames_batch(params)

    losses = []

    for batch_idx in range(pop_size):
        img = render_frame(0, frames_batch[batch_idx], 0, max_depth, width, height)
        mse = torch.mean((img - target_tensor) ** 2)
        loss = mse.item()
        losses.append(loss)

    return np.array(losses)


def objective_old(
    params: np.ndarray,
    target_tensor: torch.Tensor,
    max_depth: int,
    width: int,
    height: int,
):
    pop_size = len(params)  # [popsize, n*6]
    frames_batch = params_batch_to_frames_batch(params)

    depth_images = render_population(frames_batch, max_depth, width, height)

    losses = []
    total_weight = sum(d + 1 for d in range(max_depth))

    for batch_idx in range(pop_size):
        batch_loss = 0
        for d in range(max_depth):
            weight = d + 1
            img = depth_images[d][batch_idx]
            mse = torch.mean((img - target_tensor) ** 2)
            batch_loss += weight * mse.item()

        losses.append(batch_loss / total_weight)

    return np.array(losses)


# Optimization function
def optimize(
    target: Image.Image,
    n=5,
    max_depth=5,
    width=224,
    height=224,
    iterations=200,
    popsize=16,
):
    """
    Optimize fractal frames to match a target image with live visualization.

    Args:
        target: PIL Image or str (path to image)
        n: Number of frames
        max_depth: Maximum recursion depth
        width, height: Output image size
        iterations: CMA-ES iterations
        popsize: Population size for CMA-ES

    Returns:
        np.ndarray: Optimized frame params [n, 6] (r, g, b, split, left_idx, right_idx)
    """
    # Convert target to tensor
    if isinstance(target, str):
        target = Image.open(target).resize((width, height))
    target_tensor = (
        torch.tensor(np.array(target), dtype=torch.float32).to(device) / 255.0
    )

    # Initial params
    initial_params = get_initial_params(n=n)

    # CMA-ES setup
    es = cma.CMAEvolutionStrategy(
        initial_params,
        # 0.5,
        0.9,
        {
            "bounds": [0, 1],
            "popsize": popsize,
            "CMA_active": True,
        },
    )

    # Best fitness tracking
    best_fitness = float("inf")
    best_frames = None

    # Optimization loop
    for iteration in range(iterations):
        solutions = es.ask()  # [popsize, n*6]
        fitness = objective(
            params=solutions,
            target_tensor=target_tensor,
            max_depth=max_depth,
            width=width,
            height=height,
        )
        es.tell(solutions, fitness)

        # Track best solution
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_frames = params_to_frames(solutions[current_best_idx])

        # Visualization update every 10 iterations
        if iteration % 10 == 0 or iteration == iterations - 1:
            visualize_grid(
                frames=best_frames,
                target=target,
                max_depth=max_depth,
                n=n,
                iteration=iteration,
                best_fitness=best_fitness,
            )

    return es.result.xbest


def optimize_sa(
    target: Image.Image,
    n=5,
    max_depth=5,
    width=224,
    height=224,
    iterations=1000,
    initial_temp=1.0,
    cooling_rate=0.99,
):
    """
    Optimize fractal frames to match a target image using simulated annealing.

    Args:
        target: PIL Image or str (path to image)
        n: Number of frames
        max_depth: Maximum recursion depth
        width, height: Output image size
        iterations: Number of iterations
        initial_temp: Initial temperature for SA
        cooling_rate: Cooling rate for temperature

    Returns:
        np.ndarray: Optimized frame params
    """
    # Convert target to tensor
    if isinstance(target, str):
        target = Image.open(target).resize((width, height))
    target_tensor = torch.tensor(np.array(target), dtype=torch.float32).to(device) / 255.0

    # Initial solution
    current_params = get_initial_params(n=n)
    current_frames = params_to_frames(current_params)

    # Evaluate initial solution
    current_img = render_frame(0, current_frames, 0, max_depth, width, height)
    current_error = torch.mean((current_img - target_tensor) ** 2).item()

    # Track best solution
    best_params = current_params.copy()
    best_error = current_error
    best_frames = current_frames.clone()

    # Simulated annealing params
    temp = initial_temp

    # Progress tracking
    last_improvement = 0

    # Main optimization loop
    for iteration in range(iterations):
        # Create neighboring solution with random perturbation
        neighbor_params = current_params.copy()

        # Randomly select which parameter to modify
        param_idx = np.random.randint(0, len(neighbor_params))

        # Perturb the parameter
        if param_idx % NUM_PARAMS < NUM_COLOR_IDX:  # RGB values
            # Color perturbation (smaller for colors)
            perturbation = np.random.normal(0, 0.3)
            neighbor_params[param_idx] = np.clip(neighbor_params[param_idx] + perturbation, 0, 1)
        elif param_idx % NUM_PARAMS == SPLIT_IDX:  # Split point
            # Split point perturbation
            perturbation = np.random.normal(0, 0.3)
            neighbor_params[param_idx] = np.clip(neighbor_params[param_idx] + perturbation, 0, 1)
        else:  # Frame indices
            # For indices, sometimes just pick a completely new random frame
            if np.random.random() < 0.5:
                neighbor_params[param_idx] = np.random.random()
            else:
                # Small adjustment
                perturbation = np.random.normal(0, 0.3)
                neighbor_params[param_idx] = np.clip(neighbor_params[param_idx] + perturbation, 0, 1)

        # Evaluate neighbor
        neighbor_frames = params_to_frames(neighbor_params)
        neighbor_img = render_frame(0, neighbor_frames, 0, max_depth, width, height)
        neighbor_error = torch.mean((neighbor_img - target_tensor) ** 2).item()

        # Decide whether to accept the new solution
        if neighbor_error < current_error:
            # Always accept better solutions
            current_params = neighbor_params
            current_error = neighbor_error
            current_frames = neighbor_frames

            # Update best solution if this is better
            if neighbor_error < best_error:
                best_error = neighbor_error
                best_params = neighbor_params.copy()
                best_frames = neighbor_frames.clone()
                last_improvement = iteration
        else:
            # Accept worse solutions with a probability that decreases with temperature
            p = np.exp((current_error - neighbor_error) / temp)
            if np.random.random() < p:
                current_params = neighbor_params
                current_error = neighbor_error
                current_frames = neighbor_frames

        # Cool down temperature
        temp *= cooling_rate

        # Visualization update every 20 iterations
        if iteration % 20 == 0 or iteration == iterations - 1:
            visualize_grid(
                frames=best_frames,
                target=target,
                max_depth=max_depth,
                n=n,
                iteration=iteration,
                best_fitness=best_error,
            )
            print(f"Iteration {iteration}, Temperature: {temp:.6f}, Best error: {best_error:.6f}")

        # Optional: adaptive restart if no improvement for a while
        if iteration - last_improvement > 200:
            print("Restarting search...")
            temp = initial_temp * 0.5  # Reduce initial temp for restart
            # Partially randomize current solution but keep some parts of the best
            mix_ratio = 0.7
            random_params = get_initial_params(n=n)
            current_params = mix_ratio * best_params + (1 - mix_ratio) * random_params
            current_frames = params_to_frames(current_params)
            current_img = render_frame(0, current_frames, 0, max_depth, width, height)
            current_error = torch.mean((current_img - target_tensor) ** 2).item()
            last_improvement = iteration

    return best_params


def visualize_grid(
    frames,
    target,
    max_depth,
    n,
    iteration=None,
    width=224,
    height=224,
    best_fitness=None,
):
    # Create a new figure each time
    plt.close("all")  # Close previous figures

    # Calculate grid layout
    num_images = max_depth + 1  # Target + all depths
    grid_size = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / grid_size)
    cols = min(grid_size, num_images)

    fig, axes = plt.subplots(rows, cols, figsize=(6, 3))

    # Handle single row/column case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Display target in first position
    axes.flat[0].imshow(np.array(target))
    axes.flat[0].set_title("Target")
    axes.flat[0].axis("off")

    # Render current best frames
    depth_images = render_population(frames.unsqueeze(0), max_depth, width, height)

    # Display depth images
    for d in range(max_depth):
        if d + 1 < len(axes.flat):  # Ensure we don't go out of bounds
            img_np = (depth_images[d][0].cpu().numpy() * 255).astype(np.uint8)
            axes.flat[d + 1].imshow(img_np)
            axes.flat[d + 1].set_title(f"Depth {d + 1}")
            axes.flat[d + 1].axis("off")

    # Hide unused subplots
    for i in range(num_images, rows * cols):
        if i < len(axes.flat):
            axes.flat[i].axis("off")
            axes.flat[i].set_visible(False)

    # Update title and layout
    if iteration is not None:
        fig.suptitle(f"Iteration {iteration}, Best fitness: {best_fitness:.4f}")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    # Force display update
    plt.draw()
    plt.pause(0.2)  # Longer pause to ensure GUI updates

    if iteration is not None:
        print(f"Iteration {iteration}, Best fitness: {best_fitness:.4f}")


def generate_html(frames, max_depth):
    """Generate HTML code for the fractal frames website."""
    n = len(frames)

    # Generate HTML/CSS for each frame
    frame_html = []
    for i in range(n):
        r, g, b = frames[i, :3].astype(int)
        split = frames[i, 3]
        left_idx = int(frames[i, 4])
        right_idx = int(frames[i, 5])

        if i % 2 == 0:  # Vertical split
            frame_html.append(f"""
<html style="background: rgb({r},{g},{b})">
  <head><title>Fractal Frame {i}</title>
    <meta http-equiv="refresh" content="0.4;_frame_{i}.html" />
  </head>
</html>
            """)

            # Left frame
            left_html = f"""
<html>
  <head><title>Frame {left_idx}</title>
    <meta http-equiv="refresh" content="0.4;_frame_{left_idx}.html" />
  </head>
    <frameset cols="{split * 100}%,{(1 - split) * 100}%">
      <frame src="_frame_{left_idx}.html" scrolling="no">
      <frame src="_frame_{right_idx}.html" scrolling="no">
    </frameset>
</html>
            """

            # Write to file
            with open(f"_frame_{i}.html", "w") as f:
                f.write(left_html)
        else:  # Horizontal split
            frame_html.append(f"""
<html style="background: rgb({r},{g},{b})">
  <head><title>Fractal Frame {i}</title>
    <meta http-equiv="refresh" content="0.4;_frame_{i}.html" />
  </head>
</html>
            """)

            # Right frame
            right_html = f"""
<html>
  <head><title>Frame {right_idx}</title>
    <meta http-equiv="refresh" content="0.4;_frame_{right_idx}.html" />
  </head>
    <frameset rows="{split * 100}%,{(1 - split) * 100}%">
      <frame src="_frame_{left_idx}.html" scrolling="no">
      <frame src="_frame_{right_idx}.html" scrolling="no">
    </frameset>
</html>
            """

            # Write to file
            with open(f"_frame_{i}.html", "w") as f:
                f.write(right_html)

    # Main HTML file
    main_html = f"""
<!DOCTYPE html>
<html>
  <head>
    <title>Fractal Frames Website</title>
    <meta http-equiv="refresh" content="15; url=3.html" />
  </head>
    <frameset cols="30%,70%">
      <frame src="_frame_0.html" scrolling="no">
      <frame src="_frame_1.html" scrolling="no">
    </frameset>
</html>
    """

    return main_html


def make_test_target(width=224, height=224):
    target = Image.new("RGB", (width, height), (255, 0, 0))
    draw = ImageDraw.Draw(target)
    start = int(width * 0.2)
    end = int(width * 0.8)
    draw.rectangle([start, start, end, end], fill=(0, 0, 255))
    return target


def test():
    # Create test image
    target = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(target)
    draw.rectangle([10, 10, 54, 54], fill=(0, 0, 0))
    draw.ellipse([20, 20, 44, 44], fill=(255, 0, 0))

    # Run optimization
    frames = optimize(target=target, n=8, max_depth=5, iterations=100, popsize=16)
    return frames


if __name__ == "__main__":
    test()
