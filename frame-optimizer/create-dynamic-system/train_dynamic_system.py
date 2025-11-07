import time
import subprocess
import tempfile
import re
import os
import random
from typing import Iterator
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from cog import Input, Path, BaseModel


class ModelOutput(BaseModel):
    visualization: Path
    cycle_images: list[Path] | None = None
    cycle_losses_target1: list[float] | None = None
    cycle_losses_target2: list[float] | None = None
    historical_losses: list[float] | None = None
    model_state: Path | None = None
    optimizer_state: Path | None = None
    animation: Path | None = None


def train_dynamic_system(
    code: str,
    target1: Path,
    target2: Path,
    width: int = Input(default=128),
    height: int = Input(default=128),
    training_steps: int = Input(default=1000),
    max_variables: int = Input(default=100),
    cycle_length: int = Input(default=20),
    loss_cycles: int = Input(default=4),
    total_cycles: int = Input(default=6),
    learning_rate: float = Input(default=0.005),
    yield_every: int = Input(default=250),
    return_animation: bool = Input(default=True),
    timeout: int = Input(default=600),
    seed: int = Input(default=None),
    model_state: Path = Input(default=None),
    optimizer_state: Path = Input(default=None),
) -> Iterator[ModelOutput]:
    assert loss_cycles <= total_cycles
    assert loss_cycles % 2 == 0

    seed = seed_or_random_seed(seed)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    target1_tensor = load_image_as_tensor(target1, width, height)
    target2_tensor = load_image_as_tensor(target2, width, height)

    system = load_class(code)

    trainer = DynamicSystemTrainer(
        model=system,
        cycle_length=cycle_length,
        width=width,
        height=height,
        max_variables=max_variables,
        learning_rate=learning_rate,
        loss_cycles=loss_cycles,
        total_cycles=total_cycles,
    )

    if model_state and optimizer_state:
        trainer.load_states(model_state, optimizer_state)

    for output in trainer.train(
        target1_tensor,
        target2_tensor,
        num_iterations=training_steps,
        yield_every=yield_every,
        return_animation=return_animation,
        timeout=timeout,
    ):
        yield output


def seed_or_random_seed(seed: int | None) -> int:
    # Max seed is 2147483647
    if not seed or seed <= 0:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

    print(f"Using seed: {seed}\n")
    return seed


def extract_class_name(code: str) -> str | None:
    """Extract the name of the DynamicSystem class from code"""
    match = re.search(r"class\s+(\w+)\s*\(\s*DynamicSystem\s*\)\s*:", code)
    if match:
        return match.group(1)
    return None


def load_image_as_tensor(image_path: Path, width: int, height: int) -> torch.Tensor:
    """Load image and convert to tensor with shape [H,W,3]"""
    img = Image.open(image_path).convert("RGB").resize((width, height))
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32).to("cuda")


class DynamicSystem(nn.Module):
    """Base class for dynamic systems"""

    def reset_state(self) -> None:
        """Reset the system to initial conditions"""
        raise NotImplementedError

    def step(self) -> None:
        """Evolve the system one step forward"""
        raise NotImplementedError

    def get_image(self, width: int, height: int) -> torch.Tensor:
        """Convert current state to RGB image tensor [H,W,3]"""
        raise NotImplementedError


class DynamicSystemTrainer:
    """General framework for training iterated dynamic systems"""

    def __init__(
        self,
        model: DynamicSystem,
        learning_rate=0.005,
        cycle_length: int = 10,
        width: int = 128,
        height: int = 128,
        max_variables: int = 50,
        loss_cycles: int = 4,
        total_cycles: int = 6,
    ):
        self.model = model

        num_variables = self.count_variables()
        assert num_variables <= max_variables, (
            f"The model has {num_variables} variables (including number of elements in tensors), maximum is {max_variables}"
        )

        self.width = width
        self.height = height
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.cycle_length = cycle_length
        self.loss_cycles = loss_cycles
        self.total_cycles = total_cycles
        self.loss_fn = nn.MSELoss()
        self.neg_loss_fn = nn.HuberLoss(delta=0.8)

    def count_variables(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_image(self, tensor: torch.Tensor, path: str) -> Path:
        tensor_np = tensor.detach().cpu().numpy()
        tensor_np = np.clip(tensor_np, 0, 1)
        img_data = (tensor_np * 255).astype(np.uint8)
        img = Image.fromarray(img_data)
        img.save(path)

        return Path(path)

    def save_plot(self, figure: plt.Figure, path: str) -> Path:
        """Save a matplotlib figure to a file and return the path"""
        figure.savefig(path)
        plt.close(figure)
        return Path(path)

    def train(
        self,
        target1: torch.Tensor,
        target2: torch.Tensor,
        num_iterations: int,
        yield_every: int,
        return_animation: bool,
        timeout: int,
    ) -> Iterator[ModelOutput]:
        start_time = time.time()

        losses = []

        os.makedirs("output", exist_ok=True)

        for iteration in range(num_iterations):
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"Training terminated: timeout after {elapsed_time:.2f} seconds (limit: {timeout} seconds)")
                raise TimeoutError(f"Training exceeded time limit of {timeout} seconds")

            self.model.reset_state()
            self.optimizer.zero_grad()

            images = []
            for step in range(self.cycle_length * self.loss_cycles):
                self.model.step()
                if step % self.cycle_length == self.cycle_length - 1:
                    images.append(self.get_image())

            targets = [target1, target2]
            pos_losses = []
            neg_losses = []
            for i, image in enumerate(images):
                target_idx = i % 2  # Alternates between 0 and 1
                pos_target = targets[target_idx]
                neg_target = targets[1 - target_idx]  # The other target

                pos_losses.append(self.loss_fn(image, pos_target))
                neg_losses.append(self.neg_loss_fn(image, neg_target))

            pos_loss = torch.stack(pos_losses).mean()
            neg_loss = torch.stack(neg_losses).mean()
            loss = pos_loss - neg_loss
            losses.append(loss.item())

            # Backpropagate and update
            loss.backward()

            if torch.isnan(loss):
                raise Exception("Loss went to nan")

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Print progress
            if (iteration + 1) % 20 == 0:
                format_loss = lambda loss: f"{loss.item():.4f}"
                format_losses = lambda losses: ", ".join(
                    [format_loss(l) for l in losses]
                )
                print(f"Iteration {iteration + 1}, Loss: {format_loss(loss)}")
                print(f"  Positive losses: {format_losses(pos_losses)}")
                print(f"  Negative losses: {format_losses(neg_losses)}")

            # Visualize progress and yield Output
            if iteration % yield_every == 0 or iteration == num_iterations - 1:
                is_final = iteration == num_iterations - 1
                yield self.create_output(
                    iteration,
                    losses,
                    target1,
                    target2,
                    viz_only=not is_final,
                    return_animation=return_animation,
                )

            if len(losses) == 100:
                losses_tensor = torch.tensor(losses)
                if torch.max(torch.abs(losses_tensor - losses_tensor.mean())) < 0.0001:
                    raise Exception(
                        "System is not learning at all (loss is unchanged in the first 100 iterations)"
                    )

    def create_output(
        self, iteration, losses, target1, target2, viz_only, return_animation
    ):
        step_frames = self.images_over_time(self.total_cycles)
        viz_path = self.visualize(iteration, step_frames, losses, target1, target2)

        if viz_only:
            return ModelOutput(visualization=viz_path)

        cycle_images = []

        for c in range(self.total_cycles):
            cycle_idx = self.cycle_length * (c + 1) - 1
            img_path = f"output/cycle_{iteration}_{c + 1}.png"
            cycle_images.append(
                self.save_image(torch.tensor(step_frames[cycle_idx]), img_path)
            )

        all_frame_paths = []
        if return_animation:
            # Generate high-resolution frames for animation
            hi_res_frames = self.images_over_time(self.total_cycles, width=512, height=512)
            for i, frame in enumerate(hi_res_frames):
                frame_path = f"output/frame_{iteration}_{i:03d}.png"
                all_frame_paths.append(self.save_image(frame, frame_path))

        animation_path = create_animation(all_frame_paths) if return_animation else None

        cycle_losses_target1 = []
        cycle_losses_target2 = []

        for i, image in enumerate(step_frames):
            if i % self.cycle_length == self.cycle_length - 1:
                cycle_losses_target1.append(self.loss_fn(image, target1).item())
                cycle_losses_target2.append(self.loss_fn(image, target2).item())

        model_state_path, optimizer_state_path = self.save_states()

        return ModelOutput(
            cycle_images=cycle_images,
            cycle_losses_target1=cycle_losses_target1,
            cycle_losses_target2=cycle_losses_target2,
            historical_losses=losses.copy(),
            visualization=viz_path,
            model_state=model_state_path,
            optimizer_state=optimizer_state_path,
            animation=animation_path,
        )

    def visualize(self, iteration, cycle_frames, losses, target1, target2) -> Path:
        # Create a figure with better proportions
        viz_fig = plt.figure(figsize=(7, 7))

        # Create a gridspec layout with better space allocation
        gs = plt.GridSpec(4, 3, height_ratios=[1, 1, 1, 1.5], hspace=0.4, wspace=0.3)

        # Plot targets on the first row
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(target1.cpu().numpy())
        ax1.set_title("Target 1")
        ax1.axis("off")

        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(target2.cpu().numpy())
        ax2.set_title("Target 2")
        ax2.axis("off")

        # Plot cycle images in 2 rows of 3
        for c in range(self.total_cycles):
            cycle_idx = self.cycle_length * (c + 1) - 1

            row = 1 if c < 3 else 2
            col = c % 3

            ax = plt.subplot(gs[row, col])
            ax.imshow(cycle_frames[cycle_idx].detach().cpu().numpy())
            ax.set_title(f"Cycle {c + 1}")
            ax.axis("off")

        # Bottom row: split into two charts
        # Left chart: loss history over iterations
        if losses:
            ax_loss = plt.subplot(gs[3, 0:1])
            ax_loss.plot(losses)
            ax_loss.set_xlabel("Iteration")
            ax_loss.set_ylabel("Training Loss")
            ax_loss.grid(alpha=0.3)
            ax_loss.set_title("Loss Over Training")

        # Right chart: distance to targets over steps
        ax_targets = plt.subplot(gs[3, 1:3])

        # Calculate distances for each frame
        target1_losses = []
        target2_losses = []
        for frame in cycle_frames:
            target1_losses.append(self.loss_fn(frame, target1).item())
            target2_losses.append(self.loss_fn(frame, target2).item())

        # Plot distances to both targets
        steps = list(range(len(cycle_frames)))
        ax_targets.plot(steps, target1_losses, "b-", label="Target 1")
        ax_targets.plot(steps, target2_losses, "r-", label="Target 2")

        # Add vertical lines at cycle boundaries and labels
        max_loss = max(max(target1_losses), max(target2_losses))
        for i in range(1, self.total_cycles + 1):
            cycle_end = i * self.cycle_length - 1
            # Vertical line at cycle end
            ax_targets.axvline(x=cycle_end, color="gray", linestyle="--", alpha=0.7)
            # Add cycle number label, positioning it lower to avoid overlap
            ax_targets.text(
                x=cycle_end - self.cycle_length / 2,
                y=max_loss * 0.8,
                s=f"Cycle {i}",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            )

        ax_targets.set_xlabel("Step")
        ax_targets.set_ylabel("Loss")
        ax_targets.legend()
        ax_targets.grid(alpha=0.3)
        ax_targets.set_title("Distance to Targets Over Steps")

        plt.tight_layout()
        viz_path = self.save_plot(viz_fig, f"output/visualization_{iteration}.png")
        return viz_path

    def images_over_time(self, num_cycles, width=None, height=None) -> list[torch.Tensor]:
        with torch.no_grad():
            # Reset model
            self.model.reset_state()

            # Use provided dimensions or default ones
            width = self.width if width is None else width
            height = self.height if height is None else height

            # Run for the specified number of cycles
            frames = []
            for _ in range(self.cycle_length * num_cycles):
                self.model.step()
                frames.append(self.get_image(width=width, height=height))
        return frames

    def get_image(self, width=None, height=None):
        width = self.width if width is None else width
        height = self.height if height is None else height
        image = self.model.get_image(width, height)
        image = torch.clip(image, 0, 1)
        return image

    def save_states(self) -> tuple[Path, Path]:
        """Save model and optimizer states as PyTorch state dicts"""
        os.makedirs("output", exist_ok=True)

        # Save model state
        model_path = "output/model_state.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer state
        optimizer_path = "output/optimizer_state.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)

        return Path(model_path), Path(optimizer_path)

    def load_states(self, model_state_path: Path, optimizer_state_path: Path) -> None:
        """Load model and optimizer states from PyTorch state dicts"""
        # Load model state
        model_state = torch.load(model_state_path)
        self.model.load_state_dict(model_state)

        # Load optimizer state
        optimizer_state = torch.load(optimizer_state_path)
        self.optimizer.load_state_dict(optimizer_state)

        print("Successfully loaded model and optimizer states")


def load_class(code: str) -> DynamicSystem:
    namespace = globals().copy()
    exec(code, namespace)

    class_name = extract_class_name(code)
    if not class_name:
        raise ValueError("Could not find DynamicSystem class in generated code")

    system_class = namespace[class_name]
    system = system_class().to("cuda")
    return system


def create_animation(frames: list[Path]) -> Path:
    """Create an MP4 animation from a list of image frames using ffmpeg"""
    assert frames

    output_path = "output/animation.mp4"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        frame_list_path = f.name
        for frame_path in frames:
            f.write(f"file '{os.path.abspath(frame_path)}'\n")
            f.write(f"duration 0.05\n")  # 20 FPS (1/20 = 0.05s per frame)

        f.write(f"file '{os.path.abspath(frames[-1])}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "concat",  # Use concat demuxer for input
            "-safe",
            "0",  # Don't restrict file paths
            "-i",
            frame_list_path,  # Input file list
            "-vsync",
            "vfr",  # Variable frame rate
            "-pix_fmt",
            "yuv420p",  # Pixel format for compatibility
            "-c:v",
            "libx264",  # Video codec
            "-crf",
            "23",  # Quality (lower is better)
            output_path,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        return Path(output_path)
    finally:
        os.unlink(frame_list_path)
