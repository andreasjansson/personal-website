import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from sample_data import create_target_images


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
        optimizer_cls: Callable = optim.Adam,
        optimizer_kwargs: dict = {"lr": 0.01},
        scheduler_cls: Callable | None = None,  # optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs: dict = {"patience": 200, "factor": 0.5, "verbose": True},
        cycle_length: int = 10,
        width: int = 128,
        height: int = 128,
        max_variables: int = 50,
    ):
        self.model = model

        num_variables = self.count_variables()
        assert num_variables < max_variables, (
            f"The model has {num_variables} variables (including number of elements in tensors), maximum is {max_variables}"
        )

        self.width = width
        self.height = height
        self.optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        self.scheduler = None
        if scheduler_cls is not None:
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_kwargs)
        self.cycle_length = cycle_length
        self.loss_fn = nn.MSELoss()

    def count_variables(self) -> int:
        """Count the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_morphing(
        self,
        target1: torch.Tensor,
        target2: torch.Tensor,
        num_iterations: int,
        visualize_every: int,
    ) -> list[float]:
        """Train model to morph between two target images over 4 cycles"""
        losses = []

        for i in range(num_iterations):
            # Reset model state
            self.model.reset_state()
            self.optimizer.zero_grad()

            # Run for 4 complete cycles
            images = []
            for step in range(self.cycle_length * 4):
                self.model.step()
                if step % self.cycle_length == self.cycle_length - 1:
                    images.append(self.model.get_image(self.width, self.height))

            # Compute loss - alternating targets across 4 cycles
            loss1 = self.loss_fn(images[0], target1)
            loss2 = self.loss_fn(images[1], target2)
            loss3 = self.loss_fn(images[2], target1)
            loss4 = self.loss_fn(images[3], target2)

            neg_loss1 = self.loss_fn(images[0], target2)
            neg_loss2 = self.loss_fn(images[1], target1)
            neg_loss3 = self.loss_fn(images[2], target2)
            neg_loss4 = self.loss_fn(images[3], target1)

            # Total loss
            loss = (loss1 + loss2 + loss3 + loss4) / 4
            neg_loss = (neg_loss1 + neg_loss2 + neg_loss3 + neg_loss4) / 4
            total_loss = loss - neg_loss
            losses.append(total_loss.item())

            # Backpropagate and update
            loss.backward()

            if torch.isnan(loss):
                raise Exception("Loss went to nan")

            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(loss)

            # Print progress
            if (i + 1) % 20 == 0:
                print(f"Iteration {i + 1}, Loss: {loss.item():.6f}")
                print(
                    f"  Cycle losses: {loss1.item():.6f}, {loss2.item():.6f}, {loss3.item():.6f}, {loss4.item():.6f}"
                )

            # Visualize progress
            if i % visualize_every == 0 or i == num_iterations - 1:
                self.visualize_morphing(target1, target2)
                if i > 0:
                    self.visualize_loss_history(losses)

            if len(losses) == 100:
                losses_tensor = torch.tensor(losses)
                if torch.max(torch.abs(losses_tensor - losses_tensor.mean())) < 0.0001:
                    raise Exception(
                        "System is not learning at all (loss is unchanged in the first 100 iterations)"
                    )

        return losses

    def images_over_time(self, num_cycles):
        with torch.no_grad():
            # Reset model
            self.model.reset_state()

            # Run for six complete cycles (extended from 4 to 6)
            frames = []
            for _ in range(self.cycle_length * num_cycles):
                self.model.step()
                frames.append(
                    self.model.get_image(self.width, self.height).detach().cpu().numpy()
                )
        return frames

    def frame_distances_to_target(
        self, frames: list[np.ndarray], target: torch.Tensor
    ) -> list[float]:
        target_np = target.cpu().numpy()
        return [np.mean((frame - target_np) ** 2) for frame in frames]

    def visualize_morphing(self, target1: torch.Tensor, target2: torch.Tensor) -> None:
        """Visualize the morphing between target images across multiple cycles"""
        frames = self.images_over_time(6)

        # Create figure showing end of each cycle and target images
        fig, axes = plt.subplots(2, 4, figsize=(7, 5))

        # Top row: target images and first two cycle outputs
        axes[0, 0].imshow(target1.cpu().numpy())
        axes[0, 0].set_title("Target 1")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(target2.cpu().numpy())
        axes[0, 1].set_title("Target 2")
        axes[0, 1].axis("off")

        # Show model output at the end of first 2 cycles
        for i in range(2):
            cycle_idx = self.cycle_length * (i + 1) - 1
            axes[0, i + 2].imshow(frames[cycle_idx])
            axes[0, i + 2].set_title(f"Cycle {i + 1}")
            axes[0, i + 2].axis("off")

        # Bottom row: cycles 3-6
        for i in range(4):
            cycle_idx = self.cycle_length * (i + 3) - 1
            axes[1, i].imshow(frames[cycle_idx])
            axes[1, i].set_title(f"Cycle {i + 3}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

        # Calculate and visualize distances to targets over time
        dist_to_target1 = self.frame_distances_to_target(frames, target1)
        dist_to_target2 = self.frame_distances_to_target(frames, target2)

        # Plot losses
        plt.figure(figsize=(7, 3))
        plt.plot(dist_to_target1, label="Distance to target 1", color="blue")
        plt.plot(dist_to_target2, label="Distance to target 2", color="red")

        # Mark cycle boundaries
        for i in range(7):
            plt.axvline(
                x=i * self.cycle_length, color="gray", linestyle="--", alpha=0.5
            )
            if i < 6:
                # Mark expected closest match points
                cycle_end = (i + 1) * self.cycle_length - 1
                expected_target = "Target 1" if i % 2 == 0 else "Target 2"
                target_marker = "o" if i % 2 == 0 else "s"
                plt.scatter(
                    cycle_end,
                    dist_to_target1[cycle_end]
                    if i % 2 == 0
                    else dist_to_target2[cycle_end],
                    color="green",
                    marker=target_marker,
                    s=50,
                    label=f"Cycle {i + 1}: {expected_target}" if i < 2 else None,
                )

        plt.xlabel("Time Step")
        plt.ylabel("Mean Squared Error")
        plt.ylim(bottom=0)
        plt.title("Distance to target images over 6 cycles")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def visualize_loss_history(self, losses: list[float]) -> None:
        """Visualize the loss history during training"""
        plt.figure(figsize=(7, 4))
        plt.plot(losses, color="blue", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Loss History")
        plt.grid(True, alpha=0.3)

        # Add markers for visualization points
        visualization_points = list(
            range(0, len(losses), min(len(losses), len(losses) // 5))
        )
        if visualization_points:
            plt.scatter(
                visualization_points,
                [losses[i] for i in visualization_points],
                color="red",
                marker="o",
                s=50,
                zorder=3,
                label="Visualization Points",
            )

        # Format y-axis based on loss values
        if min(losses) > 0 and max(losses) / min(losses) > 50:
            plt.yscale("log")
            plt.title("Training Loss History (Log Scale)")

        # Add recent trend line
        if len(losses) > 20:
            recent = losses[-20:]
            recent_x = list(range(len(losses) - 20, len(losses)))
            plt.plot(recent_x, recent, color="green", linewidth=2, label="Recent Trend")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def generate_sequence(self, num_steps: int = 20) -> None:
        """Generate and visualize a sequence of frames"""
        with torch.no_grad():
            self.model.reset_state()
            frames = []

            # Run for specified number of steps
            for _ in range(num_steps):
                self.model.step()
                frames.append(
                    self.model.get_image(self.width, self.height).detach().cpu().numpy()
                )

            # Display frames in a grid
            rows = (num_steps + 3) // 4  # Ceiling division
            cols = min(4, num_steps)

            fig, axes = plt.subplots(rows, cols, figsize=(7, 7 * rows / 4))
            axes = np.array(axes).reshape(rows, cols)  # Ensure 2D array of axes

            for i, frame in enumerate(frames):
                row, col = i // cols, i % cols
                axes[row, col].imshow(frame)
                axes[row, col].set_title(f"Step {i}")
                axes[row, col].axis("off")

            # Hide unused subplots
            for i in range(len(frames), rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis("off")

            plt.tight_layout()
            plt.suptitle("Dynamic System Evolution")
            plt.show()


def main():
    # Create target images
    width = height = 128
    target1, target2 = create_target_images(width, height)
    target1 = target1.to("cuda")
    target2 = target2.to("cuda")

    print("\nTraining...")
    model = MyDynamicSystem().to("cuda")  # TODO
    trainer = DynamicSystemTrainer(
        model=model, cycle_length=12, width=width, height=height
    )

    # Show initial state
    with torch.no_grad():
        model.reset_state()
        for _ in range(10):
            model.step()

        plt.figure(figsize=(5, 5))
        plt.imshow(model.get_image(width, height).detach().cpu().numpy())
        plt.title("Initial pattern")
        plt.axis("off")
        plt.show()

    # Train model
    losses2 = trainer.train_morphing(
        target1, target2, num_iterations=2000, visualize_every=200
    )

    # Plot loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(losses2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Generate sequence
    trainer.generate_sequence(12)
