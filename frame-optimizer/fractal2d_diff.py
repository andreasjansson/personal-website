import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Fractal2DDiff(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_values: int,
        num_points_x: int = 100,
        num_points_y: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        #assert num_values >= num_classes * 2, (
        #    "num_values must be at least twice num_classes"
        #)

        self.num_classes = num_classes
        self.num_values = num_values
        self.device = device
        self.dtype = dtype
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y

        # Pre-generate fixed grid coordinates
        x = torch.linspace(0, 1, num_points_x, device=device, dtype=dtype)
        y = torch.linspace(0, 1, num_points_y, device=device, dtype=dtype)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing="ij")
        self.grid_points = torch.stack(
            [self.grid_x.flatten(), self.grid_y.flatten()], dim=1
        )

        # Initialize split points (uniformly distributed between 0.2 and 0.8)
        self.split_points = nn.Parameter(
            torch.rand(self.num_values, device=device, dtype=dtype) * 0.6 + 0.2
        )

        # Initialize split directions as continuous values
        # Values closer to 0 represent horizontal splits, closer to 1 represent vertical
        self.split_directions_logits = nn.Parameter(
            torch.randn(self.num_values, device=device, dtype=dtype)
        )

        # Initialize class probabilities (softmax over logits)
        class_logits = torch.randn(
            self.num_values, num_classes, device=device, dtype=dtype
        )
        self.class_logits = nn.Parameter(class_logits)

        # Initialize child selection weights
        # For each node, for each direction (left/right), logits for selecting each possible child
        self.child_selection_logits = nn.Parameter(
            torch.randn(self.num_values, 2, self.num_values, device=device, dtype=dtype)
        )

        # Initialize history dictionary
        self.reset_history()

    def reset_history(self):
        """Reset the history tracking."""
        self.history = {
            "loss": [],
            "iterations": [],
        }

    @property
    def split_directions(self) -> torch.Tensor:
        """Convert direction logits to probabilities (soft version of 0/1)"""
        return torch.sigmoid(self.split_directions_logits)

    @property
    def class_probs(self) -> torch.Tensor:
        """Convert class logits to probabilities"""
        return torch.softmax(self.class_logits, dim=1)

    @property
    def child_selection_probs(self) -> torch.Tensor:
        """Convert child selection logits to probabilities"""
        return torch.softmax(self.child_selection_logits, dim=2)

    def track_iteration(self, iteration: int, recon_loss: torch.Tensor):
        self.history["loss"].append(recon_loss.item())
        self.history["iterations"].append(iteration)

    def forward(
        self, max_depth: int, temperature: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        """
        Simplified fully differentiable implementation of fractal generation.
        """
        batch_size = self.num_points_x * self.num_points_y

        # Use pre-computed grid points
        x_positions = self.grid_points[:, 0]  # [batch_size]
        y_positions = self.grid_points[:, 1]  # [batch_size]

        # Start with 100% probability at root node
        node_probs = torch.zeros(
            batch_size, self.num_values, device=self.device, dtype=self.dtype
        )
        node_probs[:, 0] = 1.0  # Start at root (index 0)

        # Process all depths in sequence
        for depth in range(max_depth):
            next_probs = torch.zeros_like(node_probs)

            for node_idx in range(self.num_values):
                # Skip nodes with negligible probability
                if node_probs[:, node_idx].max() < 1e-10:
                    continue

                # Get parameters for this node
                split_point = self.split_points[node_idx]
                split_dir = self.split_directions[node_idx]  # 0=horizontal, 1=vertical

                # calculate left/right probabilities based on position
                # Horizontal split: compare x position with split point
                h_left_prob = torch.sigmoid((split_point - x_positions) / temperature)

                # Vertical split: compare y position with split point
                v_left_prob = torch.sigmoid((split_point - y_positions) / temperature)

                # Blend based on direction probability
                left_prob = (1 - split_dir) * h_left_prob + split_dir * v_left_prob
                right_prob = 1 - left_prob

                # Scale by current node probability
                left_prob = left_prob * node_probs[:, node_idx]
                right_prob = right_prob * node_probs[:, node_idx]

                # Get child probabilities
                left_child_probs = torch.softmax(
                    self.child_selection_logits[node_idx, 0] / temperature, dim=0
                )
                right_child_probs = torch.softmax(
                    self.child_selection_logits[node_idx, 1] / temperature, dim=0
                )

                # Update next probabilities
                for child_idx in range(self.num_values):
                    next_probs[:, child_idx] += (
                        left_prob * left_child_probs[child_idx]
                        + right_prob * right_child_probs[child_idx]
                    )

            # Normalize and update for next depth
            sum_probs = next_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
            node_probs = next_probs / sum_probs

        # Final class probabilities
        # For each point, weight class probs by node probs
        class_probs = torch.matmul(
            node_probs, self.class_probs
        )  # [batch_size, num_classes]

        if hard:
            # Return hard class assignments
            classes = torch.argmax(class_probs, dim=1)
            return classes.reshape(self.num_points_x, self.num_points_y)
        else:
            # Return soft class probabilities
            return class_probs.reshape(
                self.num_points_x, self.num_points_y, self.num_classes
            )

    def plot_history(
        self,
        depths: list[int],
        target_image: torch.Tensor,
        color_map: np.ndarray | None = None,
        temperature: float = 0.1,
    ):
        """Visualize the current fractal structure alongside the target image with loss history"""
        iteration = (
            self.history["iterations"][-1] if len(self.history["iterations"]) else 0
        )

        # Generate outputs at different depths
        with torch.no_grad():
            # Hard outputs for visualization
            outputs = {
                depth: self(depth, temperature=temperature, hard=True)
                for depth in depths
            }

        num_plots = len(depths) + 1  # +1 for target image

        # Create figure with two rows - top for loss, bottom for images
        fig = plt.figure(figsize=(7.5, 4))
        grid = plt.GridSpec(2, num_plots, height_ratios=[1, 4])

        # Plot loss history in a subplot that spans all columns
        ax_loss = fig.add_subplot(grid[0, :])
        if self.history["loss"]:
            ax_loss.plot(
                self.history["iterations"],
                self.history["loss"],
                color="blue",
                label="Loss",
            )
            ax_loss.set_xlabel("Iteration")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title(f"Loss (iteration {iteration})")
            ax_loss.grid(True)
        else:
            ax_loss.text(
                0.5, 0.5, "No optimization history available", ha="center", va="center"
            )
            ax_loss.set_xticks([])
            ax_loss.set_yticks([])

        # Create a colormap - use provided color map if available
        if color_map is not None:
            # Create a custom colormap from the color map
            colors = ListedColormap(color_map / 255.0)
        else:
            # Use default colormap
            colors = plt.cm.get_cmap("tab10", self.num_classes)

        # First plot target image
        target_img = target_image.cpu().numpy()
        if target_img.shape != (self.num_points_y, self.num_points_x):
            # Transpose if needed for correct orientation
            if target_img.shape == (self.num_points_x, self.num_points_y):
                target_img = target_img.T

        ax_target = fig.add_subplot(grid[1, 0])
        im = ax_target.imshow(
            target_img,
            cmap=colors,
            origin="upper",
            extent=[0, 1, 0, 1],
            vmin=0,
            vmax=self.num_classes - 1,
        )
        ax_target.set_title("Target Image")
        ax_target.set_xlabel("X")
        ax_target.set_ylabel("Y")

        # Then plot each fractal output
        for i, depth in enumerate(depths):
            ax = fig.add_subplot(grid[1, i + 1])  # +1 because target image is first
            output = outputs[depth].cpu().numpy()

            # Ensure correct orientation
            if output.shape != target_img.shape and output.T.shape == target_img.shape:
                output = output.T

            # Plot the fractal image
            im = ax.imshow(
                output,
                cmap=colors,
                origin="upper",
                extent=[0, 1, 0, 1],
                vmin=0,
                vmax=self.num_classes - 1,
            )

            # Calculate accuracy
            if target_img.shape == output.shape:
                accuracy = np.mean(target_img == output)
                ax.set_title(f"Depth {depth} (Acc: {accuracy:.2%})")
            else:
                ax.set_title(f"Depth {depth}")

            ax.set_xlabel("X")

            # Only show y-label on leftmost plot
            if i == 0:
                ax.set_ylabel("Y")

        plt.tight_layout()
        plt.show()


    def to_non_diff(self) -> 'Fractal2DNonDiff':
        """Convert this differentiable fractal model to a non-differentiable version."""
        # Import needed here to avoid circular imports
        from fractal2d_nondiff import Fractal2DNonDiff

        # Create a new non-differentiable model with the same dimensions
        non_diff_model = Fractal2DNonDiff(
            num_classes=self.num_classes,
            num_values=self.num_values,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y,
            device=self.device,
            dtype=self.dtype
        )

        # Transfer split points
        non_diff_model.split_points.data.copy_(self.split_points.data)

        # Convert continuous directions to discrete (0=horizontal, 1=vertical)
        directions = (torch.sigmoid(self.split_directions_logits) > 0.5).long()
        non_diff_model.split_directions.data.copy_(directions)

        # Convert class probabilities to discrete class assignments
        classes = torch.argmax(self.class_probs, dim=1)

        # Update classes
        non_diff_model.classes.data.copy_(classes)

        # Determine child indices and same_class_direction based on the class assignment and selection probabilities
        for value_idx in range(self.num_values):
            left_child = torch.argmax(self.child_selection_probs[value_idx, 0]).item()
            right_child = torch.argmax(self.child_selection_probs[value_idx, 1]).item()

            # Update child indices
            non_diff_model.child_indices[0, value_idx] = left_child
            non_diff_model.child_indices[1, value_idx] = right_child

        return non_diff_model
