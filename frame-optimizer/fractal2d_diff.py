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
        temperature_forward: float = 0.0001,
        ste: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_values = num_values
        self.device = device
        self.dtype = dtype
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.temperature_forward = temperature_forward
        self.ste = ste

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

    def forward(
        self,
        max_depth: int,
        temperature: float = 0.1,
        hard: bool = False,
    ) -> torch.Tensor:
        batch_size = self.num_points_x * self.num_points_y

        # Use pre-computed grid points
        x_positions = self.grid_points[:, 0]  # [batch_size]
        y_positions = self.grid_points[:, 1]  # [batch_size]

        # Initialize probabilities to start at root node
        node_probs = torch.zeros(
            batch_size, self.num_values, device=self.device, dtype=self.dtype
        )
        node_probs[:, 0] = 1.0  # Start at root (index 0)

        # Initialize bounding boxes for all points
        # We represent the bounding box as probability distributions over the box boundaries
        # Each point starts with the full [0,1] x [0,1] box
        min_x = torch.zeros_like(x_positions)
        max_x = torch.ones_like(x_positions)
        min_y = torch.zeros_like(y_positions)
        max_y = torch.ones_like(y_positions)

        for depth in range(max_depth):
            next_probs = torch.zeros_like(node_probs)
            next_min_x = torch.zeros_like(min_x)
            next_max_x = torch.zeros_like(max_x)
            next_min_y = torch.zeros_like(min_y)
            next_max_y = torch.zeros_like(max_y)

            # Accumulate the weighted bounding boxes
            weight_sum = torch.zeros_like(min_x)

            for node_idx in range(self.num_values):
                # Skip nodes with negligible probability
                if node_probs[:, node_idx].max() < 1e-10:
                    continue

                # Get parameters for this node
                split_point_rel = self.split_points[node_idx].clamp(
                    0.1, 0.9
                )  # Clamp to valid range
                split_dir = self.split_directions[node_idx]  # 0=horizontal, 1=vertical

                # Calculate actual split positions within current bounding box
                split_x = min_x + split_point_rel * (max_x - min_x)
                split_y = min_y + split_point_rel * (max_y - min_y)

                # Compute left/right probabilities based on position
                # Horizontal split: compare x position with split point
                h_left_prob = self._sigmoid_ste((split_x - x_positions), temperature)
                # Vertical split: compare y position with split point
                v_left_prob = self._sigmoid_ste((split_y - y_positions), temperature)

                # Blend based on direction probability
                left_prob = (1 - split_dir) * h_left_prob + split_dir * v_left_prob
                right_prob = 1 - left_prob

                # Scale by current node probability
                node_weight = node_probs[:, node_idx]
                left_prob = left_prob * node_weight
                right_prob = right_prob * node_weight

                # Get child probabilities
                left_child_probs = self._softmax_ste(
                    self.child_selection_logits[node_idx, 0],
                    temperature,
                )
                right_child_probs = self._softmax_ste(
                    self.child_selection_logits[node_idx, 1],
                    temperature,
                )

                # Update node probabilities for next depth
                for child_idx in range(self.num_values):
                    child_left_weight = left_prob * left_child_probs[child_idx]
                    child_right_weight = right_prob * right_child_probs[child_idx]
                    next_probs[:, child_idx] += child_left_weight + child_right_weight

                    # Update bounding boxes based on split
                    # For horizontal splits (direction near 0)
                    # Left child: min_x stays same, max_x becomes split_x
                    # Right child: min_x becomes split_x, max_x stays same
                    h_left_box_min_x = min_x
                    h_left_box_max_x = split_x
                    h_right_box_min_x = split_x
                    h_right_box_max_x = max_x

                    # For vertical splits (direction near 1)
                    # Left child: min_y stays same, max_y becomes split_y
                    # Right child: min_y becomes split_y, max_y stays same
                    v_left_box_min_y = min_y
                    v_left_box_max_y = split_y
                    v_right_box_min_y = split_y
                    v_right_box_max_y = max_y

                    # Blend the bounding boxes based on direction
                    left_box_min_x = min_x
                    left_box_max_x = (1 - split_dir) * split_x + split_dir * max_x
                    left_box_min_y = min_y
                    left_box_max_y = split_dir * split_y + (1 - split_dir) * max_y

                    right_box_min_x = (1 - split_dir) * split_x + split_dir * min_x
                    right_box_max_x = max_x
                    right_box_min_y = split_dir * split_y + (1 - split_dir) * min_y
                    right_box_max_y = max_y

                    # Update the bounding boxes with weighted averaging
                    child_weight = child_left_weight + child_right_weight
                    weight_sum += child_weight

                    # Accumulate weighted bounding boxes
                    next_min_x += (
                        left_box_min_x * child_left_weight
                        + right_box_min_x * child_right_weight
                    )
                    next_max_x += (
                        left_box_max_x * child_left_weight
                        + right_box_max_x * child_right_weight
                    )
                    next_min_y += (
                        left_box_min_y * child_left_weight
                        + right_box_min_y * child_right_weight
                    )
                    next_max_y += (
                        left_box_max_y * child_left_weight
                        + right_box_max_y * child_right_weight
                    )

            # Normalize probabilities and bounding boxes
            sum_probs = next_probs.sum(dim=1, keepdim=True).clamp(min=1e-10)
            node_probs = next_probs / sum_probs

            # Normalize the accumulated bounding boxes
            weight_sum = weight_sum.clamp(min=1e-10)
            min_x = next_min_x / weight_sum
            max_x = next_max_x / weight_sum
            min_y = next_min_y / weight_sum
            max_y = next_max_y / weight_sum

        # Convert node probabilities into class probabilities
        class_probs = torch.matmul(
            node_probs,
            self._softmax_ste(self.class_logits, temperature, dim=1),
        )

        if hard:
            classes = torch.argmax(class_probs, dim=1)
            return classes.reshape(self.num_points_x, self.num_points_y)
        else:
            return class_probs.reshape(
                self.num_points_x, self.num_points_y, self.num_classes
            )

    def _sigmoid_ste(self, x, temperature):
        """Sigmoid with straight-through estimator for different temperatures"""
        if self.ste:
            # Forward pass with t_fwd
            y_fwd = torch.sigmoid(x / self.temperature_forward)

            # In autograd, use t_bwd for the gradient calculation
            if self.training:
                # Create a custom straight-through estimator
                y_bwd = torch.sigmoid(x / temperature)
                return y_fwd.detach() + y_bwd - y_bwd.detach()
            else:
                return y_fwd

        return torch.sigmoid(x / temperature)

    def _softmax_ste(self, x, temperature, dim=-1):
        """Softmax with straight-through estimator for different temperatures"""
        if self.ste:
            # Forward pass with t_fwd
            y_fwd = torch.softmax(x / self.temperature_forward, dim=dim)

            # In autograd, use t_bwd for the gradient calculation
            if self.training:
                # Create a custom straight-through estimator
                y_bwd = torch.softmax(x / temperature, dim=dim)
                return y_fwd.detach() + y_bwd - y_bwd.detach()
            else:
                return y_fwd
        return torch.softmax(x / temperature, dim=dim)

    def track_iteration(self, iteration: int, recon_loss: torch.Tensor):
        self.history["loss"].append(recon_loss.item())
        self.history["iterations"].append(iteration)

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

    def to_non_diff(self) -> "Fractal2DNonDiff":
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
            dtype=self.dtype,
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

        for value_idx in range(self.num_values):
            left_child = torch.argmax(self.child_selection_probs[value_idx, 0]).item()
            right_child = torch.argmax(self.child_selection_probs[value_idx, 1]).item()

            # Update child indices
            non_diff_model.child_indices[0, value_idx] = left_child
            non_diff_model.child_indices[1, value_idx] = right_child

        return non_diff_model
