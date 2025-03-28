import torch
from torch import nn
import numpy as np
from sa import PerturbableParameter, ChildIndicesParameter, ClassesParameter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Fractal2DNonDiff(nn.Module):
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
        split_points = (
            torch.rand(self.num_values, device=device, dtype=dtype) * 0.6 + 0.2
        )
        self.split_points = PerturbableParameter(
            split_points, requires_grad=False, name="split_points", min=0.1, max=0.9
        )

        # Initialize split directions (0=horizontal, 1=vertical)
        split_directions = torch.randint(
            0, 2, (self.num_values,), device=device, dtype=torch.long
        )
        self.split_directions = PerturbableParameter(
            split_directions,
            requires_grad=False,  # Discrete parameter, no gradient
            name="split_directions",
            min=0,
            max=1,
        )

        # Initialize classes for each value index
        # Ensure each class has at least 2 values
        min_values_per_class = 2
        values_per_class = num_values // num_classes
        extra_values = num_values % num_classes

        classes = []
        for class_idx in range(num_classes):
            # Determine how many values this class gets
            class_values = values_per_class + (1 if class_idx < extra_values else 0)
            # Add at least min_values_per_class values for this class
            classes.extend([class_idx] * max(class_values, min_values_per_class))

        # Shuffle the classes (in case we assigned extras)
        classes = classes[:num_values]  # Ensure we have exactly num_values
        classes = torch.tensor(classes, device=device, dtype=torch.long)
        indices = torch.randperm(num_values, device=device)
        classes = classes[indices]

        self.classes = ClassesParameter(
            classes,
            num_classes=num_classes,
            requires_grad=False,  # Discrete parameter, no gradient
            name="classes",
        )

        # Initialize same_class_direction (0=left/up, 1=right/down)
        same_class_direction = torch.randint(
            0, 2, (self.num_values,), device=device, dtype=torch.long
        )
        self.same_class_direction = PerturbableParameter(
            same_class_direction,
            requires_grad=False,  # Discrete parameter, no gradient
            name="same_class_direction",
            min=0,
            max=1,
        )

        left_children = torch.zeros(self.num_values, dtype=torch.long, device=device)
        right_children = torch.zeros(self.num_values, dtype=torch.long, device=device)

        # Set children indices respecting class constraints
        for i in range(self.num_values):
            row_class = self.classes[i].item()

            # Find indices of values with the same class
            same_class_indices = (self.classes == row_class).nonzero().flatten()
            same_class_indices = same_class_indices[
                same_class_indices != i
            ]  # Exclude self

            # Find indices of values with different classes
            diff_class_indices = (self.classes != row_class).nonzero().flatten()

            # Choose a random index for each direction according to same_class_direction
            if self.same_class_direction[i] == 0:  # Left/up maintains class
                left_idx = same_class_indices[
                    torch.randint(0, len(same_class_indices), (1,))
                ].item()
                right_idx = diff_class_indices[
                    torch.randint(0, len(diff_class_indices), (1,))
                ].item()
            else:  # Right/down maintains class
                left_idx = diff_class_indices[
                    torch.randint(0, len(diff_class_indices), (1,))
                ].item()
                right_idx = same_class_indices[
                    torch.randint(0, len(same_class_indices), (1,))
                ].item()

            left_children[i] = left_idx
            right_children[i] = right_idx

        children = torch.stack([left_children, right_children])

        # Create the children parameter that knows about class constraints
        self.child_indices = ChildIndicesParameter(
            children,
            num_classes=self.num_classes,
            same_class_direction=self.same_class_direction,
            classes=self.classes,
            requires_grad=False,  # Discrete indices, no gradient
            name="child_indices",
        )

        # Initialize history dictionary
        self.reset_history()

    def reset_history(self):
        """Reset the history tracking."""
        self.history = {
            "loss": [],
            "split_points": [],
            "values": [],
            "left_children": [],
            "right_children": [],
            "iterations": [],
        }

    @property
    def left_child_indices(self) -> torch.Tensor:
        return self.child_indices[0]

    @property
    def right_child_indices(self) -> torch.Tensor:
        return self.child_indices[1]

    def forward(self, max_depth: int) -> torch.Tensor:
        """
        Parallel implementation of fractal generation using GPU
        """
        batch_size = self.num_points_x * self.num_points_y

        # Initialize with all pixels assigned to the root node (index 0)
        current_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Use pre-computed grid points
        x_positions = self.grid_points[:, 0]
        y_positions = self.grid_points[:, 1]

        # Process all depths in sequence, but process all pixels in parallel
        for depth in range(max_depth):
            # Get split parameters for current nodes
            curr_directions = self.split_directions[current_indices]
            curr_split_points = self.split_points[current_indices]

            # Determine which side of the split each pixel falls on
            # For horizontal splits (direction=0): compare x position to split
            h_is_left = x_positions < curr_split_points

            # For vertical splits (direction=1): compare y position to split
            v_is_left = y_positions < curr_split_points

            # Blend based on direction (0=horizontal, 1=vertical)
            is_left = torch.where(curr_directions == 0, h_is_left, v_is_left)

            # Determine child indices based on which side of the split
            left_children = self.left_child_indices[current_indices]
            right_children = self.right_child_indices[current_indices]

            # Update current indices based on which side of the split
            current_indices = torch.where(is_left, left_children, right_children)

        # After processing all depths, map to final class values
        final_classes = self.classes[current_indices]

        # Reshape back to grid
        return final_classes.reshape(self.num_points_x, self.num_points_y)

    def forward_old(self, max_depth: int) -> torch.Tensor:
        """
        Parallel implementation of fractal generation using GPU
        """
        batch_size = self.num_points_x * self.num_points_y

        # Initialize with all pixels assigned to the root node (index 0)
        current_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Use pre-computed grid points
        x_positions = self.grid_points[:, 0]
        y_positions = self.grid_points[:, 1]

        # Track bounding box for each point
        # Start with the full [0,1] x [0,1] range for all points
        min_x = torch.zeros_like(x_positions)
        max_x = torch.ones_like(x_positions)
        min_y = torch.zeros_like(y_positions)
        max_y = torch.ones_like(y_positions)

        # Process all depths in sequence, but process all pixels in parallel
        for depth in range(max_depth):
            # Get split parameters for current nodes
            curr_directions = self.split_directions[current_indices]
            curr_split_points = self.split_points[current_indices]

            # Calculate split positions in normalized coordinates
            split_x = min_x + curr_split_points * (max_x - min_x)
            split_y = min_y + curr_split_points * (max_y - min_y)

            # Determine which side of the split each pixel falls on
            is_left = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            # For horizontal splits (direction=0): compare x position to split
            horizontal_mask = curr_directions == 0

            is_left[horizontal_mask] = (
                x_positions[horizontal_mask] < split_x[horizontal_mask]
            )

            # For vertical splits (direction=1): compare y position to split
            vertical_mask = curr_directions == 1
            is_left[vertical_mask] = y_positions[vertical_mask] < split_y[vertical_mask]

            # Update bounding boxes based on the split direction and which side each point falls on
            # For horizontal splits, update min_x or max_x
            new_min_x = torch.where(horizontal_mask & ~is_left, split_x, min_x)
            new_max_x = torch.where(horizontal_mask & is_left, split_x, max_x)

            # For vertical splits, update min_y or max_y
            new_min_y = torch.where(vertical_mask & ~is_left, split_y, min_y)
            new_max_y = torch.where(vertical_mask & is_left, split_y, max_y)

            min_x, max_x = new_min_x, new_max_x
            min_y, max_y = new_min_y, new_max_y

            # Determine child indices based on which side of the split
            left_children = self.left_child_indices[current_indices]
            right_children = self.right_child_indices[current_indices]

            # Update current indices based on which side of the split
            current_indices = torch.where(is_left, left_children, right_children)

        # After processing all depths, map to final class values
        final_classes = self.classes[current_indices]

        # Reshape back to grid
        return final_classes.reshape(self.num_points_x, self.num_points_y)


    def track_iteration(
        self,
        iteration: int,
        recon_loss: torch.Tensor,
    ):
        self.history["loss"].append(recon_loss.item())
        self.history["split_points"].append(self.split_points.detach().cpu().numpy())
        self.history["left_children"] = self.left_child_indices.detach().cpu().numpy()
        self.history["right_children"] = self.right_child_indices.detach().cpu().numpy()
        self.history["iterations"].append(iteration)

    def plot_history(
        self,
        depths: list[int],
        target_image: torch.Tensor,
        color_map: np.ndarray | None = None,
    ):
        """Visualize the current fractal structure alongside the target image with loss history"""
        iteration = self.history["iterations"][-1] if len(self.history["iterations"]) else 0

        # Generate outputs at different depths
        with torch.no_grad():
            outputs = {depth: self(depth) for depth in depths}

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

    def to_diff(self) -> 'Fractal2DDiff':
        """Convert this non-differentiable fractal model to a differentiable version."""
        from fractal2d_diff import Fractal2DDiff

        # Create a new differentiable model with the same dimensions
        diff_model = Fractal2DDiff(
            num_classes=self.num_classes,
            num_values=self.num_values,
            num_points_x=self.num_points_x,
            num_points_y=self.num_points_y,
            device=self.device,
            dtype=self.dtype
        )

        # Transfer split points directly
        diff_model.split_points.data.copy_(self.split_points.data)

        # Convert discrete split directions to continuous logits
        # Use high values to ensure sigmoid gives values close to 0 or 1
        direction_logits = torch.where(
            self.split_directions.data == 0,
            torch.tensor(-10.0, device=self.device, dtype=self.dtype),
            torch.tensor(10.0, device=self.device, dtype=self.dtype)
        )
        diff_model.split_directions_logits.data.copy_(direction_logits)

        # Initialize class logits to create a sharp distribution
        # For each value, create a one-hot-like distribution of class probabilities
        for value_idx in range(self.num_values):
            class_idx = self.classes[value_idx].item()
            # Set high logit for the assigned class, low for others
            diff_model.class_logits.data[value_idx] = -10.0
            diff_model.class_logits.data[value_idx, class_idx] = 10.0

        # Initialize child selection weights based on current child indices
        for value_idx in range(self.num_values):
            # Get current left and right child indices
            left_child = self.left_child_indices[value_idx].item()
            right_child = self.right_child_indices[value_idx].item()

            # Initialize logits (negative values for all, high positive for selected child)
            diff_model.child_selection_logits.data[value_idx, 0] = -10.0
            diff_model.child_selection_logits.data[value_idx, 1] = -10.0
            diff_model.child_selection_logits.data[value_idx, 0, left_child] = 10.0
            diff_model.child_selection_logits.data[value_idx, 1, right_child] = 10.0

        return diff_model
