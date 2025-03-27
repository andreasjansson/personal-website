import torch
from torch import nn
import numpy as np
from sa import PerturbableParameter, ValueMatricesParameter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Fractal2DNonDiff(nn.Module):
    """
    2D fractal function with both horizontal and vertical splits.
    """

    def __init__(
        self,
        num_values: int,
        num_dupes: int = 2,
        num_points_x: int = 100,
        num_points_y: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # TODO:
        # * replace num_values and num_dupes with num_classes and num_values
        # * assert num_values >= num_classes * 2
        # * replace self.values with self.classes (long)
        # * make a same_class_direction (left/up or right/down) (int, 0/1) instead of mod constraints in perturb_row
        # * make self.classes, self.split_points, self.split_directions, self.same_class_direction perturbable parameters (with custom perturb classes?)
        # * initialize self.classes, self.split_points, self.split_directions, self.same_class_direction randomly (replacing the current initialization), as long as there are at least two values with the same class
        # That way more common classes can have more values to create complex shapes

        assert num_dupes > 0
        assert num_dupes % 2 == 0

        self.num_values = num_values
        self.num_dupes = num_dupes
        self.num_values_with_dupes = self.num_values * self.num_dupes
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

        base_split_points = torch.linspace(
            0, 1, (num_dupes // 2) + 2, device=device, dtype=self.dtype
        )[1:-1]
        mid_idx = len(base_split_points) // 2
        base_split_points = torch.cat(
            [
                base_split_points[mid_idx : mid_idx + 1],
                base_split_points[:mid_idx],
                base_split_points[mid_idx + 1 :],
            ]
        )
        split_points = []
        split_directions = []
        values = []
        for dupe in range(num_dupes):
            split_point = base_split_points[dupe // 2]
            split_direction = dupe % 2
            for value in range(num_values):
                split_points.append(split_point)
                split_directions.append(split_direction)
                values.append(value)

        self.split_points_param = torch.tensor(split_points).to(device=device, dtype=dtype)
        self.split_directions = torch.tensor(split_directions).to(device=device, dtype=dtype)
        self.values = torch.tensor(values).to(device=device, dtype=dtype)

        # Initialize tree transition matrices (for both left/right or top/bottom)
        left_mat = torch.zeros(
            self.num_values_with_dupes, self.num_values_with_dupes, dtype=self.dtype
        )
        right_mat = torch.zeros(
            self.num_values_with_dupes, self.num_values_with_dupes, dtype=self.dtype
        )

        # Make one random element per row active
        for i in range(self.num_values_with_dupes):
            left_mat[i, torch.randint(0, self.num_values_with_dupes, (1,))] = 1.0
            right_mat[i, torch.randint(0, self.num_values_with_dupes, (1,))] = 1.0

        matrices = torch.stack([left_mat, right_mat]).to(
            device=device, dtype=self.dtype
        )
        self.matrices_param = ValueMatricesParameter(
            matrices,
            num_values=self.num_values,
            requires_grad=True,
        )

        # Initialize history dictionary
        self.reset_history()

    def reset_history(self):
        """Reset the history tracking."""
        self.history = {
            "loss": [],
            "split_points": [],
            "values": [],
            "left_matrix": [],
            "right_matrix": [],
            "iterations": [],
        }

    @property
    def split_points(self) -> torch.Tensor:
        # return torch.sigmoid(torch.clamp(self.split_points_param, 0, 1) * 4 - 2)
        return self.split_points_param

    @property
    def left_matrix(self) -> torch.Tensor:
        return self.matrices_param[0]

    @property
    def right_matrix(self) -> torch.Tensor:
        return self.matrices_param[1]

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
            left_children = torch.argmax(self.left_matrix[current_indices], dim=1)
            right_children = torch.argmax(self.right_matrix[current_indices], dim=1)

            # Update current indices based on which side of the split
            current_indices = torch.where(is_left, left_children, right_children)

        # After processing all depths, map to final values
        final_values = self.values[current_indices]

        # Reshape back to grid
        return final_values.reshape(self.num_points_x, self.num_points_y)

    def forward_old(self, max_depth: int) -> torch.Tensor:
        """
        Generate a 2D fractal image.

        Args:
            max_depth: Maximum recursion depth

        Returns:
            Tensor of shape [num_points_x, num_points_y] containing class indices
        """
        # Pre-allocate result tensor
        result = torch.zeros(
            (self.num_points_x, self.num_points_y),
            device=self.device,
            dtype=self.dtype,
        )

        # Start with the entire grid
        self.generate_recursive(
            parent_idx=0,
            x_start=0,
            x_end=self.num_points_x,
            y_start=0,
            y_end=self.num_points_y,
            depth=0,
            max_depth=max_depth,
            result=result,
        )

        return result

    def generate_recursive(
        self,
        parent_idx: int,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        depth: int,
        max_depth: int,
        result: torch.Tensor,
    ) -> None:
        """
        Recursively evaluate 2D fractal function using integer indices.
        """
        # Skip if empty region
        if x_start >= x_end or y_start >= y_end:
            return

        # At maximum depth, directly set the values
        if depth == max_depth:
            value_idx = parent_idx % self.num_values
            result[x_start:x_end, y_start:y_end] = value_idx
            return

        # Get split direction (0=horizontal, 1=vertical)
        split_direction = self.split_directions[parent_idx]

        # Calculate split position
        split_ratio = self.split_points[parent_idx]

        # Get child indices
        left_child_idx = torch.argmax(self.left_matrix[parent_idx]).item()
        right_child_idx = torch.argmax(self.right_matrix[parent_idx]).item()

        if split_direction == 0:  # Horizontal split (left/right)
            # Convert ratio to integer split point
            x_split = x_start + int((x_end - x_start) * split_ratio)
            x_split = max(x_start + 1, min(x_split, x_end - 1))  # Ensure valid split

            # Process left side
            self.generate_recursive(
                parent_idx=left_child_idx,
                x_start=x_start,
                x_end=x_split,
                y_start=y_start,
                y_end=y_end,
                depth=depth + 1,
                max_depth=max_depth,
                result=result,
            )

            # Process right side
            self.generate_recursive(
                parent_idx=right_child_idx,
                x_start=x_split,
                x_end=x_end,
                y_start=y_start,
                y_end=y_end,
                depth=depth + 1,
                max_depth=max_depth,
                result=result,
            )
        else:  # Vertical split (top/bottom)
            # Convert ratio to integer split point
            y_split = y_start + int((y_end - y_start) * split_ratio)
            y_split = max(y_start + 1, min(y_split, y_end - 1))  # Ensure valid split

            # Process top side
            self.generate_recursive(
                parent_idx=left_child_idx,  # We reuse left_matrix for top
                x_start=x_start,
                x_end=x_end,
                y_start=y_start,
                y_end=y_split,
                depth=depth + 1,
                max_depth=max_depth,
                result=result,
            )

            # Process bottom side
            self.generate_recursive(
                parent_idx=right_child_idx,  # We reuse right_matrix for bottom
                x_start=x_start,
                x_end=x_end,
                y_start=y_split,
                y_end=y_end,
                depth=depth + 1,
                max_depth=max_depth,
                result=result,
            )

    def track_iteration(
        self,
        iteration: int,
        recon_loss: torch.Tensor,
    ):
        self.history["loss"].append(recon_loss.item())
        self.history["split_points"].append(
            self.split_points_param.detach().cpu().numpy()
        )
        self.history["left_matrix"].append(self.left_matrix.detach().cpu().numpy())
        self.history["right_matrix"].append(self.right_matrix.detach().cpu().numpy())
        self.history["iterations"].append(iteration)

    def plot_history(
        self,
        depths: list[int],
        target_image: torch.Tensor,
        color_map: np.ndarray | None = None,
    ):
        """Visualize the current fractal structure alongside the target image with loss history"""
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
            ax_loss.set_title("Optimization Progress")
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
            colors = plt.cm.get_cmap("tab10", self.num_values)

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
            vmax=self.num_values - 1,
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
                vmax=self.num_values - 1,
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
