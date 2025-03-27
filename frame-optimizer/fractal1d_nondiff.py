from torch import nn
import torch

from fractal1d import Fractal1D


class Fractal1DNonDiff(Fractal1D):
    """Highly optimized version of Fractal1D for simulated annealing with fixed output size."""

    def __init__(
        self,
        num_values: int = 3,
        num_dupes: int = 2,
        num_points: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            num_values=num_values, num_dupes=num_dupes, device=device, dtype=dtype
        )
        self.num_points = num_points
        # Pre-generate fixed x values
        self.fixed_x = torch.linspace(0, 1, num_points, device=device, dtype=dtype)

    def forward(self, max_depth: int) -> torch.Tensor:
        """
        Generate fixed-size output using pre-determined x values.

        Args:
            max_depth: Maximum recursion depth
            use_argmax: Whether to use deterministic path

        Returns:
            Tensor of shape [num_points] containing class indices (0 to num_values-1)
        """
        # Pre-allocate result tensor for the entire range of points
        result = torch.zeros(
            self.num_points,
            device=self.fixed_x.device,
            dtype=self.dtype,
        )

        # Compute the function for the entire range at once
        self.generate_recursive(
            parent_idx=0,
            domain_start=0.0,
            domain_end=1.0,
            point_start_idx=0,
            point_end_idx=self.num_points,
            depth=0,
            max_depth=max_depth,
            result=result,
        )

        return result

    def generate_recursive(
        self,
        parent_idx: int,
        domain_start: float,
        domain_end: float,
        point_start_idx: int,
        point_end_idx: int,
        depth: int,
        max_depth: int,
        result: torch.Tensor,
    ) -> None:
        """
        Recursively evaluate fractal function, writing directly to result tensor.

        Args:
            parent_idx: Index of parent node
            domain_start: Start of domain range
            domain_end: End of domain range
            point_start_idx: Start index in points array
            point_end_idx: End index in points array
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            result: Output tensor to write results into
        """
        # Skip if no points in this segment
        if point_start_idx >= point_end_idx:
            return

        # At maximum depth, directly set the values
        if depth == max_depth:
            value_idx = parent_idx % self.num_values
            result[point_start_idx:point_end_idx] = value_idx
            return

        # Calculate split position
        split = self.split_points[parent_idx]
        domain_mid = domain_start + split * (domain_end - domain_start)

        # Get children indices using deterministic selection
        left_child_idx = torch.argmax(self.left_matrix[parent_idx]).item()
        right_child_idx = torch.argmax(self.right_matrix[parent_idx]).item()

        # Find the split point in our array
        # This is a binary search to find where fixed_x crosses domain_mid
        split_idx = point_start_idx
        while split_idx < point_end_idx and self.fixed_x[split_idx] < domain_mid:
            split_idx += 1

        # Process left branch (points from start to split)
        if split_idx > point_start_idx:
            self.generate_recursive(
                parent_idx=left_child_idx,
                domain_start=domain_start,
                domain_end=domain_mid,
                point_start_idx=point_start_idx,
                point_end_idx=split_idx,
                depth=depth + 1,
                max_depth=max_depth,
                result=result,
            )

        # Process right branch (points from split to end)
        if split_idx < point_end_idx:
            self.generate_recursive(
                parent_idx=right_child_idx,
                domain_start=domain_mid,
                domain_end=domain_end,
                point_start_idx=split_idx,
                point_end_idx=point_end_idx,
                depth=depth + 1,
                max_depth=max_depth,
                result=result,
            )
