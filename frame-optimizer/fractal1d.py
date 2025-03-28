import matplotlib.pyplot as plt
import torch
from torch import nn
from sa import PerturbableParameter, ClassMatricesParameter


class Fractal1D(nn.Module):
    """PyTorch module for fractal function with direct recursive computation."""

    def __init__(
        self,
        num_values: int = 3,
        smoothing_width: float = 0.1,
        num_dupes: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,  # Specify default dtype
    ):
        super().__init__()
        self.num_values = num_values
        self.smoothing_width = smoothing_width
        self.num_dupes = num_dupes
        self.num_values_with_dupes = self.num_values * self.num_dupes
        self.device = device
        self.dtype = dtype

        # Only split_points need to be differentiable
        self.split_points_param = PerturbableParameter(
            torch.rand(self.num_values_with_dupes, device=device, dtype=self.dtype)
            * 0.5
            + 0.25,
            name="split_points",
        )

        # Non-differentiable values
        self.values_param = torch.linspace(
            0, 1, num_values, device=device, dtype=self.dtype
        )

        # Non-differentiable matrix parameters
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

        left_mat = torch.rand_like(left_mat)
        right_mat = torch.rand_like(right_mat)

        matrices = torch.stack([left_mat, right_mat]).to(
            device=device, dtype=self.dtype
        )
        self.matrices_param = ClassMatricesParameter(
        # self.matrices_param = nn.Parameter(
            matrices,
            num_values=self.num_values,
            requires_grad=True,
        )

        # Create branch selection masks
        # These masks will replicate ValueMatricesParameter behavior
        self.left_branch_mask = torch.zeros(
            self.num_values_with_dupes,
            self.num_values_with_dupes,
            device=device,
            dtype=torch.bool,
        )
        self.right_branch_mask = torch.zeros(
            self.num_values_with_dupes,
            self.num_values_with_dupes,
            device=device,
            dtype=torch.bool,
        )

        # Setup valid transition patterns similar to ValueMatricesParameter
        # For each parent node, determine valid child nodes based on modulo constraints
        for row in range(self.num_values_with_dupes):
            # Get valid positions (exclude diagonal)
            valid_positions = list(range(self.num_values_with_dupes))
            if row in valid_positions:
                valid_positions.remove(row)  # Exclude self-transitions

            # For one matrix, apply modulo constraint: same class index (parent_idx % num_values == child_idx % num_values)
            valid_modulo_positions = [
                pos
                for pos in valid_positions
                if (pos % self.num_values) == (row % self.num_values) and pos != row
            ]

            # For the other matrix, use any valid position except those with modulo constraint
            valid_other_positions = [
                pos for pos in valid_positions if pos not in valid_modulo_positions
            ]

            # Randomly decide which matrix gets the modulo constraint
            if torch.rand(1).item() < 0.5:
                # Left matrix gets modulo constraint
                for pos in valid_modulo_positions:
                    self.left_branch_mask[row, pos] = True
                for pos in valid_other_positions:
                    self.right_branch_mask[row, pos] = True
            else:
                # Right matrix gets modulo constraint
                for pos in valid_modulo_positions:
                    self.right_branch_mask[row, pos] = True
                for pos in valid_other_positions:
                    self.left_branch_mask[row, pos] = True

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
            "predicted_y": [],
            "iterations": [],
        }

    @property
    def split_points(self) -> torch.Tensor:
        return torch.sigmoid(torch.clamp(self.split_points_param, 0, 1) * 4 - 2)

    @property
    def values(self) -> torch.Tensor:
        return (self.values_param * 3.0 + 1.0).repeat(self.num_dupes)

    @property
    def left_matrix(self) -> torch.Tensor:
        return self.matrices_param[0]

    @property
    def right_matrix(self) -> torch.Tensor:
        return self.matrices_param[1]

    def sigmoid_blend(
        self, x: torch.Tensor, center: float, width: float
    ) -> torch.Tensor:
        """Compute sigmoid transition function."""
        scale = 20.0 / width
        return torch.sigmoid((x - center) * scale)

    def forward(
        self,
        x: torch.Tensor,
        max_depth: int,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """
        Directly compute fractal function values through recursion.

        Args:
            x: Input tensor of shape [batch_size]
            max_depth: Maximum recursion depth
            use_argmax: Whether to use argmax (deterministic) instead of softmax (probabilistic)

        Returns:
            Output tensor of shape [batch_size, num_values] containing either:
            - One-hot encoded vectors (if use_argmax=True)
            - Logits (if use_argmax=False)
        """

        # Test simple computation to check gradient flow
        test_sum = self.left_matrix.sum() + self.right_matrix.sum()

        # Compute values for the full function with domain (0,1)
        result = self._evaluate_recursive(x, 0, 0.0, 1.0, 0, max_depth, use_argmax)

        return result

    def _evaluate_recursive(
        self,
        x: torch.Tensor,
        parent_idx: int,
        start: float,
        end: float,
        depth: int,
        max_depth: int,
        use_argmax: bool,
    ) -> torch.Tensor:
        """
        Recursively evaluate fractal function at points x.
        """
        # At maximum depth, return one-hot encoded value or logit
        if depth == max_depth:
            batch_size = x.shape[0]
            value_idx = parent_idx % self.num_values

            if use_argmax:
                result = torch.zeros(
                    (batch_size, self.num_values), device=x.device, dtype=self.dtype
                )
                result[:, value_idx] = 1.0
            else:
                result = torch.full(
                    (batch_size, self.num_values),
                    -10.0,
                    device=x.device,
                    dtype=self.dtype,
                )
                result[:, value_idx] = 10.0

            return result

        # Calculate split position and width
        split = self.split_points[parent_idx]
        mid = start + split * (end - start)
        segment_size = end - start
        trans_width = self.smoothing_width * segment_size

        # Compute blend weights using sigmoid
        t = self.sigmoid_blend(x, mid, trans_width)

        if use_argmax:
            # Deterministic path - just get the best child for each branch
            left_child_idx = torch.argmax(self.left_matrix[parent_idx]).item()
            right_child_idx = torch.argmax(self.right_matrix[parent_idx]).item()

            left_values = self._evaluate_recursive(
                x, left_child_idx, start, mid, depth + 1, max_depth, use_argmax
            )
            right_values = self._evaluate_recursive(
                x, right_child_idx, mid, end, depth + 1, max_depth, use_argmax
            )

            t = t.unsqueeze(1)
            result = (1 - t) * left_values + t * right_values
        else:
            # Optimized probabilistic path - only evaluate top K branches
            top_k = 2  # Only consider top 3 branches

            # Apply branch masks to restrict valid transitions
            left_matrix_masked = torch.where(
                self.left_branch_mask[parent_idx],
                self.left_matrix[parent_idx],
                torch.tensor(-1e10, device=x.device, dtype=self.dtype),
            )
            right_matrix_masked = torch.where(
                self.right_branch_mask[parent_idx],
                self.right_matrix[parent_idx],
                torch.tensor(-1e10, device=x.device, dtype=self.dtype),
            )

            # Get indices and probabilities for top-k left and right branches
            left_values, left_indices = torch.topk(left_matrix_masked, top_k)
            right_values, right_indices = torch.topk(right_matrix_masked, top_k)

            # Normalize to get probabilities just for the top-k
            left_probs = torch.nn.functional.softmax(left_values * 10.0, dim=0)
            right_probs = torch.nn.functional.softmax(right_values * 10.0, dim=0)

            # Initialize result tensors
            batch_size = x.shape[0]
            left_result = torch.zeros(
                (batch_size, self.num_values), device=x.device, dtype=self.dtype
            )
            right_result = torch.zeros(
                (batch_size, self.num_values), device=x.device, dtype=self.dtype
            )

            # Process top-k left branches
            for i in range(top_k):
                if i < left_indices.shape[0]:  # Safety check
                    idx = left_indices[i].item()
                    prob = left_probs[i]

                    # Only evaluate if probability is significant
                    if prob > 1e-5:
                        child_values = self._evaluate_recursive(
                            x, idx, start, mid, depth + 1, max_depth, use_argmax
                        )
                        left_result += prob * child_values

            # Process top-k right branches
            for i in range(top_k):
                if i < right_indices.shape[0]:  # Safety check
                    idx = right_indices[i].item()
                    prob = right_probs[i]

                    # Only evaluate if probability is significant
                    if prob > 1e-5:
                        child_values = self._evaluate_recursive(
                            x, idx, mid, end, depth + 1, max_depth, use_argmax
                        )
                        right_result += prob * child_values

            # Blend the functions
            t = t.unsqueeze(1)
            result = (1 - t) * left_result + t * right_result

        return result

    def track_iteration(
        self,
        iteration: int,
        recon_loss: torch.Tensor,
    ):
        self.history["loss"].append(recon_loss.item())

        self.history["split_points"].append(
            self.split_points_param.detach().cpu().numpy()
        )
        self.history["values"].append(self.values_param.detach().cpu().numpy())
        self.history["left_matrix"].append(self.left_matrix.detach().cpu().numpy())
        self.history["right_matrix"].append(self.right_matrix.detach().cpu().numpy())
        self.history["iterations"].append(iteration)

    def plot_history(
        self,
        depths: list[int],
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        viz_x: torch.Tensor,
    ):
        # Evaluate at dense points for visualization
        with torch.no_grad():
            eval_outputs = {
                #depth: self(viz_x, depth, use_argmax=True) for depth in depths
                depth: self(depth) for depth in depths
            }

            # Convert outputs from one-hot/logits to scalar class values
            eval_ys = {}
            for depth, outputs in eval_outputs.items():
                # If one-hot encoded (from use_argmax=True), get the class index
                # For visualization, we convert class indices to scalar values (0 to num_values-1)
                #class_indices = outputs.argmax(dim=1)
                class_indices = outputs
                # Scale indices to the 0-1 range for better visualization
                eval_ys[depth] = class_indices.float() / (self.num_values - 1)

        iteration = self.history["iterations"][-1]

        # Create figure with subplots
        num_charts = 1 + len(depths)
        fig, axs = plt.subplots(num_charts, 1, figsize=(6, 1.1 * num_charts))
        fig.tight_layout(pad=0.0)

        # Plot losses
        axs[0].plot(self.history["loss"], label="Reconstruction Loss", color="blue")
        axs[0].set_title(f"Loss (Iteration {iteration})")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss Value")
        axs[0].grid(True)

        # For target_y, convert to scalar values if it's one-hot or class indices
        if len(target_y.shape) > 1 and target_y.shape[1] > 1:
            # If target_y is one-hot encoded
            target_y_plot = target_y.argmax(dim=1).float() / (self.num_values - 1)
        elif target_y.dtype == torch.long:
            # If target_y contains class indices
            target_y_plot = target_y.float() / (self.num_values - 1)
        else:
            # Assume target_y is already in scalar form
            target_y_plot = target_y

        # Plot current fit vs target
        for i, depth in enumerate(depths):
            ax_i = i + 1
            axs[ax_i].scatter(
                target_x.detach().cpu(),
                target_y_plot.detach().cpu(),
                alpha=0.7,
                label="Target Points",
                color="black",
            )
            axs[ax_i].plot(
                viz_x.detach().cpu(),
                eval_ys[depth].detach().cpu(),
                label=f"Fit (depth={depth})",
                color="blue",
                linewidth=2,
            )

            axs[ax_i].set_title(f"Target vs. Fit (depth={depth})")
            axs[ax_i].set_ylabel("Class (scaled)")
            axs[ax_i].grid(True)

        plt.tight_layout()
        plt.show()
