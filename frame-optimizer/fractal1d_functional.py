import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from func1d import generate_fractal_function
from sa import PerturbableParameter, ClassMatricesParameter


class Fractal1D(nn.Module):
    """PyTorch module wrapper for fractal function generator with integrated history tracking."""

    def __init__(
        self,
        num_values: int = 3,
        smoothing_width: float = 0.1,
        num_dupes: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.num_values = num_values
        self.smoothing_width = smoothing_width
        self.num_dupes = num_dupes
        self.num_values_with_dupes = self.num_values * self.num_dupes
        self.device = device

        # Create standard perturbable parameters
        self.split_points_param = PerturbableParameter(
            torch.rand(self.num_values_with_dupes, device=device) * 0.5 + 0.25,
            name="split_points",
        )
        # self.values_param = PerturbableParameter(torch.rand(num_values, device=device))
        #self.values_param = PerturbableParameter(
        #    torch.linspace(0, 1, num_values, device=device)
        #)
        self.values_param = torch.linspace(0, 1, num_values, device=device)

        # Create categorical matrix parameters for left and right matrices
        # Initialize with inactive values
        left_mat = torch.zeros(self.num_values_with_dupes, self.num_values_with_dupes)
        right_mat = torch.zeros(self.num_values_with_dupes, self.num_values_with_dupes)

        # Make one random element per row active
        for i in range(self.num_values_with_dupes):
            left_mat[i, torch.randint(0, self.num_values_with_dupes, (1,))] = 1.0
            right_mat[i, torch.randint(0, self.num_values_with_dupes, (1,))] = 1.0

        matrices = torch.stack([left_mat, right_mat]).to(device)
        self.matrices_param = ClassMatricesParameter(
            matrices, num_values=self.num_values, requires_grad=False,
        )
        # self.right_matrix = right_mat

        # Initialize history dictionary
        self.reset_history()

    def reset_history(self):
        """Reset the history tracking."""
        self.history = {
            "loss": [],
            "entropy_loss": [],
            "split_points": [],
            "values": [],
            "left_matrix": [],
            "right_matrix": [],
            "predicted_y": [],
            "iterations": [],
        }

    @property
    def split_points(self) -> torch.Tensor:
        return torch.sigmoid(self.split_points_param * 4 - 2)

    @property
    def values(self) -> torch.Tensor:
        return (self.values_param * 3.0 + 1.0).repeat(self.num_dupes)

    @property
    def left_matrix(self) -> torch.Tensor:
        return self.matrices_param[0]

    @property
    def right_matrix(self) -> torch.Tensor:
        return self.matrices_param[1]

    def get_function(self, max_depth: int, use_argmax: bool):
        func = generate_fractal_function(
            split_points=self.split_points,
            values=self.values,
            left_children_matrix=self.left_matrix,
            right_children_matrix=self.right_matrix,
            max_depth=max_depth,
            smoothing_width=self.smoothing_width,
            use_argmax=use_argmax,
        )
        return func

    def forward(
        self,
        x: torch.Tensor,
        max_depth: int,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """
        Evaluate the fractal function at the given points.

        Args:
            x: Input tensor of shape [batch_size]
            use_argmax: Whether to use argmax (deterministic) instead of softmax (probabilistic)
            current_max_depth: Override the max_depth for this forward pass

        Returns:
            Output tensor of shape [batch_size]
        """
        func = self.get_function(max_depth=max_depth, use_argmax=use_argmax)

        # Evaluate the function
        return func(x)

    def get_entropy_loss(self) -> torch.Tensor:
        """Calculate entropy loss for the left and right matrices."""
        left_probs = torch.nn.functional.softmax(self.left_matrix * 10.0, dim=1)
        right_probs = torch.nn.functional.softmax(self.right_matrix * 10.0, dim=1)

        left_entropy = -torch.sum(
            left_probs * torch.log(left_probs + 1e-10), dim=1
        ).mean()
        right_entropy = -torch.sum(
            right_probs * torch.log(right_probs + 1e-10), dim=1
        ).mean()

        return (left_entropy + right_entropy) / 2.0

    def track_iteration(
        self,
        iteration: int,
        recon_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
    ):
        self.history["loss"].append(recon_loss.item())
        self.history["entropy_loss"].append(entropy_loss.item())

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
            # current_y = self.forward(viz_x, max_depth, use_argmax=False)
            eval_ys = {depth: self(viz_x, depth, use_argmax=True) for depth in depths}

        iteration = self.history["iterations"][-1]

        # Create figure with subplots
        # fig, axs = plt.subplots(3, 1, figsize=(6, 4))
        num_charts = 1 + len(depths)
        fig, axs = plt.subplots(num_charts, 1, figsize=(6, 1.1 * num_charts))
        fig.tight_layout(pad=0.0)

        # Plot losses
        axs[0].plot(self.history["loss"], label="Reconstruction Loss", color="blue")
        axs[0].set_title(f"Loss (Iteration {iteration})")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss Value")
        axs[0].grid(True)

        # Plot current fit vs target (softmax version)
        for i, depth in enumerate(depths):
            ax_i = i + 1
            axs[ax_i].scatter(
                target_x.detach().cpu(),
                target_y.detach().cpu(),
                alpha=0.7,
                label="Target Points",
                color="black",
            )
            axs[ax_i].plot(
                viz_x.detach().cpu(),
                eval_ys[depth].detach().cpu(),
                label="Softmax Fit",
                color="blue",
                linewidth=2,
            )

            axs[ax_i].set_title(f"Target vs. Softmax Fit (depth={depth})")
            axs[ax_i].set_ylabel("y")
            axs[ax_i].grid(True)

        # Plot argmax version
        # axs[2].scatter(
        #     target_x, target_y, alpha=0.7, label="Target Points", color="black"
        # )
        # axs[2].plot(viz_x, eval_y, label="Argmax Fit", color="red", linewidth=2)
        # axs[2].set_title("Target vs. Argmax Fit")
        # axs[2].set_ylabel("y")
        # axs[2].grid(True)

        plt.tight_layout()
        plt.show()
