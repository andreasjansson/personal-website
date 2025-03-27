import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam

from fractal1d import Fractal1D
from fractal1d_nondiff import Fractal1DNonDiff
from sa import SimulatedAnnealing
from optimizer import Optimizer, OptimizerPart


def optimize_fractal_function(
    model: Fractal1D,
    max_depth: int,
    target_y: torch.Tensor,
    target_x: torch.Tensor,
    num_iterations: int = 2000,
    adam_lr: float = 0.01,
    sa_initial_temp: float = 1.0,
    sa_min_temp: float = 0.1,
    sa_num_restarts: int = 3,
    sa_step_size: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    plot_every_n: int = 50,
    show_progress: bool = True,
) -> Fractal1D:
    target_x = target_x.to(device=device, dtype=model.dtype)
    target_y = target_y.to(device=device, dtype=torch.long)
    num_viz_points = len(target_y)
    viz_x = torch.linspace(0, 1, num_viz_points, device=device, dtype=model.dtype)

    parts = [
        # optimize the first value
        OptimizerPart(Adam([model.values_param], lr=adam_lr), max_depth=0),
    ]

    # for depth in range(1, max_depth + 1):
    # depth = max_depth
    # for _ in range(3):
    #     parts += [
    #         # optimize matrices split points, and values at levels 1-5
    #         OptimizerPart(
    #             SimulatedAnnealing(model.parameters()),
    #             max_depth=depth,
    #             weight=5,
    #         ),
    #         OptimizerPart(
    #             Adam([model.split_points_param, model.values_param], lr=adam_lr),
    #             max_depth=depth,
    #         ),
    #     ]

    ## TODO
    # parts = [OptimizerPart(SimulatedAnnealing(model.parameters()), max_depth=max_depth)]
    # parts = [OptimizerPart(SimulatedAnnealing([model.split_points_param, model.left_matrix, model.right_matrix]), max_depth=max_depth)]
    parts = [
        OptimizerPart(
            lambda num_iterations: SimulatedAnnealing(
                [model.split_points_param, model.matrices_param],
                num_iterations=num_iterations,
                initial_temp=sa_initial_temp,
                num_restarts=sa_num_restarts,
                min_temp=sa_min_temp,
                step_size=sa_step_size,
            ),
            max_depth=max_depth,
            weight=50,
        ),
        # OptimizerPart(
        #    Adam([model.split_points_param, model.values_param], lr=adam_lr),
        #    max_depth=max_depth,
        # ),
    ]

    # parts = []
    # # for _ in range(num_iterations // 20):
    # for _ in range(2):
    #     parts += [
    #         OptimizerPart(
    #             lambda num_iterations: SimulatedAnnealing(
    #                 [model.split_points_param, model.matrices_param],
    #                 num_iterations=num_iterations,
    #                 initial_temp=sa_initial_temp,
    #                 num_restarts=0,
    #                 min_temp=sa_min_temp,
    #                 step_size=sa_step_size,
    #             ),
    #             max_depth=max_depth,
    #         ),
    #         OptimizerPart(
    #             Adam([model.split_points_param], lr=adam_lr),
    #             max_depth=max_depth,
    #             # weight=8,
    #         ),
    #     ]

    # parts = [OptimizerPart(Adam(model.parameters(), lr=adam_lr), max_depth=max_depth)]
    # parts = [OptimizerPart(Adam([model.matrices_param], lr=adam_lr), max_depth=max_depth)]
    parts = [
        OptimizerPart(
            SimulatedAnnealing(
                model.parameters(),
                num_iterations=num_iterations,
                initial_temp=sa_initial_temp,
                num_restarts=sa_num_restarts,
                min_temp=sa_min_temp,
                step_size=sa_step_size,
            ),
            max_depth=max_depth,
        )
    ]

    optimizer = Optimizer(parts, num_iterations=num_iterations)

    best_loss = float("inf")
    best_state = {}

    def closure():
        is_sa = isinstance(optimizer.current_optimizer, SimulatedAnnealing)
        grad_class = torch.no_grad if is_sa else torch.enable_grad
        with grad_class():
            optimizer.zero_grad()
            # predicted_y = model(
            #    target_x,
            #    optimizer.max_depth,
            #    # use_argmax=is_sa,
            #    use_argmax=True,
            # )
            # recon_loss = compute_loss(predicted_y, target_y)
            predicted_y1 = model(
                # target_x,
                optimizer.max_depth + 3,
                # use_argmax=is_sa,
                # use_argmax=True,
            )
            recon_loss1 = compute_loss(predicted_y1, target_y)

            # value_diversity_loss = -torch.std(model.values_param)
            # left_entropy = get_entropy_loss(model.left_matrix * 10.0)
            # right_entropy = get_entropy_loss(model.right_matrix * 10.0)
            # entropy_loss = (left_entropy + right_entropy) / 2.0

            # loss = recon_loss + value_diversity_loss * 0.01 + entropy_loss * 3.0  # 0.1
            # loss = recon_loss + recon_loss1
            loss = recon_loss1

            if not is_sa:
                loss.backward()

            return loss

    for i in range(num_iterations):
        plot_fn = lambda: model.plot_history(
            depths=[optimizer.max_depth, optimizer.max_depth + 3],
            target_x=target_x,
            target_y=target_y,
            viz_x=viz_x,
        )

        loss = optimizer.step(closure, plot_fn)

        model.track_iteration(
            iteration=i,
            recon_loss=loss,
        )

        if plot_every_n > 0 and (i % plot_every_n == 0 or i == num_iterations - 1):
            plot_fn()

        is_sa = isinstance(optimizer.current_optimizer, SimulatedAnnealing)

        if show_progress and (i % plot_every_n == 0 or i == num_iterations - 1):
            print(
                f"Iteration {i}/{num_iterations}, Loss: {loss.item():.6f}, {'sa' if is_sa else 'adam'}, Max depth: {optimizer.max_depth}"
            )

        if is_sa and loss.item() < best_loss:
            best_loss = loss.item()
            print("best loss!")
            plot_fn()

        # Save best state
        if optimizer.is_last_part and loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                "split_points_param": model.split_points_param.detach().clone(),
                "values_param": model.values_param.detach().clone(),
                "matrices_param": model.matrices_param.detach().clone(),
            }

    model.split_points_param.data.copy_(best_state["split_points_param"])
    model.values_param.data.copy_(best_state["values_param"])
    model.matrices_param.data.copy_(best_state["matrices_param"])

    return model


def compute_loss(predicted_y, target_y):
    return 10 * (1 - (torch.sum(predicted_y == target_y) / len(predicted_y)))
    # return nn.CrossEntropyLoss()(predicted_y, target_y)
    # return nn.functional.mse_loss(predicted_y, target_y)
    # threshold_loss = SmoothThresholdLoss()
    # mse_loss = nn.MSELoss()
    # return threshold_loss(predicted_y, target_y)
    # return threshold_loss(predicted_y, target_y) + mse_loss(predicted_y, target_y)
    # return mse_loss(predicted_y, target_y)
    # return nn.HuberLoss()(predicted_y, target_y)


class SmoothThresholdLoss(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        small_loss_factor: float = 0.01,
        large_loss_base: float = 10.0,
    ):
        """
        Smooth threshold loss with sigmoid transition

        Args:
            threshold: Error threshold for transition
            sharpness: How sharp the transition is (higher = more abrupt)
            small_loss_factor: Scaling factor for errors within threshold
            large_loss_base: Base penalty for errors beyond threshold
        """
        super().__init__()
        self.threshold = threshold
        self.sharpness = sharpness
        self.small_loss_factor = small_loss_factor
        self.large_loss_base = large_loss_base

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        errors = torch.abs(y_true - y_pred)

        # Sigmoid transition at the threshold
        sigmoid_factor = 1 / (
            1 + torch.exp(-self.sharpness * (errors - self.threshold))
        )

        # Small loss within threshold, large loss outside
        small_loss = self.small_loss_factor * errors.pow(2)
        large_loss = self.large_loss_base + errors.pow(2)

        # Combine using sigmoid weighting
        loss = small_loss * (1 - sigmoid_factor) + large_loss * sigmoid_factor

        return loss.mean()


# Run the demo if executed as main script
if __name__ == "__main__":
    optimized_func, history, fig = demo_optimization()
    plt.show()
