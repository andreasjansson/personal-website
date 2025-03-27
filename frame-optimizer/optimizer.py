import torch
from typing import Callable

from sa import SimulatedAnnealing


class OptimizerPart:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer | Callable[[int], torch.optim.Optimizer],
        max_depth: int = 1,
        weight: float = 1.0,
    ):
        """
        A wrapper for an optimizer that operates at a specific max_depth.

        Args:
            optimizer: A PyTorch optimizer instance
            max_depth: The max_depth to use during this optimization part
            weight: Relative weight for allocating iterations (default=1.0)
        """
        self.optimizer = optimizer
        self.max_depth = max_depth
        self.weight = weight

    def finalize_optimizer(self, num_iterations: int):
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            self.optimizer = self.optimizer(num_iterations)


class Optimizer:
    def __init__(self, optimizer_parts: list[OptimizerPart], num_iterations: int):
        """
        A composite optimizer that cycles through different optimizer parts.

        Args:
            optimizer_parts: List of OptimizerPart instances
            num_iterations: Total number of iterations to distribute among parts
        """
        self.parts = optimizer_parts
        self.num_iterations = num_iterations
        self.total_parts = len(optimizer_parts)

        # Calculate weighted distribution of iterations
        total_weight = sum(part.weight for part in optimizer_parts)

        # Allocate iterations based on weights
        self.iterations_per_part = []
        remaining_iterations = num_iterations

        for i, part in enumerate(optimizer_parts[:-1]):  # All but the last part
            # Calculate weighted share and round to integer
            part_iterations = int(round((part.weight / total_weight) * num_iterations))
            # Ensure at least 1 iteration per part
            part_iterations = max(1, part_iterations)
            # Ensure we don't exceed total iterations
            part_iterations = min(
                part_iterations, remaining_iterations - (self.total_parts - i - 1)
            )

            self.iterations_per_part.append(part_iterations)
            remaining_iterations -= part_iterations

        # Last part gets any remaining iterations
        self.iterations_per_part.append(max(1, remaining_iterations))

        # Initialize state
        self.current_part_idx = 0
        self.current_part_iter = 0
        self.total_iter = 0

        self.did_switch_optimizer = False

        # Print iteration distribution
        for i, (part, iters) in enumerate(
            zip(optimizer_parts, self.iterations_per_part)
        ):
            part.finalize_optimizer(iters)

            part_name = type(part.optimizer).__name__
            print(
                f"Part {i}: {part_name} (max_depth={part.max_depth}, weight={part.weight:.2f}) - {iters} iterations"
            )

    def zero_grad(self):
        """Zero gradients for the current optimizer part"""
        self.current_optimizer.zero_grad()

    def step(self, closure, plot_fn=None):
        """
        Step the current optimizer part and handle transitions
        between optimization parts.
        """
        # Perform the optimization step with current part
        is_sa = isinstance(self.current_optimizer, SimulatedAnnealing)
        if is_sa:
            result = self.current_optimizer.step(closure, plot_fn)
        else:
            result = self.current_optimizer.step(closure)

        # Increment counters
        self.current_part_iter += 1
        self.total_iter += 1

        # Check if we should move to the next part
        if (
            self.current_part_iter >= self.iterations_per_part[self.current_part_idx]
            and self.total_iter < self.num_iterations - 1
        ):
            # If current optimizer is SA and next one is Adam, transfer best parameters
            current_part = self.parts[self.current_part_idx]
            current_optimizer = current_part.optimizer

            next_idx = (self.current_part_idx + 1) % self.total_parts
            next_part = self.parts[next_idx]

            # Transfer the best parameters from SA to Adam
            if hasattr(current_optimizer, "best_params"):
                current_optimizer.reset_to_best_params()
                print(
                    f"Transferred best parameters from SA to Adam (best loss: {current_optimizer.best_loss:.6f})"
                )

            print("switching optimizer...")
            if plot_fn:
                plot_fn()

            # Switch to next part
            self.current_part_idx = next_idx
            self.current_part_iter = 0

            # Print transition information
            part_name = type(self.current_optimizer).__name__
            print(
                f"Switching to optimization part {self.current_part_idx}: {part_name} with max_depth={self.max_depth}"
            )
            self.did_switch_optimizer = True
        else:
            self.did_switch_optimizer = False

        return result

    @property
    def param_groups(self):
        """Return parameter groups from the current optimizer part"""
        return self.parts[self.current_part_idx].optimizer.param_groups

    @property
    def current_optimizer(self):
        return self.parts[self.current_part_idx].optimizer

    @property
    def max_depth(self) -> int:
        return self.parts[self.current_part_idx].max_depth

    @property
    def is_last_part(self) -> bool:
        return self.current_part_idx == len(self.parts) - 1
