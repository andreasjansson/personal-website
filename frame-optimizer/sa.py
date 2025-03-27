import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from typing import Callable, Iterable


class PerturbableParameter(nn.Parameter):
    def __new__(cls, data, requires_grad=True, name=None, min=0.1, max=0.9):
        return super(PerturbableParameter, cls).__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad=True, name=None, min=0.1, max=0.9):
        super(PerturbableParameter, self).__init__()
        self.previous_value = None
        self.param_name = name
        self.min = min
        self.max = max

    def perturb(self, power: float) -> None:
        # Store current value before perturbation
        self.previous_value = self.detach().clone()
        # Apply standard Gaussian perturbation
        with torch.no_grad():
            noise = torch.randn_like(self) * power
            self.add_(noise)
            self.clamp_(min=self.min, max=self.max)

    def revert_perturbation(self) -> None:
        if self.previous_value is not None:
            with torch.no_grad():
                self.copy_(self.previous_value)
            self.previous_value = None


class CategoricalMatrixParameter(PerturbableParameter):
    def __init__(self, data, active=1.0, inactive=0.0, requires_grad=True):
        super().__init__(data, requires_grad)
        self.active = active
        self.inactive = inactive
        # self.allow_diagonal = allow_diagonal
        self.allow_diagonal = False
        # self.allow_diagonal = True

        # Ensure diagonals are inactive if not allowed
        if not self.allow_diagonal:
            with torch.no_grad():
                rows, cols = self.shape
                for i in range(min(rows, cols)):
                    self[i, i] = self.inactive

    def perturb(self, power: float) -> None:
        # Store current value before perturbation
        self.previous_value = self.detach().clone()

        with torch.no_grad():
            # Get the current matrix shape
            rows, cols = self.shape

            # For each row, randomly select which element to change
            for row in range(rows):
                # Random probability for this perturbation
                if torch.rand(1).item() < power:  # Scale probability with power
                    # Create valid positions (exclude diagonal if needed)
                    valid_positions = list(range(cols))
                    if not self.allow_diagonal and row < cols:
                        valid_positions.remove(row)

                    # Skip if no valid positions (shouldn't happen with reasonable matrices)
                    if not valid_positions:
                        continue

                    # Pick a random new position to make active
                    new_pos = valid_positions[
                        torch.randint(0, len(valid_positions), (1,)).item()
                    ]

                    # Reset row to inactive values
                    self[row, :] = self.inactive
                    self[row, new_pos] = self.active


class ValueMatricesParameter(PerturbableParameter):
    def __new__(cls, data, num_values: int, requires_grad=True, name=None):
        return super(ValueMatricesParameter, cls).__new__(cls, data, requires_grad)

    def __init__(self, data, num_values: int, requires_grad=True, name=None):
        """
        Parameter for managing paired left/right matrix values with specific constraints.

        Args:
            data: Tensor of shape [2, rows, cols] containing [left_matrix, right_matrix]
            requires_grad: Whether parameter requires gradients
            name: Optional name for the parameter
        """
        super().__init__(data, requires_grad, name)
        self.num_values = num_values
        self.num_values_with_dupes = data.shape[1]
        assert self.num_values_with_dupes == data.shape[2]
        assert data.shape[0] == 2
        assert data.shape[1] % self.num_values == 0
        self.num_dupes = data.shape[1] / self.num_values
        self.allow_diagonal = False

    def __getitem__(self, idx):
        return self.data[idx]

    def perturb(self, power: float) -> None:
        # Store current value before perturbation
        self.previous_value = self.detach().clone()

        with torch.no_grad():
            for row in range(self.num_values_with_dupes):
                # Random probability for this perturbation
                if torch.rand(1).item() < power:  # Scale probability with power
                    self.perturb_row(row)

    def perturb_row(self, row: int) -> None:
        # TODO: make self.same_class_direction a perturbale parameter
        # on Fractal2DNonDiff, and make valid positions be positions
        # where the class is the same (instead of the current modulo
        # logic). guarantee that at least two values exist with the
        # same class.

        # Decide randomly which matrix to apply the modulo constraint to
        modulo_constrained_matrix = torch.randint(0, 2, (1,)).item()  # 0 or 1

        # Get valid positions for both matrices (exclude diagonal if needed)
        valid_positions = list(range(self.num_values_with_dupes))
        if not self.allow_diagonal:
            valid_positions.remove(row)

        assert len(valid_positions) > 0

        # For the modulo-constrained matrix:
        valid_modulo_positions = [
            pos
            for pos in valid_positions
            if (pos % self.num_values) == (row % self.num_values) and pos != row
        ]

        assert len(valid_modulo_positions) > 0

        # Pick a random position
        new_pos_modulo = valid_modulo_positions[
            torch.randint(0, len(valid_modulo_positions), (1,)).item()
        ]

        # For the other matrix, just pick any valid position (excluding row)
        valid_other_positions = [
            pos for pos in valid_positions if pos != row and pos != new_pos_modulo
        ]
        assert len(valid_other_positions) > 0

        new_pos_other = valid_other_positions[
            torch.randint(0, len(valid_other_positions), (1,)).item()
        ]

        # Reset rows to zeros (inactive)
        self[0][row, :] = 0.0
        self[1][row, :] = 0.0

        # Set the active element in each matrix
        if modulo_constrained_matrix == 0:
            self[0][row, new_pos_modulo] = 1.0
            self[1][row, new_pos_other] = 1.0
        else:
            self[0][row, new_pos_other] = 1.0
            self[1][row, new_pos_modulo] = 1.0


class SimulatedAnnealing(Optimizer):
    def __init__(
        self,
        params: Iterable[PerturbableParameter],
        num_iterations: int,
        initial_temp: float = 1.0,
        num_restarts: int = 3,
        min_temp: float = 0.1,
        step_size: float = 0.1,
    ):
        # TODO:
        iterations_per_restart = num_iterations // (num_restarts + 1)
        cooling_rate = np.power(min_temp / initial_temp, 1.0 / iterations_per_restart)
        defaults = dict(
            initial_temp=initial_temp,
            temp=initial_temp,
            cooling_rate=cooling_rate,
            step_size=step_size,
            min_temp=min_temp,
        )
        super().__init__(params, defaults)
        # Track best parameters and loss
        self.best_loss = float("inf")
        self.best_params = {}

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.FloatTensor] | None = None, plot_fn=None
    ) -> torch.FloatTensor | None:
        """
        Performs a single optimization step using simulated annealing.
        """
        assert closure is not None

        with torch.no_grad():
            loss = closure()

        # Update best parameters if the current loss is better
        current_loss = loss.item()
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_params = {
                id(p): p.detach().clone()
                for group in self.param_groups
                for p in group["params"]
            }

        for group in self.param_groups:
            temp = group["temp"]
            step_size = group["step_size"]

            if group["temp"] <= group["min_temp"]:
                print("resetting")
                plot_fn()
                group["temp"] = group["initial_temp"]
                self.reset_to_best_params()

            # print(f"{temp=}, {group['min_temp']}")  # TODO(andreas): remove debug

            for p in group["params"]:
                # Store current loss
                state = self.state[p]
                if "prev_loss" not in state:
                    state["prev_loss"] = (
                        loss.item() if loss is not None else float("inf")
                    )

                # Apply perturbation
                # p.perturb(step_size * temp)
                p.perturb(step_size)

            # Evaluate the new configuration
            with torch.enable_grad():
                new_loss = closure()

            # Metropolis acceptance criterion
            delta_e = new_loss.item() - state["prev_loss"]

            # print(f"{delta_e=}, {np.exp(-delta_e / temp)=}")  # TODO(andreas): remove debug

            # If energy increased, accept with probability based on temperature
            if delta_e >= 0 and np.random.random() > np.exp(-delta_e / temp):
                # print(f"reject!")  # TODO(andreas): remove debug
                # Reject the move, restore previous parameters
                for p in group["params"]:
                    p.revert_perturbation()
            else:
                # Accept the move
                state["prev_loss"] = new_loss.item()
                loss = new_loss

            # Cool down the temperature
            group["temp"] = max(
                group["temp"] * group["cooling_rate"], group["min_temp"]
            )

        return loss

    def reset_to_best_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_id = id(p)
                p.data.copy_(self.best_params[param_id])
