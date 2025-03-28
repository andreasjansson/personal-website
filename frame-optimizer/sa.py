import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from typing import Callable, Iterable


class PerturbableParameter(nn.Parameter):
    def __new__(
        cls,
        data,
        requires_grad=True,
        name=None,
        min: int | float = 0.1,
        max: int | float = 0.9,
    ):
        return super(PerturbableParameter, cls).__new__(cls, data, requires_grad)

    def __init__(
        self,
        data,
        requires_grad=True,
        name=None,
        min: int | float = 0.1,
        max: int | float = 0.9,
    ):
        super(PerturbableParameter, self).__init__()
        self.previous_value = None
        self.param_name = name
        self.min = min
        self.max = max
        self.is_integer = data.dtype in (
            torch.int,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
        )

    @property
    def is_categorical(self):
        return self.is_integer

    def get_num_categories(self) -> int:
        """Return the number of possible values for categorical parameters"""
        if not self.is_categorical:
            return 0

        # For integer parameters, return the number of possible values
        return int(self.max - self.min + 1)

    def perturb(self, power: float) -> None:
        # Store current value before perturbation
        self.previous_value = self.detach().clone()

        # Apply perturbation based on type
        with torch.no_grad():
            if self.is_integer:
                # For integer parameters, use discrete perturbations
                # Calculate how many elements to change based on power
                num_elements = max(1, int(power * self.numel()))
                # Select random indices to change
                indices = torch.randperm(self.numel())[:num_elements]
                # Reshape tensor to 1D for easy modification
                flat_tensor = self.view(-1)
                # Generate random integers within valid range for the selected indices
                new_values = torch.randint(
                    int(self.min),
                    int(self.max) + 1,
                    (num_elements,),
                    device=self.device,
                )
                # Update selected elements
                flat_tensor[indices] = new_values
            else:
                # For floating point parameters, use Gaussian noise
                noise = torch.randn_like(self) * power
                self.add_(noise)
                self.clamp_(min=self.min, max=self.max)

    def revert_perturbation(self) -> None:
        if self.previous_value is not None:
            with torch.no_grad():
                self.copy_(self.previous_value)
            self.previous_value = None


class ClassesParameter(PerturbableParameter):
    def __new__(
        cls,
        data,
        num_classes: int,
        requires_grad=False,
        name=None,
    ):
        return super(ClassesParameter, cls).__new__(cls, data, requires_grad)

    def __init__(
        self,
        data,
        num_classes: int,
        requires_grad=False,
        name=None,
    ):
        """
        A parameter for class assignments that ensures each class has at least two values.

        Args:
            data: Tensor of class assignments (integers)
            num_classes: Number of distinct classes (0 to num_classes-1)
            requires_grad: Whether parameter requires gradients (usually False for discrete values)
            name: Optional name for the parameter
        """
        super().__init__(data, requires_grad, name, min=0, max=num_classes - 1)
        self.num_classes = num_classes

        # Verify initial state is valid
        #self._verify_class_distribution()

    @property
    def is_categorical(self):
        return True

    def get_num_categories(self) -> int:
        return self.num_classes

    def _verify_class_distribution(self) -> None:
        """Verify that each class has at least two representatives"""
        for class_idx in range(self.num_classes):
            count = torch.sum(self == class_idx).item()
            if count < 2:
                raise ValueError(
                    f"Class {class_idx} has only {count} values. Each class must have at least 2 values."
                )

    def perturb(self, power: float) -> None:
        """
        Perturb the class assignments while ensuring at least two values per class.
        """
        # Store current value before perturbation
        self.previous_value = self.detach().clone()

        with torch.no_grad():
            # Calculate how many elements to potentially change based on power
            num_elements = max(1, int(power * self.numel()))

            # Select random indices to potentially change
            indices = torch.randperm(self.numel())[:num_elements]

            for idx in indices:
                # Get current class for this index
                current_class = self[idx].item()

                # Count how many instances of this class we have
                class_count = torch.sum(self == current_class).item()

                # Only change if we have more than 2 of this class
                if class_count > 2:
                    # Choose a random new class
                    possible_classes = list(range(self.num_classes))
                    possible_classes.remove(current_class)
                    new_class = possible_classes[
                        torch.randint(0, len(possible_classes), (1,)).item()
                    ]

                    # Apply the change
                    self[idx] = new_class

    def revert_perturbation(self) -> None:
        """Revert to previous state if perturbation is rejected"""
        if self.previous_value is not None:
            with torch.no_grad():
                self.copy_(self.previous_value)
            self.previous_value = None


class ChildIndicesParameter(PerturbableParameter):
    def __new__(
        cls,
        data,
        num_classes: int,
        same_class_direction: torch.Tensor | nn.Parameter,
        classes: torch.Tensor | nn.Parameter,
        requires_grad=False,  # Typically false since these are discrete indices
        name=None,
    ):
        return super(ChildIndicesParameter, cls).__new__(cls, data, requires_grad)

    def __init__(
        self,
        data,
        num_classes: int,
        same_class_direction: torch.Tensor | nn.Parameter,
        classes: torch.Tensor | nn.Parameter,
        requires_grad=False,
        name=None,
    ):
        """
        Parameter for managing child indices [left_children, right_children] with specific class constraints.

        Args:
            data: Tensor of shape [2, num_values] containing [left_children, right_children] indices
            num_classes: Number of distinct classes
            same_class_direction: Which direction maintains the same class (0=left, 1=right)
            classes: Class assignment for each value index
            requires_grad: Whether parameter requires gradients (usually False for discrete indices)
            name: Optional name for the parameter
        """
        super().__init__(data, requires_grad, name)
        self.num_classes = num_classes
        self.same_class_direction = same_class_direction
        self.classes = classes
        self.num_values = self.classes.shape[0]
        assert data.shape[0] == 2  # [left_children, right_children]
        assert data.shape[1] == self.num_values

    @property
    def is_categorical(self):
        return True

    def get_num_categories(self) -> int:
        return self.num_values

    def perturb(self, power: float) -> None:
        # Store current value before perturbation
        self.previous_value = self.detach().clone()

        with torch.no_grad():
            for row in range(self.num_values):
                # Random probability for this perturbation
                if torch.rand(1).item() < power:  # Scale probability with power
                    self.perturb_row(row)

    def perturb_row(self, row: int) -> None:
        # Get valid positions (exclude self)
        valid_positions = list(range(self.num_values))
        valid_positions.remove(row)  # Don't allow self-reference

        assert len(valid_positions) > 0

        row_class = self.classes[row]
        valid_same_class_positions = [
            pos for pos in valid_positions if self.classes[pos] == row_class
        ]

        assert len(valid_same_class_positions) > 0, (
            f"No same-class positions found for row {row}, class {row_class}"
        )

        # Pick a random position with the same class
        new_pos_same_class = valid_same_class_positions[
            torch.randint(0, len(valid_same_class_positions), (1,)).item()
        ]

        valid_other_positions = [
            pos for pos in valid_positions if pos != new_pos_same_class
        ]
        assert len(valid_other_positions) > 0

        new_pos_other = valid_other_positions[
            torch.randint(0, len(valid_other_positions), (1,)).item()
        ]

        # Update child indices based on same_class_direction
        if self.same_class_direction[row] == 0:
            self[0, row] = new_pos_same_class
            self[1, row] = new_pos_other
        else:
            self[0, row] = new_pos_other
            self[1, row] = new_pos_same_class


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
