import torch


class FractalFunction:
    """Base class for all fractal function components"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the function at point x"""
        raise NotImplementedError("Subclasses must implement __call__")

    def __repr__(self) -> str:
        """Return the mathematical notation of the function"""
        raise NotImplementedError("Subclasses must implement __repr__")


class ConstantFunction(FractalFunction):
    """A constant function f(x) = c"""

    def __init__(self, value: float | torch.Tensor):
        self.value = value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # If value is a tensor, broadcast it to the shape of x
        if isinstance(self.value, torch.Tensor):
            # Use expansion/broadcasting instead of full_like
            return self.value.expand_as(x).to("cuda")
        else:
            return torch.full_like(x, self.value)

    def __repr__(self) -> str:
        return str(self.value)


class SigmoidFunction(FractalFunction):
    """A sigmoid function centered at a point"""

    def __init__(self, center: float, width: float):
        self.center = center
        self.width = width
        self.scale = 20.0 / width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - self.center) * self.scale)

    def __repr__(self) -> str:
        return f"σ((x - {self.center:.3f}) · {self.scale:.3f})"


class BlendedFunction(FractalFunction):
    """A smooth blend of two functions using a sigmoid transition"""

    def __init__(
        self,
        left_func: FractalFunction,
        right_func: FractalFunction,
        transition: SigmoidFunction | ConstantFunction,
        depth: int,
    ):
        self.left_func = left_func
        self.right_func = right_func
        self.transition = transition
        self.depth = depth

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = self.transition(x)
        return (1 - t) * self.left_func(x) + t * self.right_func(x)

    def __repr__(self) -> str:
        return f"(1 - {self.transition}) · ({self.left_func}) + ({self.transition}) · ({self.right_func})"


def generate_fractal_function(
    split_points: torch.Tensor,
    values: torch.Tensor,
    left_children_matrix: torch.Tensor,   # Shape [len(values), len(values)]
    right_children_matrix: torch.Tensor,  # Shape [len(values), len(values)]
    max_depth: int = 3,
    smoothing_width: float = 0.1,
    use_argmax: bool = False,  # Use argmax instead of softmax for inference
) -> FractalFunction:
    """
    Creates a fractal function through differentiable function composition.

    Args:
        split_points: List of split points (0-1) for each segment
        values: Base values for each segment
        left_children_matrix: Matrix of child logits for left branches [len(values), len(values)]
        right_children_matrix: Matrix of child logits for right branches [len(values), len(values)]
        max_depth: Maximum recursion depth
        smoothing_width: Width of the smoothing transition (0-1)
        use_argmax: If True, uses deterministic argmax path instead of probabilistic softmax

    Returns:
        A composable differentiable function object
    """
    if use_argmax:
        # Argmax path - use indices directly
        left_indices = torch.argmax(left_children_matrix, dim=1)
        right_indices = torch.argmax(right_children_matrix, dim=1)
        # Don't need probabilities since we're just following single best path
        left_children_probs = None
        right_children_probs = None
    else:
        # Softmax path - use probability distributions
        left_children_probs = torch.nn.functional.softmax(left_children_matrix * 10.0, dim=1)
        right_children_probs = torch.nn.functional.softmax(right_children_matrix * 10.0, dim=1)

    # Function cache to prevent exponential explosion
    function_cache = {}

    def build_function(
        depth: int,
        start: float,
        end: float,
        parent_idx: int,
    ) -> FractalFunction:
        """Recursively build the fractal function for range [start, end]"""
        # Use memoization to avoid exponential explosion
        cache_key = (depth, start, end, parent_idx)
        if cache_key in function_cache:
            return function_cache[cache_key]

        value = values[parent_idx]

        # Base case - maximum depth reached
        if depth == max_depth:
            result = ConstantFunction(value)
            function_cache[cache_key] = result
            return result

        # Calculate split position in absolute coordinates
        split = split_points[parent_idx % len(split_points)]
        mid = start + split * (end - start)

        # Create transition function with width that scales with segment size
        segment_size = end - start
        trans_width = smoothing_width * segment_size
        transition = SigmoidFunction(mid, trans_width)

        if use_argmax:
            # Simply use the best child for each branch
            left_child_idx = left_indices[parent_idx].item()
            right_child_idx = right_indices[parent_idx].item()

            left_func = build_function(depth + 1, start, mid, left_child_idx)
            right_func = build_function(depth + 1, mid, end, right_child_idx)
        else:
            # Get LEFT branch probabilities for this parent
            left_probs = left_children_probs[parent_idx]

            # Create weighted left branch (blend of all possible children)
            left_branches = []
            for i, prob in enumerate(left_probs):
                if prob > 1e-5:  # Skip negligible contributions
                    child_func = build_function(depth + 1, start, mid, i)
                    left_branches.append((prob, child_func))

            # Get RIGHT branch probabilities for this parent
            right_probs = right_children_probs[parent_idx]

            # Create weighted right branch (blend of all possible children)
            right_branches = []
            for i, prob in enumerate(right_probs):
                if prob > 1e-5:  # Skip negligible contributions
                    child_func = build_function(depth + 1, mid, end, i)
                    right_branches.append((prob, child_func))

            # Combine left branches with weighted blending
            if not left_branches:
                left_func = ConstantFunction(values[0])
            else:
                # Sort by weight descending to blend most significant first
                left_branches.sort(key=lambda x: x[0], reverse=True)
                left_func = left_branches[0][1]
                total_weight = left_branches[0][0]

                # Incrementally blend the functions with proper weighting
                for weight, func in left_branches[1:3]:
                    # Calculate relative weight for this blend
                    blend_weight = weight / (total_weight + weight)
                    weight_func = ConstantFunction(blend_weight)
                    left_func = BlendedFunction(left_func, func, weight_func, depth)
                    total_weight = total_weight + weight

            # Repeat for right branches
            if not right_branches:
                right_func = ConstantFunction(values[0])
            else:
                right_branches.sort(key=lambda x: x[0], reverse=True)
                right_func = right_branches[0][1]
                total_weight = right_branches[0][0]

                for weight, func in right_branches[1:3]:
                    blend_weight = weight / (total_weight + weight)
                    weight_func = ConstantFunction(blend_weight)
                    right_func = BlendedFunction(right_func, func, weight_func, depth)
                    total_weight = total_weight + weight

        # Create final blended function
        result = BlendedFunction(left_func, right_func, transition, depth)
        function_cache[cache_key] = result
        return result

    # Create the full function with domain restriction
    fractal_func = build_function(depth=0, start=0.0, end=1.0, parent_idx=0)

    # Create wrapper with domain restriction
    class DomainRestrictedFractal(FractalFunction):
        def __init__(self, func: FractalFunction):
            self.func = func
            # Store parameters to allow gradient flow
            self.left_children_matrix = left_children_matrix
            self.right_children_matrix = right_children_matrix
            self.split_points = split_points
            self.values = values
            self.use_argmax = use_argmax

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            x_clipped = x
            #x_clipped = torch.clamp(x, 0.0, 1.0)
            return self.func(x_clipped)

        def __repr__(self) -> str:
            return f"f(x) = {self.func}"

    return DomainRestrictedFractal(fractal_func)

# import torch
# split_points = [0.4, 0.6, 0.5]
# values = [1, 3, 5]

# max_depth = 4
# n = 1000

# left_children_matrix = torch.tensor([
#     [0.0, 10.0, 0.0],  # Parent 0 preferences
#     [0.0, 0.0, 10.0],  # Parent 1 preferences
#     [10.0, 0.0, 0.0]   # Parent 2 preferences
# ], requires_grad=True)
# right_children_matrix = torch.tensor([
#     [0.0, 0.0, 10.0],  # Parent 0 preferences
#     [10.0, 0.0, 0.0],  # Parent 1 preferences
#     [0.0, 10.0, 0.0]   # Parent 2 preferences
# ], requires_grad=True)

# # Create a fractal function
# func = generate_fractal_function(
#     left_children_matrix=left_children_matrix,
#     right_children_matrix=right_children_matrix,
#     split_points=split_points,
#     values=values,
#     max_depth=max_depth,
#     smoothing_width=0.15
# )
