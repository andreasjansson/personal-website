import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

class NodeNet(nn.Module):
    """
    Neural network for each node that determines the left and right child distribution
    """
    def __init__(self, num_nodes: int, hidden_size: int = 64, device: str = "cuda"):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        # Create a small neural network for each node
        self.left_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_nodes)
        ).to(device)

        self.right_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_nodes)
        ).to(device)

    def forward(self):
        """
        Returns distribution over child nodes
        """
        # Each node has a fixed behavior
        dummy_input = torch.ones(1, 1, device=self.device, dtype=torch.float32)

        # Get distributions over left and right children
        left_logits = self.left_net(dummy_input)
        right_logits = self.right_net(dummy_input)

        # Apply softmax to ensure valid probability distributions
        left_probs = F.softmax(left_logits, dim=-1)
        right_probs = F.softmax(right_logits, dim=-1)

        return left_probs.squeeze(), right_probs.squeeze()


class FractalNet(nn.Module):
    """
    Network that recursively splits frames to create a fractal-like image
    """
    def __init__(self, depth: int = 5, num_nodes: int = 20, num_colors: int = 8, device: str = "cuda"):
        super().__init__()
        self.depth = depth
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.device = device

        # Create node networks
        self.node_nets = nn.ModuleList([
            NodeNet(num_nodes, device=device) for _ in range(num_nodes)
        ])

        # Initialize color parameters for each node
        self.node_colors = nn.Parameter(torch.rand([num_nodes, num_colors], device=device, dtype=torch.float32))

        # Calculate dimensions for horizontal and vertical splits
        self.h_splits = (depth + 1) // 2  # Ceiling division for horizontal splits
        self.v_splits = depth // 2        # Floor division for vertical splits

        # Calculate final width and height
        self.width = 2 ** self.h_splits
        self.height = 2 ** self.v_splits

    def forward(self) -> torch.Tensor:
        """
        Forward pass that builds the fractal image
        Returns a tensor of shape [height, width, num_colors]
        """
        # Process the tree structure and generate the final image
        leaf_nodes = self.process_tree()

        # Render the image from leaf nodes
        final_image = self.render_image(leaf_nodes)

        return final_image

    def process_tree(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Process the fractal tree structure
        Returns a dictionary mapping (x,y) coordinates to node activation tensors
        """
        # Initialize with the root node
        current_level = {(0, 0): torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)}
        current_level[(0, 0)][0] = 1.0  # Node 0 is active at the root

        # Recursively split frames
        for d in range(self.depth):
            next_level = {}

            is_horizontal = (d % 2 == 0)  # Alternate between horizontal and vertical splits

            for (x, y), node_activations in current_level.items():
                # Skip if no significant activations
                if torch.max(node_activations) < 1e-6:
                    continue

                # Initialize child activations
                left_activations = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)
                right_activations = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

                # Process each active node
                for node_idx in range(self.num_nodes):
                    # Skip if node isn't significantly active
                    if node_activations[node_idx] < 1e-6:
                        continue

                    # Get weight of this node in the cell
                    weight = node_activations[node_idx]

                    # Get left and right child distributions
                    left_probs, right_probs = self.node_nets[node_idx]()

                    # Add weighted probabilities to child activations
                    left_activations += left_probs * weight
                    right_activations += right_probs * weight

                # Compute child cell coordinates based on split direction
                if is_horizontal:  # Horizontal split
                    next_level[(x*2, y)] = left_activations
                    next_level[(x*2+1, y)] = right_activations
                else:  # Vertical split
                    next_level[(x, y*2)] = left_activations
                    next_level[(x, y*2+1)] = right_activations

            current_level = next_level

        return current_level

    def render_image(self, leaf_nodes: Dict[Tuple[int, int], torch.Tensor]) -> torch.Tensor:
        """
        Render the final image from the node activations at leaf nodes
        """
        # Create the final image with proper dimensions
        image = torch.zeros(self.height, self.width, self.num_colors, device=self.device, dtype=torch.float32)

        # Fill the image with colors based on node activations
        for (x, y), node_activations in leaf_nodes.items():
            # Skip if out of bounds
            if x >= self.width or y >= self.height:
                continue

            # Get color for this cell by weighting the node colors by their activations
            cell_color = torch.matmul(node_activations.unsqueeze(0), self.node_colors).squeeze(0)

            # Assign color to the cell
            image[y, x] = cell_color

        return image


def train_fractal_net(target_image: np.ndarray, num_colors: int = 3, depth: int = 5, num_nodes: int = 20,
                     lr: float = 0.001, epochs: int = 1000, device: str = "cuda", viz_every: int = 50):
    """
    Train a FractalNet to approximate a target image
    """
    # Convert target image to float32
    target_image = target_image.astype(np.float32)

    # Initialize model
    model = FractalNet(depth=depth, num_nodes=num_nodes, num_colors=num_colors, device=device)
    model.to(device)

    # Calculate image dimensions based on depth
    h_splits = (depth + 1) // 2  # Ceiling division for horizontal splits
    v_splits = depth // 2        # Floor division for vertical splits

    width = 2 ** h_splits
    height = 2 ** v_splits

    # Target image should be of shape [height, width, num_colors]
    if target_image.shape != (height, width, num_colors):
        raise ValueError(f"Target image should have shape {(height, width, num_colors)}, "
                         f"got {target_image.shape}")

    target = torch.tensor(target_image, device=device, dtype=torch.float32)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track loss history
    loss_history = []

    # Training loop
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Forward pass
        output = model()

        # Compute loss
        loss = F.mse_loss(output, target)

        # Backprop
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        loss_history.append(loss.item())

        # Visualize progress
        if epoch % viz_every == 0 or epoch == epochs - 1:
            visualize_progress(target, output, loss_history, epoch)

    return model, loss_history


def visualize_progress(target: torch.Tensor, output: torch.Tensor, loss_history: list, epoch: int):
    """Visualize training progress"""
    with torch.no_grad():
        current_output = output.detach().cpu().numpy()
        target_np = target.cpu().numpy()

    plt.figure(figsize=(7, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(np.argmax(target_np, axis=2))
    plt.title("Target Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.argmax(current_output, axis=2))
    plt.title(f"Model Output (Epoch {epoch})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()


def create_target_image(depth: int, num_colors: int) -> np.ndarray:
    """
    Create a target image with the proper dimensions for the given depth
    """
    # Calculate image dimensions based on depth
    h_splits = (depth + 1) // 2  # Ceiling division for horizontal splits
    v_splits = depth // 2        # Floor division for vertical splits

    width = 2 ** h_splits
    height = 2 ** v_splits

    # Create a target image with different colored regions
    target = np.zeros((height, width, num_colors), dtype=np.float32)

    # Fill with different patterns
    for i in range(num_colors):
        mask_x = ((i * width // num_colors <= np.arange(width)) &
                (np.arange(width) < (i + 1) * width // num_colors))
        for y in range(height):
            if y % 2 == 0:
                target[y, mask_x, i] = 1.0
            else:
                target[y, mask_x, (i+1) % num_colors] = 1.0

    return target


# Example usage
if __name__ == "__main__":
    # Set parameters
    depth = 5  # Try with odd depth to test the fix
    num_colors = 8
    num_nodes = 50

    # Create appropriate target image
    target = create_target_image(depth, num_colors)

    # Print dimensions to verify
    h_splits = (depth + 1) // 2
    v_splits = depth // 2
    expected_width = 2 ** h_splits
    expected_height = 2 ** v_splits
    print(f"Depth: {depth} -> Expected image shape: ({expected_height}, {expected_width}, {num_colors})")
    print(f"Target image shape: {target.shape}")

    # Train the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, loss_history = train_fractal_net(
        target_image=target,
        num_colors=num_colors,
        depth=depth,
        num_nodes=num_nodes,
        lr=0.001,
        epochs=500,  # Reduced for faster demonstration
        device=device,
        viz_every=50
    )

    # Display final result
    with torch.no_grad():
        final_output = model().detach().cpu().numpy()

    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(np.argmax(target, axis=2))
    plt.title("Target Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.argmax(final_output, axis=2))
    plt.title("Final Output")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("final_result.png")
    plt.show()
