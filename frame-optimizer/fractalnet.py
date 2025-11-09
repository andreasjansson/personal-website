import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sklearn.cluster import KMeans


import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sklearn.cluster import KMeans


class FastNodeNet(nn.Module):
    def __init__(self, num_nodes: int, hidden_size: int = 64, device: str = "cuda"):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        # More efficient implementation with direct parameters instead of networks
        self.left_logits = nn.Parameter(torch.randn(num_nodes, device=device))
        self.right_logits = nn.Parameter(torch.randn(num_nodes, device=device))

    def forward(self):
        left_probs = F.softmax(self.left_logits, dim=-1)
        right_probs = F.softmax(self.right_logits, dim=-1)
        return left_probs, right_probs


class FastFractalNet(nn.Module):
    def __init__(self, num_nodes: int = 20, num_colors: int = 8, device: str = "cuda"):
        super().__init__()
        assert num_nodes % 2 == 0

        self.num_nodes = num_nodes
        self.nodes_per_direction = num_nodes // 2
        self.num_colors = num_colors
        self.device = device

        self.h_node_nets = nn.ModuleList(
            [
                FastNodeNet(self.nodes_per_direction, device=device)
                for _ in range(self.nodes_per_direction)
            ]
        )
        self.v_node_nets = nn.ModuleList(
            [
                FastNodeNet(self.nodes_per_direction, device=device)
                for _ in range(self.nodes_per_direction)
            ]
        )

        # Initialize color logits
        self.h_node_colors = nn.Parameter(
            torch.randn([self.nodes_per_direction, num_colors], device=device)
        )
        self.v_node_colors = nn.Parameter(
            torch.randn([self.nodes_per_direction, num_colors], device=device)
        )

        # Cache transition matrices for faster forward passes
        self.h_transition_left = None
        self.h_transition_right = None
        self.v_transition_left = None
        self.v_transition_right = None

        self.build_transition_matrices()

    def build_transition_matrices(self):
        """Pre-compute transition matrices for faster forward passes"""
        # Horizontal transitions to vertical nodes
        h_left = torch.zeros(
            self.nodes_per_direction, self.nodes_per_direction, device=self.device
        )
        h_right = torch.zeros(
            self.nodes_per_direction, self.nodes_per_direction, device=self.device
        )

        # Vertical transitions to horizontal nodes
        v_left = torch.zeros(
            self.nodes_per_direction, self.nodes_per_direction, device=self.device
        )
        v_right = torch.zeros(
            self.nodes_per_direction, self.nodes_per_direction, device=self.device
        )

        # Compute all transitions at once
        for i in range(self.nodes_per_direction):
            h_left_probs, h_right_probs = self.h_node_nets[i]()
            v_left_probs, v_right_probs = self.v_node_nets[i]()

            h_left[i] = h_left_probs
            h_right[i] = h_right_probs
            v_left[i] = v_left_probs
            v_right[i] = v_right_probs

        self.h_transition_left = h_left
        self.h_transition_right = h_right
        self.v_transition_left = v_left
        self.v_transition_right = v_right

    def forward(self, depth: int) -> torch.Tensor:
        """
        Optimized forward pass using vectorized operations and cached transition matrices.
        """
        # Rebuild transition matrices to get updated values
        self.build_transition_matrices()

        # Initialize with root node
        activations = torch.zeros(1, self.nodes_per_direction, device=self.device)
        activations[0, 0] = 1.0  # Node 0 is active at the root

        # Tensor to keep track of positions of cells
        positions = torch.zeros(1, 2, device=self.device)  # [batch, (x,y)]

        # Process through all layers with a more efficient iteration
        for d in range(depth):
            is_horizontal = d % 2 == 0

            if is_horizontal:
                # Horizontal split (creates left and right cells)
                left_activations = torch.matmul(activations, self.h_transition_left)
                right_activations = torch.matmul(activations, self.h_transition_right)

                # Compute new positions
                current_positions = positions.clone()
                left_positions = current_positions.clone()
                right_positions = current_positions.clone()

                # Update x-coordinate for left and right children
                left_positions[:, 0] = 2 * current_positions[:, 0]
                right_positions[:, 0] = 2 * current_positions[:, 0] + 1

                # Combine into batch dimension
                new_activations = torch.cat(
                    [left_activations, right_activations], dim=0
                )
                new_positions = torch.cat([left_positions, right_positions], dim=0)
            else:
                # Vertical split (creates top and bottom cells)
                top_activations = torch.matmul(activations, self.v_transition_left)
                bottom_activations = torch.matmul(activations, self.v_transition_right)

                # Compute new positions
                current_positions = positions.clone()
                top_positions = current_positions.clone()
                bottom_positions = current_positions.clone()

                # Update y-coordinate for top and bottom children
                top_positions[:, 1] = 2 * current_positions[:, 1]
                bottom_positions[:, 1] = 2 * current_positions[:, 1] + 1

                # Combine
                new_activations = torch.cat(
                    [top_activations, bottom_activations], dim=0
                )
                new_positions = torch.cat([top_positions, bottom_positions], dim=0)

            # Update for next iteration
            activations = new_activations
            positions = new_positions

        # Get dimensions of final grid
        is_final_split_horizontal = (depth - 1) % 2 == 0
        node_colors = (
            self.h_node_colors if is_final_split_horizontal else self.v_node_colors
        )

        h_splits = (depth + 1) // 2
        v_splits = depth // 2
        width = 2**h_splits
        height = 2**v_splits

        # Create empty image with logits
        image_logits = torch.zeros(height, width, self.num_colors, device=self.device)

        # Convert positions to integer indices
        pos_indices = positions.long()

        # For each leaf cell, compute its color logits and place in the image
        for i in range(activations.shape[0]):
            x, y = pos_indices[i, 0].item(), pos_indices[i, 1].item()
            cell_logits = torch.matmul(
                activations[i].unsqueeze(0), node_colors
            ).squeeze(0)
            image_logits[y, x] = cell_logits

        return image_logits

    def to_slow(self) -> FractalNet:
        """
        Convert FastFractalNet to the original slower FractalNet implementation.
        Useful for HTML rendering or other operations that require the original model structure.
        """
        slow_model = FractalNet(num_nodes=self.num_nodes, num_colors=self.num_colors, device=self.device)

        # Transfer horizontal node parameters
        for node_idx in range(self.nodes_per_direction):
            # Copy left and right logits to the slow model's networks
            with torch.no_grad():
                # Get the probabilities from FastNodeNet
                h_left_probs, h_right_probs = self.h_node_nets[node_idx]()

                # Set the parameters in the slow model to match these probabilities
                # We need to reshape for the linear layer's expected input shape
                dummy_input = torch.ones(1, 1, device=self.device)

                # Adjust the final layer of each network to output the desired probabilities
                slow_h_left_out = slow_model.h_node_nets[node_idx].left_net[-1]
                slow_h_right_out = slow_model.h_node_nets[node_idx].right_net[-1]

                # Compute what the output should be (without softmax since it's applied later)
                h_left_logits = torch.log(h_left_probs + 1e-10)  # Add small epsilon to avoid log(0)
                h_right_logits = torch.log(h_right_probs + 1e-10)

                # Set the bias directly since we have a dummy input of 1
                slow_h_left_out.bias.data = h_left_logits
                slow_h_left_out.weight.data.fill_(0)  # Zero out weights

                slow_h_right_out.bias.data = h_right_logits
                slow_h_right_out.weight.data.fill_(0)  # Zero out weights

        # Transfer vertical node parameters
        for node_idx in range(self.nodes_per_direction):
            with torch.no_grad():
                # Get the probabilities from FastNodeNet
                v_left_probs, v_right_probs = self.v_node_nets[node_idx]()

                # Set the parameters in the slow model
                dummy_input = torch.ones(1, 1, device=self.device)

                slow_v_left_out = slow_model.v_node_nets[node_idx].left_net[-1]
                slow_v_right_out = slow_model.v_node_nets[node_idx].right_net[-1]

                v_left_logits = torch.log(v_left_probs + 1e-10)
                v_right_logits = torch.log(v_right_probs + 1e-10)

                slow_v_left_out.bias.data = v_left_logits
                slow_v_left_out.weight.data.fill_(0)

                slow_v_right_out.bias.data = v_right_logits
                slow_v_right_out.weight.data.fill_(0)

        # Transfer color parameters
        slow_model.h_node_colors.data = self.h_node_colors.data.clone()
        slow_model.v_node_colors.data = self.v_node_colors.data.clone()

        return slow_model

def fast_train_fractal_net(
    target: np.ndarray,
    color_map: list[tuple],
    num_colors: int = 8,
    depth: int = 5,
    num_nodes: int = 20,
    lr: float = 0.001,
    epochs: int = 1000,
    device: str = "cuda",
    viz_every: int = 50,
):
    """
    Train a fractal network to approximate a target image.
    Optimized for performance with the FastFractalNet.
    """
    target = target.astype(np.float32)

    model = FastFractalNet(num_nodes=num_nodes, num_colors=num_colors, device=device)
    model.to(device)

    h_splits = (depth + 1) // 2
    v_splits = depth // 2

    width = 2**h_splits
    height = 2**v_splits

    if target.shape != (height, width, num_colors):
        raise ValueError(
            f"Target image should have shape {(height, width, num_colors)}, "
            f"got {target.shape}"
        )

    target_tensor = torch.tensor(target, device=device, dtype=torch.float32)
    target_indices = torch.argmax(target_tensor, dim=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=50,
        factor=0.5,
    )

    loss_history = []

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Get color logits
        logits = model(depth)

        # Reshape for cross entropy loss
        logits_flat = logits.reshape(-1, num_colors)
        targets_flat = target_indices.reshape(-1)

        # Calculate cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())

        if epoch % viz_every == 0 or epoch == epochs - 1:
            visualize_progress(target_tensor, logits, loss_history, epoch, color_map)

    return model, loss_history


class NodeNet(nn.Module):
    def __init__(self, num_nodes: int, hidden_size: int = 64, device: str = "cuda"):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        self.left_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_nodes),
        ).to(device)

        self.right_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_nodes),
        ).to(device)

    def forward(self):
        dummy_input = torch.ones(1, 1, device=self.device, dtype=torch.float32)

        left_logits = self.left_net(dummy_input)
        right_logits = self.right_net(dummy_input)

        left_probs = F.softmax(left_logits, dim=-1)
        right_probs = F.softmax(right_logits, dim=-1)

        return left_probs.squeeze(), right_probs.squeeze()


class FractalNet(nn.Module):
    def __init__(self, num_nodes: int = 20, num_colors: int = 8, device: str = "cuda"):
        super().__init__()
        assert num_nodes % 2 == 0

        self.num_nodes = num_nodes
        self.nodes_per_direction = num_nodes // 2
        self.num_colors = num_colors
        self.device = device

        self.h_node_nets = nn.ModuleList(
            [
                NodeNet(self.nodes_per_direction, device=device)
                for _ in range(self.nodes_per_direction)
            ]
        )
        self.v_node_nets = nn.ModuleList(
            [
                NodeNet(self.nodes_per_direction, device=device)
                for _ in range(self.nodes_per_direction)
            ]
        )

        # Initialize color logits instead of probabilities
        self.h_node_colors = nn.Parameter(
            torch.randn([self.nodes_per_direction, num_colors], device=device)
        )
        self.v_node_colors = nn.Parameter(
            torch.randn([self.nodes_per_direction, num_colors], device=device)
        )

    def forward(self, depth: int) -> torch.Tensor:
        leaf_nodes = self.process_tree(depth)
        is_horizontal = (depth - 1) % 2 == 0
        final_image = self.render_image(leaf_nodes, is_horizontal)
        return final_image

    def process_tree(self, depth: int) -> dict[tuple[int, int], torch.Tensor]:
        # Initialize with root node
        current_level = {
            (0, 0): torch.zeros(self.nodes_per_direction, device=self.device)
        }
        current_level[(0, 0)][0] = 1.0  # Node 0 is active at the root

        for d in range(depth):
            next_level = {}
            is_horizontal = d % 2 == 0
            node_nets = self.h_node_nets if is_horizontal else self.v_node_nets

            for (x, y), node_activations in current_level.items():
                # Skip if no significant activations
                if torch.max(node_activations) < 1e-6:
                    continue

                left_activations = torch.zeros(
                    self.nodes_per_direction, device=self.device
                )
                right_activations = torch.zeros(
                    self.nodes_per_direction, device=self.device
                )

                for node_idx in range(self.nodes_per_direction):
                    weight = node_activations[node_idx]
                    if weight > 1e-6:  # Only process active nodes
                        left_probs, right_probs = node_nets[node_idx]()
                        left_activations += left_probs * weight
                        right_activations += right_probs * weight

                if is_horizontal:
                    next_level[(x * 2, y)] = left_activations
                    next_level[(x * 2 + 1, y)] = right_activations
                else:
                    next_level[(x, y * 2)] = left_activations
                    next_level[(x, y * 2 + 1)] = right_activations

            current_level = next_level

        return current_level

    def render_image(
        self, leaf_nodes: dict[tuple[int, int], torch.Tensor], is_horizontal: bool
    ) -> torch.Tensor:
        x_idxs, y_idxs = zip(*leaf_nodes.keys())
        width = max(x_idxs) + 1
        height = max(y_idxs) + 1

        # Store color logits rather than probabilities
        image_logits = torch.zeros(height, width, self.num_colors, device=self.device)

        node_colors = self.h_node_colors if is_horizontal else self.v_node_colors

        for (x, y), node_activations in leaf_nodes.items():
            cell_logits = torch.matmul(
                node_activations.unsqueeze(0), node_colors
            ).squeeze(0)
            image_logits[y, x] = cell_logits

        return image_logits


def load_image(
    image_src: str, num_colors: int, depth: int
) -> tuple[np.ndarray, list[tuple]]:
    h_splits = (depth + 1) // 2
    v_splits = depth // 2
    width = 2**h_splits
    height = 2**v_splits

    # Download image if it's a URL
    if image_src.startswith("http"):
        response = requests.get(image_src)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_src)

    # Resize image to match target dimensions
    img = img.resize((width, height), Image.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img)

    # Handle both RGB and grayscale
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Drop alpha channel

    # Reshape to 2D array of pixels
    pixels = img_array.reshape(-1, 3)

    # Use KMeans to find the specified number of colors
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)

    # Get the color palette (in RGB)
    color_map = [tuple(map(int, color)) for color in kmeans.cluster_centers_]

    # Assign each pixel to the nearest cluster center
    labels = kmeans.predict(pixels).reshape(height, width)

    # Create one-hot encoded array
    quantized = np.zeros((height, width, num_colors), dtype=np.float32)
    for i in range(num_colors):
        quantized[:, :, i] = (labels == i).astype(np.float32)

    return quantized, color_map


def create_target_image(
    depth: int, num_colors: int, mode: str = "stripes"
) -> tuple[np.ndarray, list[tuple]]:
    """
    Create a target image for training, along with its color map.

    Args:
        depth: Depth of the fractal
        num_colors: Number of colors to use
        mode: 'stripes', 'checkerboard', or 'circular'

    Returns:
        Tuple of (target image as one-hot encoded array, color map as RGB tuples)
    """
    h_splits = (depth + 1) // 2
    v_splits = depth // 2

    width = 2**h_splits
    height = 2**v_splits

    # Create one-hot encoded target
    target = np.zeros((height, width, num_colors), dtype=np.float32)

    # Create a color map using a colorful palette
    if mode == "stripes":
        color_map = [plt.cm.rainbow(i / num_colors)[:3] for i in range(num_colors)]
        for i in range(num_colors):
            mask_x = (i * width // num_colors <= np.arange(width)) & (
                np.arange(width) < (i + 1) * width // num_colors
            )
            for y in range(height):
                if y % 2 == 0:
                    target[y, mask_x, i] = 1.0
                else:
                    target[y, mask_x, (i + 1) % num_colors] = 1.0

    elif mode == "checkerboard":
        color_map = [plt.cm.viridis(i / num_colors)[:3] for i in range(num_colors)]
        for y in range(height):
            for x in range(width):
                color_idx = (x + y) % num_colors
                target[y, x, color_idx] = 1.0

    elif mode == "circular":
        color_map = [plt.cm.plasma(i / num_colors)[:3] for i in range(num_colors)]
        center_y, center_x = height // 2, width // 2
        max_radius = min(height, width) // 2

        for y in range(height):
            for x in range(width):
                dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                color_idx = int((dist / max_radius) * num_colors) % num_colors
                target[y, x, color_idx] = 1.0

    # Convert color map to 8-bit RGB
    color_map = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_map]

    return target, color_map


def train_fractal_net(
    target: np.ndarray,
    color_map: list[tuple],
    num_colors: int = 8,
    depth: int = 5,
    num_nodes: int = 20,
    lr: float = 0.001,
    epochs: int = 1000,
    device: str = "cuda",
    viz_every: int = 50,
):
    """
    Train a fractal network to approximate a target image.

    Args:
        target: Target image as one-hot encoded numpy array
        color_map: List of RGB tuples representing the colors
        num_colors: Number of colors in the target
        depth: Depth of the fractal
        num_nodes: Number of nodes in the model
        lr: Learning rate
        epochs: Number of training epochs
        device: Computation device
        viz_every: Visualize progress every N epochs

    Returns:
        Trained model and loss history
    """
    target = target.astype(np.float32)

    model = FractalNet(num_nodes=num_nodes, num_colors=num_colors, device=device)
    model.to(device)

    h_splits = (depth + 1) // 2
    v_splits = depth // 2

    width = 2**h_splits
    height = 2**v_splits

    if target.shape != (height, width, num_colors):
        raise ValueError(
            f"Target image should have shape {(height, width, num_colors)}, "
            f"got {target.shape}"
        )

    target_tensor = torch.tensor(target, device=device, dtype=torch.float32)
    target_indices = torch.argmax(target_tensor, dim=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=50,
        factor=0.5,
    )

    loss_history = []

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # Get color logits
        logits = model(depth)

        # Reshape for cross entropy loss
        logits_flat = logits.reshape(-1, num_colors)
        targets_flat = target_indices.reshape(-1)

        # Calculate cross entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())

        if epoch % viz_every == 0 or epoch == epochs - 1:
            visualize_progress(target_tensor, logits, loss_history, epoch, color_map)

    return model, loss_history


def visualize_progress(
    target: torch.Tensor,
    logits: torch.Tensor,
    loss_history: list,
    epoch: int,
    color_map: list[tuple],
):
    with torch.no_grad():
        current_output = logits.detach().cpu().numpy()
        target_np = target.cpu().numpy()

    plt.figure(figsize=(7, 2.5))

    # Create label maps using color map for better visualization
    target_indices = np.argmax(target_np, axis=2)
    pred_indices = np.argmax(current_output, axis=2)

    height, width = target_indices.shape
    target_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    output_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            target_idx = target_indices[y, x]
            pred_idx = pred_indices[y, x]

            if target_idx < len(color_map):
                target_rgb[y, x] = color_map[target_idx]

            if pred_idx < len(color_map):
                output_rgb[y, x] = color_map[pred_idx]

    plt.subplot(1, 3, 1)
    plt.imshow(target_rgb)
    plt.title("Target Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(output_rgb)
    plt.title(f"Model Output (Epoch {epoch})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    accuracy = np.mean(pred_indices == target_indices)
    plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()




def visualize_fractal_image(
    model: FractalNet, depth: int, color_map: list[tuple], figsize: tuple = (7, 7)
):
    """
    Visualize the output of the fractal model.

    Args:
        model: The trained fractal model
        depth: Depth to render at
        color_map: List of RGB tuples representing colors
        figsize: Size of the figure
    """
    with torch.no_grad():
        # Get the model output
        logits = model(depth)
        pred_indices = torch.argmax(logits, dim=2).cpu().numpy()

        # Create RGB image using the color map
        height, width = pred_indices.shape
        output_img = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                color_idx = pred_indices[y, x]
                if color_idx < len(color_map):
                    output_img[y, x] = color_map[color_idx]

    # Display the image
    plt.figure(figsize=figsize)
    plt.imshow(output_img)
    plt.title(f"Fractal Approximation (Depth {depth})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return output_img


def render_html_fractal_old(
    model: FastFractalNet,
    color_map: list[tuple],
    output_dir: str = "fractal-output",
    depth: int = 5,
    delay_seconds: int = 2
):
    """
    Render the trained fractal model as a single HTML file with JavaScript animation.
    This version matches the exact color behavior shown in visualize_fractal_image.

    Args:
        model: The trained FastFractalNet model
        color_map: List of RGB tuples representing the color palette
        output_dir: Directory to save the HTML file
        depth: Maximum depth to render
        delay_seconds: Seconds between splits
    """
    import os
    import json
    from pathlib import Path

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # First, get the actual model output at each depth level
    # This ensures we match the exact colors from visualize_fractal_image
    node_maps = []

    with torch.no_grad():
        for d in range(1, depth + 1):
            # Get model output for this depth
            logits = model(d)
            pred_indices = torch.argmax(logits, dim=2).cpu().numpy()

            # Get the dimensions for this depth
            is_horizontal = (d - 1) % 2 == 0
            h_splits = (d + 1) // 2
            v_splits = d // 2
            width = 2**h_splits
            height = 2**v_splits

            # Create node map for this depth
            node_map = {
                "depth": d,
                "is_horizontal": is_horizontal,
                "width": width,
                "height": height,
                "grid": []
            }

            # Store colors in a grid format
            for y in range(height):
                row = []
                for x in range(width):
                    color_idx = pred_indices[y, x]
                    if color_idx < len(color_map):
                        r, g, b = color_map[color_idx]
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                    else:
                        hex_color = "#808080"  # Fallback gray
                    row.append(hex_color)
                node_map["grid"].append(row)

            node_maps.append(node_map)

    # Create the HTML file with JavaScript animation
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Fractal Animation!!</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #000;
            font-family: monospace;
        }}

        html, body, table {{
            width: 100%;
            height: 100%;
            border-collapse: collapse;
        }}

        td {{
            padding: 0;
            position: relative;
            transition: background-color 0.5s;
        }}
    </style>
</head>
<body>
    <table id="fractal-table"></table>

    <script>
        // Node data from the trained model organized by depth
        const nodeMaps = {json.dumps(node_maps)};

        // Configuration
        const delaySeconds = {delay_seconds};
        let currentDepth = 0;
        const maxDepth = {depth};

        // Initialize the table with the root cell
        function initTable() {{
            const table = document.getElementById('fractal-table');
            table.innerHTML = '<tr><td id="root-cell"></td></tr>';

            // Set initial color
            const rootCell = document.getElementById('root-cell');
            rootCell.style.backgroundColor = nodeMaps[0].grid[0][0];
        }}

        // Update the table to show the fractal at the given depth
        function updateToDepth(depth) {{
            if (depth >= nodeMaps.length) return;

            const nodeMap = nodeMaps[depth];
            const table = document.getElementById('fractal-table');
            table.innerHTML = '';

            // Create the table structure based on the node map
            for (let y = 0; y < nodeMap.height; y++) {{
                const row = document.createElement('tr');
                row.style.height = `${{100 / nodeMap.height}}%`;

                for (let x = 0; x < nodeMap.width; x++) {{
                    const cell = document.createElement('td');
                    cell.style.width = `${{100 / nodeMap.width}}%`;
                    cell.style.backgroundColor = nodeMap.grid[y][x];
                    row.appendChild(cell);
                }}

                table.appendChild(row);
            }}
        }}

        // Initialize
        initTable();

        // Schedule updates
        function scheduleUpdates() {{
            setTimeout(() => {{
                currentDepth++;
                if (currentDepth < maxDepth) {{
                    updateToDepth(currentDepth);
                    scheduleUpdates();
                }}
            }}, delaySeconds * 1000);
        }}

        // Start the animation
        scheduleUpdates();
    </script>
</body>
</html>
"""

    # Write the HTML file
    with open(os.path.join(output_dir, "fractal.html"), "w") as f:
        f.write(html)

    print(f"Generated fractal animation in {output_dir}/fractal.html")


def render_html_fractal(
    model: FastFractalNet,
    color_map: list[tuple],
    output_dir: str = "fractal-output",
):
    """
    Render the trained fractal model as a single HTML file with JavaScript animation
    that can infinitely generate the fractal pattern.

    Args:
        model: The trained FastFractalNet model
        color_map: List of RGB tuples representing the color palette
        output_dir: Directory to save the HTML file
    """
    import os
    import json
    from pathlib import Path

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert the color map to hex format
    hex_color_map = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in color_map]

    # Extract node transition matrices
    with torch.no_grad():
        # Make sure transition matrices are updated
        model.build_transition_matrices()

        # Extract horizontal node transitions
        h_transitions = {
            "left": model.h_transition_left.cpu().numpy().tolist(),
            "right": model.h_transition_right.cpu().numpy().tolist()
        }

        # Extract vertical node transitions
        v_transitions = {
            "left": model.v_transition_left.cpu().numpy().tolist(),
            "right": model.v_transition_right.cpu().numpy().tolist()
        }

        # Extract color logits for nodes
        h_color_logits = model.h_node_colors.cpu().numpy()
        v_color_logits = model.v_node_colors.cpu().numpy()

        # Convert logits to color indices
        h_colors = np.argmax(h_color_logits, axis=1).tolist()
        v_colors = np.argmax(v_color_logits, axis=1).tolist()

    # Create the HTML file with JavaScript for infinite rendering
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Infinite Fractal Animation</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #000;
            font-family: monospace;
            color: white;
        }}

        #fractal-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}

        .cell {{
            position: absolute;
            transition: all 0.5s ease;
        }}

        #controls {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            z-index: 100;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }}

        button {{
            margin: 5px;
            padding: 5px 10px;
        }}
    </style>
</head>
<body>
    <div id="fractal-container"></div>
    <div id="controls">
        <button id="btn-split">Split</button>
        <button id="btn-reset">Reset</button>
        <span id="depth-display">Depth: 0</span>
    </div>

    <script>
        // Node transition data from the trained model
        const hTransitions = {json.dumps(h_transitions)};
        const vTransitions = {json.dumps(v_transitions)};
        const hNodeColors = {json.dumps(h_colors)};
        const vNodeColors = {json.dumps(v_colors)};
        const colorMap = {json.dumps(hex_color_map)};

        // State tracking
        let currentDepth = 0;
        let cells = [];

        // Numerical stability threshold - ignore tiny values
        const EPSILON = 1e-8;

        // Initialize with root cell
        function initialize() {{
            const container = document.getElementById('fractal-container');
            container.innerHTML = '';

            // Start with node 0 at root
            const rootNodeActivations = Array(hNodeColors.length).fill(0);
            rootNodeActivations[0] = 1.0;

            // Create root cell
            const rootCell = {{
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                nodeActivations: rootNodeActivations,
                element: document.createElement('div')
            }};

            rootCell.element.className = 'cell';
            rootCell.element.style.left = '0%';
            rootCell.element.style.top = '0%';
            rootCell.element.style.width = '100%';
            rootCell.element.style.height = '100%';

            // For the root (depth 0), we use the vertical node color
            const colorIdx = vNodeColors[0];  // Root always starts with node 0
            rootCell.element.style.backgroundColor = colorMap[colorIdx];

            container.appendChild(rootCell.element);
            cells = [rootCell];

            // Update depth display
            document.getElementById('depth-display').textContent = 'Depth: ' + currentDepth;
        }}

        // Split all cells at current depth
        function splitCells() {{
            if (cells.length === 0) return;

            const container = document.getElementById('fractal-container');
            const newCells = [];
            const isHorizontalSplit = (currentDepth % 2 === 0);

            // Process each cell at current depth
            for (const cell of cells) {{
                // Remove the current cell element
                container.removeChild(cell.element);

                // Calculate split dimensions and positions
                let child1, child2;

                if (isHorizontalSplit) {{
                    // Horizontal split (left and right)
                    child1 = {{
                        x: cell.x,
                        y: cell.y,
                        width: cell.width / 2,
                        height: cell.height,
                        nodeActivations: calculateChildActivations(cell.nodeActivations, 'left', true)
                    }};

                    child2 = {{
                        x: cell.x + cell.width / 2,
                        y: cell.y,
                        width: cell.width / 2,
                        height: cell.height,
                        nodeActivations: calculateChildActivations(cell.nodeActivations, 'right', true)
                    }};
                }} else {{
                    // Vertical split (top and bottom)
                    child1 = {{
                        x: cell.x,
                        y: cell.y,
                        width: cell.width,
                        height: cell.height / 2,
                        nodeActivations: calculateChildActivations(cell.nodeActivations, 'left', false)
                    }};

                    child2 = {{
                        x: cell.x,
                        y: cell.y + cell.height / 2,
                        width: cell.width,
                        height: cell.height / 2,
                        nodeActivations: calculateChildActivations(cell.nodeActivations, 'right', false)
                    }};
                }}

                // Create and position DOM elements for children
                child1.element = document.createElement('div');
                child1.element.className = 'cell';
                child1.element.style.left = child1.x * 100 + '%';
                child1.element.style.top = child1.y * 100 + '%';
                child1.element.style.width = child1.width * 100 + '%';
                child1.element.style.height = child1.height * 100 + '%';

                child2.element = document.createElement('div');
                child2.element.className = 'cell';
                child2.element.style.left = child2.x * 100 + '%';
                child2.element.style.top = child2.y * 100 + '%';
                child2.element.style.width = child2.width * 100 + '%';
                child2.element.style.height = child2.height * 100 + '%';

                // In FastFractalNet.forward, the color determination happens with:
                // is_final_split_horizontal = (depth - 1) % 2 == 0
                // node_colors = self.h_node_colors if is_final_split_horizontal else self.v_node_colors

                // The next depth will be currentDepth + 1, so use that to determine colors
                const willFinalSplitBeHorizontal = ((currentDepth + 1) - 1) % 2 === 0;
                const nodeColors = willFinalSplitBeHorizontal ? hNodeColors : vNodeColors;

                // Set colors based on max node activation
                const color1 = getColorForActivations(child1.nodeActivations, nodeColors);
                const color2 = getColorForActivations(child2.nodeActivations, nodeColors);

                child1.element.style.backgroundColor = colorMap[color1];
                child2.element.style.backgroundColor = colorMap[color2];

                // Add children to container
                container.appendChild(child1.element);
                container.appendChild(child2.element);

                // Store new cells
                newCells.push(child1, child2);
            }}

            // Update cells array
            cells = newCells;
            currentDepth++;

            // Update depth display
            document.getElementById('depth-display').textContent = 'Depth: ' + currentDepth;
        }}

        // Calculate child node activations based on parent activations and transitions
        function calculateChildActivations(parentActivations, direction, isHorizontal) {{
            const transitions = isHorizontal ? hTransitions : vTransitions;
            const directionTransitions = transitions[direction];

            // Initialize child activations
            const childActivations = Array(parentActivations.length).fill(0);

            // Apply transitions with improved numerical stability
            let totalActivation = 0;

            for (let parentNodeIdx = 0; parentNodeIdx < parentActivations.length; parentNodeIdx++) {{
                const parentWeight = parentActivations[parentNodeIdx];

                // Skip negligible weights to prevent accumulation of tiny errors
                if (parentWeight < EPSILON) continue;

                const nodeTransitions = directionTransitions[parentNodeIdx];

                for (let childNodeIdx = 0; childNodeIdx < nodeTransitions.length; childNodeIdx++) {{
                    const transitionValue = nodeTransitions[childNodeIdx];

                    // Skip negligible transition values
                    if (transitionValue < EPSILON) continue;

                    const activation = parentWeight * transitionValue;
                    childActivations[childNodeIdx] += activation;
                    totalActivation += activation;
                }}
            }}

            // Normalize if total is significant (prevents division by very small numbers)
            if (totalActivation > EPSILON) {{
                for (let i = 0; i < childActivations.length; i++) {{
                    childActivations[i] /= totalActivation;
                }}
            }} else {{
                // If total is too small, just use the first node
                childActivations[0] = 1.0;
            }}

            return childActivations;
        }}

        // Get color based on node activations
        function getColorForActivations(activations, nodeColors) {{
            // Find node with highest activation
            let maxActivation = -1;
            let maxNodeIdx = 0;

            for (let i = 0; i < activations.length; i++) {{
                if (activations[i] > maxActivation) {{
                    maxActivation = activations[i];
                    maxNodeIdx = i;
                }}
            }}

            return nodeColors[maxNodeIdx];
        }}

        // Event listeners
        document.getElementById('btn-split').addEventListener('click', splitCells);
        document.getElementById('btn-reset').addEventListener('click', () => {{
            currentDepth = 0;
            initialize();
        }});

        // Initialize on load
        window.onload = initialize;
    </script>
</body>
</html>
"""

    # Write the HTML file
    with open(os.path.join(output_dir, "fractal.html"), "w") as f:
        f.write(html)

    print(f"Generated infinite fractal animation in {output_dir}/fractal.html")


def debug_fractal_traversal(
    model: FastFractalNet,
    color_map: list[tuple],
    depth: int = 4,  # Start with a smaller depth to identify issues early
    output_path: str = "debug_traversal.png",
    figsize: tuple = (7, 7)
):
    """
    Generate an image using the same tree traversal logic as the JavaScript renderer.
    Also closely examine the model's forward pass to identify discrepancies.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    # Extract all necessary model parameters
    with torch.no_grad():
        # Make sure transition matrices are updated
        model.build_transition_matrices()

        # Extract all model parameters we need
        h_transitions_left = model.h_transition_left.cpu().numpy()
        h_transitions_right = model.h_transition_right.cpu().numpy()
        v_transitions_left = model.v_transition_left.cpu().numpy()
        v_transitions_right = model.v_transition_right.cpu().numpy()

        # Get actual color matrices, not just the argmax indices
        h_colors_matrix = model.h_node_colors.cpu().numpy()
        v_colors_matrix = model.v_node_colors.cpu().numpy()

    # Calculate grid dimensions
    h_splits = (depth + 1) // 2
    v_splits = depth // 2
    width = 2**h_splits
    height = 2**v_splits

    # Create an image array
    traversal_image = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Manually implementing the model's forward pass ---

    # Storage for activations at each level
    level_activations = {}

    # Start with root
    root_activations = np.zeros(model.nodes_per_direction)
    root_activations[0] = 1.0

    # Function to precisely compute child activations
    def compute_activations(parent_activations, transitions):
        result = np.zeros_like(parent_activations)
        for parent_idx, parent_weight in enumerate(parent_activations):
            if parent_weight > 0:
                for child_idx, trans_value in enumerate(transitions[parent_idx]):
                    result[child_idx] += parent_weight * trans_value
        return result

    # Trace through the model's forward pass step by step
    current_activations = {(0, 0): root_activations}
    level_activations[0] = current_activations

    # Process each level
    for d in range(depth):
        is_horizontal = d % 2 == 0
        next_activations = {}

        for (x, y), node_acts in current_activations.items():
            if is_horizontal:
                # Horizontal split
                left_acts = compute_activations(node_acts, h_transitions_left)
                right_acts = compute_activations(node_acts, h_transitions_right)

                next_activations[(x*2, y)] = left_acts
                next_activations[(x*2+1, y)] = right_acts
            else:
                # Vertical split
                top_acts = compute_activations(node_acts, v_transitions_left)
                bottom_acts = compute_activations(node_acts, v_transitions_right)

                next_activations[(x, y*2)] = top_acts
                next_activations[(x, y*2+1)] = bottom_acts

        current_activations = next_activations
        level_activations[d+1] = current_activations

    # Generate image from leaf activations
    is_final_split_horizontal = (depth - 1) % 2 == 0
    color_matrix = h_colors_matrix if is_final_split_horizontal else v_colors_matrix

    for (x, y), acts in current_activations.items():
        # This is the key difference - we compute color logits through matrix multiplication
        # just like the model does, rather than using argmax directly on node activations
        color_logits = np.matmul(acts[np.newaxis, :], color_matrix).squeeze(0)
        color_idx = np.argmax(color_logits)
        color = color_map[color_idx]
        traversal_image[y, x] = color

    # --- Now run the model's actual forward method for comparison ---
    with torch.no_grad():
        model_logits = model(depth)
        model_indices = torch.argmax(model_logits, dim=2).cpu().numpy()

        # Create image from model output
        model_image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                color_idx = model_indices[y, x]
                if color_idx < len(color_map):
                    model_image[y, x] = color_map[color_idx]

    # --- Compare and analyze ---
    # Display both images for visual comparison
    plt.figure(figsize=(7, 3.5))
    plt.subplot(1, 2, 1)
    plt.imshow(traversal_image)
    plt.title("Manual Traversal")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(model_image)
    plt.title("Model.forward()")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Find differences
    diff_mask = ~np.all(traversal_image == model_image, axis=2)
    different_pixel_count = np.sum(diff_mask)
    print(f"Images identical: {not np.any(diff_mask)}")
    print(f"Number of different pixels: {different_pixel_count} out of {width*height}")

    # If there are differences, highlight them and analyze
    if different_pixel_count > 0:
        # Create difference image
        diff_image = np.zeros((height, width, 3), dtype=np.uint8)
        diff_image[diff_mask] = [255, 0, 0]

        plt.figure(figsize=(4, 4))
        plt.imshow(diff_image)
        plt.title("Pixel Differences (Red)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_diff.png'), dpi=150, bbox_inches='tight')
        plt.show()

        # Find the different pixels and analyze them
        diff_coords = np.where(diff_mask)
        for i in range(min(10, len(diff_coords[0]))):  # Look at first 10 differences
            y, x = diff_coords[0][i], diff_coords[1][i]

            # Get the leaf node activations for this pixel
            trav_activation = level_activations[depth].get((x, y), None)
            if trav_activation is not None:
                # Calculate color logits the same way the model does
                color_logits = np.matmul(trav_activation[np.newaxis, :], color_matrix).squeeze(0)
                trav_color_idx = np.argmax(color_logits)

                # Get the model's color for this pixel
                model_color_idx = model_indices[y, x]

                print(f"Pixel ({x}, {y}):")
                print(f"  Traversal color_idx={trav_color_idx}")
                print(f"  Model color_idx={model_color_idx}")
                print(f"  Color logits: {color_logits}")

                # Look at the top few logits to see if it's a close call
                sorted_logits = np.sort(color_logits)[::-1]
                if len(sorted_logits) > 1:
                    top_diff = sorted_logits[0] - sorted_logits[1]
                    print(f"  Top logit difference: {top_diff}")
                    if abs(top_diff) < 0.01:
                        print("  Very close logits - potential precision issue")
                print()

    # Return for further analysis
    return traversal_image, model_image, level_activations
