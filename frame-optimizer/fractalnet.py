import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


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

        self.h_node_nets = nn.ModuleList([
            NodeNet(self.nodes_per_direction, device=device) for _ in range(self.nodes_per_direction)
        ])
        self.v_node_nets = nn.ModuleList([
            NodeNet(self.nodes_per_direction, device=device) for _ in range(self.nodes_per_direction)
        ])

        # Initialize color logits instead of probabilities
        self.h_node_colors = nn.Parameter(torch.randn([self.nodes_per_direction, num_colors], device=device))
        self.v_node_colors = nn.Parameter(torch.randn([self.nodes_per_direction, num_colors], device=device))

    def forward(self, depth: int) -> torch.Tensor:
        leaf_nodes = self.process_tree(depth)
        is_horizontal = (depth - 1) % 2 == 0
        final_image = self.render_image(leaf_nodes, is_horizontal)
        return final_image

    def process_tree(self, depth: int) -> dict[tuple[int, int], torch.Tensor]:
        # Initialize with root node
        current_level = {(0, 0): torch.zeros(self.nodes_per_direction, device=self.device)}
        current_level[(0, 0)][0] = 1.0  # Node 0 is active at the root

        for d in range(depth):
            next_level = {}
            is_horizontal = (d % 2 == 0)
            node_nets = self.h_node_nets if is_horizontal else self.v_node_nets

            for (x, y), node_activations in current_level.items():
                # Skip if no significant activations
                if torch.max(node_activations) < 1e-6:
                    continue

                left_activations = torch.zeros(self.nodes_per_direction, device=self.device)
                right_activations = torch.zeros(self.nodes_per_direction, device=self.device)

                for node_idx in range(self.nodes_per_direction):
                    weight = node_activations[node_idx]
                    if weight > 1e-6:  # Only process active nodes
                        left_probs, right_probs = node_nets[node_idx]()
                        left_activations += left_probs * weight
                        right_activations += right_probs * weight

                if is_horizontal:
                    next_level[(x*2, y)] = left_activations
                    next_level[(x*2+1, y)] = right_activations
                else:
                    next_level[(x, y*2)] = left_activations
                    next_level[(x, y*2+1)] = right_activations

            current_level = next_level

        return current_level

    def render_image(self, leaf_nodes: dict[tuple[int, int], torch.Tensor], is_horizontal: bool) -> torch.Tensor:
        x_idxs, y_idxs = zip(*leaf_nodes.keys())
        width = max(x_idxs) + 1
        height = max(y_idxs) + 1

        # Store color logits rather than probabilities
        image_logits = torch.zeros(height, width, self.num_colors, device=self.device)

        node_colors = self.h_node_colors if is_horizontal else self.v_node_colors

        for (x, y), node_activations in leaf_nodes.items():
            cell_logits = torch.matmul(node_activations.unsqueeze(0), node_colors).squeeze(0)
            image_logits[y, x] = cell_logits

        return image_logits


def train_fractal_net(target_image: np.ndarray, num_colors: int = 3, depth: int = 5, num_nodes: int = 20,
                     lr: float = 0.001, epochs: int = 1000, device: str = "cuda", viz_every: int = 50):
    target_image = target_image.astype(np.float32)

    model = FractalNet(num_nodes=num_nodes, num_colors=num_colors, device=device)
    model.to(device)

    h_splits = (depth + 1) // 2
    v_splits = depth // 2

    width = 2 ** h_splits
    height = 2 ** v_splits

    if target_image.shape != (height, width, num_colors):
        raise ValueError(f"Target image should have shape {(height, width, num_colors)}, "
                         f"got {target_image.shape}")

    target = torch.tensor(target_image, device=device, dtype=torch.float32)
    target_indices = torch.argmax(target, dim=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, verbose=True)

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
            visualize_progress(target, logits, loss_history, epoch)

    return model, loss_history


def visualize_progress(target: torch.Tensor, logits: torch.Tensor, loss_history: list, epoch: int):
    with torch.no_grad():
        current_output = logits.detach().cpu().numpy()
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

    # Calculate and show accuracy
    pred_indices = np.argmax(current_output, axis=2)
    target_indices = np.argmax(target_np, axis=2)
    accuracy = np.mean(pred_indices == target_indices)

    print(f"{accuracy=}")  # TODO(andreas): remove debug


    plt.tight_layout()
    plt.show()


def create_target_image(depth: int, num_colors: int, mode: str = 'stripes') -> np.ndarray:
    h_splits = (depth + 1) // 2
    v_splits = depth // 2

    width = 2 ** h_splits
    height = 2 ** v_splits

    target = np.zeros((height, width, num_colors), dtype=np.float32)

    if mode == 'stripes':
        for i in range(num_colors):
            mask_x = ((i * width // num_colors <= np.arange(width)) &
                    (np.arange(width) < (i + 1) * width // num_colors))
            for y in range(height):
                if y % 2 == 0:
                    target[y, mask_x, i] = 1.0
                else:
                    target[y, mask_x, (i+1) % num_colors] = 1.0

    elif mode == 'checkerboard':
        for y in range(height):
            for x in range(width):
                color_idx = (x + y) % num_colors
                target[y, x, color_idx] = 1.0

    elif mode == 'circular':
        center_y, center_x = height // 2, width // 2
        max_radius = min(height, width) // 2

        for y in range(height):
            for x in range(width):
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                color_idx = int((dist / max_radius) * num_colors) % num_colors
                target[y, x, color_idx] = 1.0

    return target


def visualize_tree_structure(model: FractalNet, depth: int, save_path: str = None):
    """Visualize the tree structure of the fractal model"""
    with torch.no_grad():
        leaf_nodes = model.process_tree(depth)
        is_horizontal = (depth - 1) % 2 == 0

        x_idxs, y_idxs = zip(*leaf_nodes.keys())
        width = max(x_idxs) + 1
        height = max(y_idxs) + 1

        # Create a visualization showing which nodes are active at each position
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        # For each leaf node position
        for (x, y), node_activations in leaf_nodes.items():
            # Get the index of the most active node
            node_idx = torch.argmax(node_activations).item()

            # Calculate color based on node index
            color = plt.cm.viridis(node_idx / model.nodes_per_direction)

            # Create a rectangle at the cell position
            rect = plt.Rectangle((x, y), 1, 1, color=color, alpha=0.7)
            ax.add_patch(rect)

            # Add text showing the node index
            ax.text(x + 0.5, y + 0.5, str(node_idx),
                    horizontalalignment='center', verticalalignment='center')

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # To match image coordinates

        plt.title(f"Node activation pattern (depth {depth})")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()


def render_html_fractal(model: FractalNet, output_dir: str="fractal-output", delay_seconds: int=2):
    """
    Render the trained fractal model as a single HTML file with JavaScript animation.
    Creates a table-based simulation of fractal splitting that matches the model's forward pass.
    Splits all cells at the same level simultaneously without depth limitation.

    Args:
        model: The trained FractalNet model
        output_dir: Directory to save the HTML file
        delay_seconds: Seconds between splits
    """
    import os
    import json
    from pathlib import Path

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each node to get its children and color
    node_data = {"h": [], "v": []}

    # Get data for both horizontal and vertical nodes
    for direction in ["h", "v"]:
        is_horizontal = direction == "h"
        node_nets = model.h_node_nets if is_horizontal else model.v_node_nets
        node_colors = model.h_node_colors if is_horizontal else model.v_node_colors

        for node_idx in range(model.nodes_per_direction):
            with torch.no_grad():
                # Get child nodes
                left_probs, right_probs = node_nets[node_idx]()
                left_idx = torch.argmax(left_probs).item()
                right_idx = torch.argmax(right_probs).item()

                # Get color
                color_idx = torch.argmax(node_colors[node_idx]).item()
                color = plt.cm.rainbow(color_idx / model.num_colors)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]*255), int(color[1]*255), int(color[2]*255))

            # Store node data
            node_data[direction].append({
                "id": node_idx,
                "color": hex_color,
                "left": left_idx,
                "right": right_idx
            })

    # Create the single HTML file with JavaScript animation
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Retro Fractal Animation</title>
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

        .debug-info {{
            position: absolute;
            bottom: 2px;
            right: 2px;
            font-size: 8px;
            color: rgba(255,255,255,0.5);
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <table id="root-table">
        <tr>
            <td id="root-cell" data-node-dir="h" data-node-id="0" data-depth="0"></td>
        </tr>
    </table>

    <script>
        // Node data from the trained model
        const nodeData = {json.dumps(node_data)};

        // Configuration
        const delaySeconds = {delay_seconds};

        // Track cells by their depth level
        let cellsByDepth = [[]];
        let currentDepth = 0;

        // Initialize with root node (h-0)
        let rootCell = document.getElementById('root-cell');
        rootCell.style.backgroundColor = nodeData.h[0].color;
        rootCell.innerHTML = '<div class="debug-info">h-0</div>';

        // Add root cell to depth 0
        cellsByDepth[0].push(rootCell);

        // Schedule the next split
        let nextSplitTime = Date.now() + delaySeconds * 1000;

        // Main animation loop
        function updateFractal() {{
            const now = Date.now();

            // Check if it's time to split
            if (now >= nextSplitTime) {{
                // Get the current level's cells
                const currentCells = cellsByDepth[currentDepth];

                // Create the next level array
                cellsByDepth[currentDepth + 1] = [];

                // Split all cells at the current depth
                currentCells.forEach(cell => {{
                    const nodeDir = cell.dataset.nodeDir;
                    const nodeId = parseInt(cell.dataset.nodeId);
                    const node = nodeData[nodeDir][nodeId];

                    const nextDir = (nodeDir === 'h') ? 'v' : 'h';

                    // Get child nodes
                    const leftChildId = node.left;
                    const rightChildId = node.right;
                    const leftChild = nodeData[nextDir][leftChildId];
                    const rightChild = nodeData[nextDir][rightChildId];

                    // Create a replacement cell with a nested table
                    const newCell = document.createElement('td');
                    newCell.style.padding = '0';

                    // Create nested table for the split
                    const nestedTable = document.createElement('table');
                    nestedTable.style.width = '100%';
                    nestedTable.style.height = '100%';

                    if (nodeDir === 'h') {{
                        // Horizontal split - one row with two columns
                        const row = document.createElement('tr');

                        // Left cell (50% width)
                        const leftTd = document.createElement('td');
                        leftTd.style.backgroundColor = leftChild.color;
                        leftTd.style.width = '50%';
                        leftTd.dataset.nodeDir = nextDir;
                        leftTd.dataset.nodeId = leftChildId;
                        leftTd.dataset.depth = (parseInt(cell.dataset.depth) + 1).toString();
                        leftTd.innerHTML = `<div class="debug-info">${{nextDir}}-${{leftChildId}}</div>`;

                        // Right cell (50% width)
                        const rightTd = document.createElement('td');
                        rightTd.style.backgroundColor = rightChild.color;
                        rightTd.style.width = '50%';
                        rightTd.dataset.nodeDir = nextDir;
                        rightTd.dataset.nodeId = rightChildId;
                        rightTd.dataset.depth = (parseInt(cell.dataset.depth) + 1).toString();
                        rightTd.innerHTML = `<div class="debug-info">${{nextDir}}-${{rightChildId}}</div>`;

                        // Assemble the row
                        row.appendChild(leftTd);
                        row.appendChild(rightTd);
                        nestedTable.appendChild(row);

                        // Add to next depth
                        cellsByDepth[currentDepth + 1].push(leftTd);
                        cellsByDepth[currentDepth + 1].push(rightTd);
                    }}
                    else {{
                        // Vertical split - two rows with one column each

                        // Top row (50% height)
                        const topRow = document.createElement('tr');
                        topRow.style.height = '50%';
                        const topTd = document.createElement('td');
                        topTd.style.backgroundColor = leftChild.color;
                        topTd.dataset.nodeDir = nextDir;
                        topTd.dataset.nodeId = leftChildId;
                        topTd.dataset.depth = (parseInt(cell.dataset.depth) + 1).toString();
                        topTd.innerHTML = `<div class="debug-info">${{nextDir}}-${{leftChildId}}</div>`;
                        topRow.appendChild(topTd);

                        // Bottom row (50% height)
                        const bottomRow = document.createElement('tr');
                        bottomRow.style.height = '50%';
                        const bottomTd = document.createElement('td');
                        bottomTd.style.backgroundColor = rightChild.color;
                        bottomTd.dataset.nodeDir = nextDir;
                        bottomTd.dataset.nodeId = rightChildId;
                        bottomTd.dataset.depth = (parseInt(cell.dataset.depth) + 1).toString();
                        bottomTd.innerHTML = `<div class="debug-info">${{nextDir}}-${{rightChildId}}</div>`;
                        bottomRow.appendChild(bottomTd);

                        // Assemble the table
                        nestedTable.appendChild(topRow);
                        nestedTable.appendChild(bottomRow);

                        // Add to next depth
                        cellsByDepth[currentDepth + 1].push(topTd);
                        cellsByDepth[currentDepth + 1].push(bottomTd);
                    }}

                    // Add nested table to new cell
                    newCell.appendChild(nestedTable);

                    // Replace the original cell with the new cell
                    if (cell.parentNode) {{
                        cell.parentNode.replaceChild(newCell, cell);
                    }}
                }});

                // Move to the next depth level
                currentDepth++;

                // Schedule the next split
                nextSplitTime = now + delaySeconds * 1000;
            }}

            // Continue the animation loop
            requestAnimationFrame(updateFractal);
        }}

        // Start the animation
        requestAnimationFrame(updateFractal);
    </script>
</body>
</html>
"""

    # Write the HTML file
    with open(os.path.join(output_dir, "index.html"), 'w') as f:
        f.write(html)

    print(f"Generated single HTML fractal animation in {output_dir}/index.html")
