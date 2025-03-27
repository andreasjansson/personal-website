import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F


class FractalFrameGenerator(nn.Module):
    def __init__(
        self,
        num_frames: int,
        frame_size: tuple[int, int],
        max_depth: int = 5,
        device="cuda",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.max_depth = max_depth
        self.device = device

        # Learnable frame colors
        frame_colors = torch.rand(num_frames, 3, device=device)
        self.frame_colors = nn.Parameter(frame_colors)

        # Frame selection is our learnable parameter
        # Initialize with slight preference for diversity
        frame_sel_init = torch.rand(num_frames, 2, num_frames, device=device) * 5
        self.frame_selection = nn.Parameter(frame_sel_init)

        # Fixed split direction and ratio
        self.split_directions = [i % 2 for i in range(num_frames)]
        split_ratios = torch.ones(num_frames, device=device) * 0.5  # Fixed at 50%
        self.register_buffer("split_ratios", split_ratios)

    def render_frame(
        self,
        frame_idx: int,
        depth: int = 0,
        start_y: int = 0,
        start_x: int = 0,
        height: int = None,
        width: int = None,
        temperature: float = 1.0,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        """
        Render a single frame with efficient weighted selection of child frames.

        Args:
            frame_idx: Index of the frame to render
            depth: Current recursion depth
            start_y, start_x: Starting coordinates
            height, width: Dimensions of the frame
            temperature: Temperature for softmax
            use_argmax: If True, use hard argmax instead of softmax
        """
        # Default dimensions for root frame
        if height is None:
            height = self.frame_size[0]
        if width is None:
            width = self.frame_size[1]

        # Base color for this frame
        bg_color = torch.sigmoid(self.frame_colors[frame_idx])

        # If at max depth or frame is too small, return solid color
        if depth >= self.max_depth or width < 2 or height < 2:
            return bg_color.view(3, 1, 1).expand(3, height, width)

        # Get split information
        is_vertical = self.split_directions[frame_idx]
        split_ratio = self.split_ratios[frame_idx]

        # Initialize the output tensor
        output = torch.zeros(3, height, width, device=self.device)

        # Calculate dimensions and slice information for both split directions
        if is_vertical:
            # Vertical split (columns)
            left_width = max(1, int(width * split_ratio))
            right_width = max(1, width - left_width)

            left_height = height
            right_height = height

            left_slice = slice(None), slice(None), slice(0, left_width)
            right_slice = slice(None), slice(None), slice(left_width, None)

            left_start_y, left_start_x = start_y, start_x
            right_start_y, right_start_x = start_y, start_x + left_width
        else:
            # Horizontal split (rows)
            top_height = max(1, int(height * split_ratio))
            bottom_height = max(1, height - top_height)

            left_width = width  # "left" is now "top"
            right_width = width  # "right" is now "bottom"

            left_height = top_height
            right_height = bottom_height

            left_slice = slice(None), slice(0, top_height), slice(None)
            right_slice = slice(None), slice(top_height, None), slice(None)

            left_start_y, left_start_x = start_y, start_x
            right_start_y, right_start_x = start_y + top_height, start_x

        # Determine which child frames to render
        if use_argmax:
            # Hard selection with argmax
            left_idx = torch.argmax(self.frame_selection[frame_idx, 0]).item()
            right_idx = torch.argmax(self.frame_selection[frame_idx, 1]).item()

            # Render left/top child
            left_child = self.render_frame(
                left_idx, depth + 1, left_start_y, left_start_x,
                left_height, left_width, temperature, use_argmax
            )
            output[left_slice] = left_child

            # Render right/bottom child
            right_child = self.render_frame(
                right_idx, depth + 1, right_start_y, right_start_x,
                right_height, right_width, temperature, use_argmax
            )
            output[right_slice] = right_child
        else:
            # Soft selection with softmax
            # Create frame selection probabilities with temperature
            left_probs = F.softmax(self.frame_selection[frame_idx, 0] / temperature, dim=0)
            right_probs = F.softmax(self.frame_selection[frame_idx, 1] / temperature, dim=0)

            # Determine how many top frames to consider based on depth
            k = 3 if depth < 2 else (2 if depth < 4 else 1)
            k = min(k, self.num_frames)

            # Get the top-k frame indices and their probabilities
            left_top_values, left_top_indices = torch.topk(left_probs, k)
            right_top_values, right_top_indices = torch.topk(right_probs, k)

            # Render and blend top-k left/top frames
            for i in range(k):
                if left_top_values[i] > 1e-3:  # Skip negligible contributions
                    idx = left_top_indices[i].item()
                    child_frame = self.render_frame(
                        idx, depth + 1, left_start_y, left_start_x,
                        left_height, left_width, temperature, use_argmax=False
                    )
                    output[left_slice] += left_top_values[i] * child_frame

            # Render and blend top-k right/bottom frames
            for i in range(k):
                if right_top_values[i] > 1e-3:  # Skip negligible contributions
                    idx = right_top_indices[i].item()
                    child_frame = self.render_frame(
                        idx, depth + 1, right_start_y, right_start_x,
                        right_height, right_width, temperature, use_argmax=False
                    )
                    output[right_slice] += right_top_values[i] * child_frame

        return output

    def forward(
        self, temperature: float = 1.0, is_training: bool = True
    ) -> torch.Tensor:
        """
        Generate the full fractal image.

        Args:
            temperature: Temperature for softmax
            is_training: If True, use softmax; if False, use argmax
        """
        h, w = self.frame_size
        return self.render_frame(
            0, 0, 0, 0, h, w, temperature=temperature, use_argmax=not is_training
        )

    def export_html(self) -> list[tuple[str, str]]:
        """Export the optimized frames as HTML files"""
        html_files = []

        # Generate HTML for each frame - use argmax for the final result
        for i in range(self.num_frames):
            is_vertical = self.split_directions[i]
            split_ratio = self.split_ratios[i].item() * 100  # Convert to percentage

            # Get the background color
            color = torch.sigmoid(self.frame_colors[i]).detach().cpu()
            r, g, b = [int(c * 255) for c in color]

            # Get the most probable child frame indices (argmax)
            left_idx = torch.argmax(self.frame_selection[i, 0]).item()
            right_idx = torch.argmax(self.frame_selection[i, 1]).item()

            # Create frameset based on split direction
            if is_vertical:
                frameset = f'<frameset cols="{split_ratio}%,{100 - split_ratio}%" style="background-color:rgb({r},{g},{b})">'
            else:
                frameset = f'<frameset rows="{split_ratio}%,{100 - split_ratio}%" style="background-color:rgb({r},{g},{b})">'

            html = f"""<html>
  <head><title>Fractal Frame {i}</title></head>
  {frameset}
    <frame src="frame_{left_idx}.html" scrolling="no">
    <frame src="frame_{right_idx}.html" scrolling="no">
  </frameset>
</html>"""
            html_files.append((f"frame_{i}.html", html))

        return html_files


def create_circle_target(w, h):
    # Create a clean circle target using PIL
    from PIL import Image, ImageDraw
    import numpy as np

    # Create a blank image
    img = Image.new("RGB", (w, h), color="black")
    draw = ImageDraw.Draw(img)

    # Draw a white circle
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.4)  # Circle radius as 40% of the smallest dimension
    left_top = (center[0] - radius, center[1] - radius)
    right_bottom = (center[0] + radius, center[1] + radius)
    # draw.ellipse([left_top, right_bottom], fill="white")
    draw.ellipse([left_top, right_bottom], fill="white")

    # Convert to tensor
    circle_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    circle_np = circle_np.transpose(2, 0, 1)  # Change to CxHxW format
    circle_target = torch.tensor(circle_np)

    return circle_target


def optimize_fractal(
    generator: nn.Module,
    target_image: torch.Tensor,
    iterations: int = 200,
    initial_lr: float = 0.2,
    grad_clip: float = 1.0,
    initial_temp: float = 5.0,  # Start with high temperature for exploration
    final_temp: float = 0.5,  # End with low temperature for exploitation
):
    """
    Optimize the fractal generator with temperature annealing and debugging info.
    """
    # Set generator to train mode
    generator.train()

    # Create optimizer
    optimizer = optim.Adam(
        [p for p in generator.parameters() if p.requires_grad], lr=initial_lr
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=initial_lr / 10
    )

    # Track loss history
    losses = []
    grad_norms = []

    for i in range(iterations):
        # Calculate current temperature (anneal from initial to final)
        temperature = initial_temp - (initial_temp - final_temp) * (i / iterations)

        # Zero gradients
        optimizer.zero_grad()

        # Generate image with current temperature
        generated_image = generator(temperature=temperature)

        # Compute loss
        loss = F.mse_loss(generated_image, target_image)
        losses.append(loss.item())

        # Backward pass
        loss.backward()

        # Check gradient norms
        fs_grad_norm = (
            generator.frame_selection.grad.norm().item()
            if generator.frame_selection.grad is not None
            else 0
        )
        grad_norms.append(fs_grad_norm)

        # Print detailed debug info periodically
        if i % (iterations // 20) == 0 or i == iterations - 1:
            print(f"Iteration {i}, Loss: {loss.item():.6f}, Temp: {temperature:.3f}")
            print(f"frame_selection grad norm: {fs_grad_norm:.6f}")

            # Check if gradients are reasonable
            if fs_grad_norm < 1e-6:
                print("WARNING: Very small gradients detected!")

            # Visualize current state
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))
            ax1.imshow(generated_image.permute(1, 2, 0).detach().cpu().numpy())
            ax1.set_title(f"Iteration {i}: Generated")
            ax1.axis("off")

            ax2.imshow(target_image.permute(1, 2, 0).detach().cpu().numpy())
            ax2.set_title("Target")
            ax2.axis("off")

            plt.tight_layout()
            plt.show()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

        # Step optimizer and scheduler
        optimizer.step()
        scheduler.step()

    # Plot loss and gradient norms
    plt.figure(figsize=(5, 2.5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(grad_norms)
    plt.title("Gradient Norm Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return generated_image.detach()


def optimize_for_circle(
    generator: FractalFrameGenerator,
    iterations: int = 500,
    lr: float = 0.1,
    grad_clip: float = 10.0,
):
    """Optimize the fractal to create a circle shape"""
    # Set generator to train mode
    generator.train()

    h, w = generator.frame_size
    circle_target = create_circle_target(w, h).to(generator.device)

    # Create separate optimizer groups with even higher learning rates for split_ratios
    # optimizer = optim.AdamW(
    #     [
    #         {"params": generator.leaf_colors, "lr": lr},
    #         {
    #             "params": generator.split_ratios,
    #             "lr": lr * 20,
    #         },  # Very high LR for split_ratios
    #         {"params": generator.frame_selection, "lr": lr * 0.5},
    #     ],
    #     weight_decay=1e-6,
    # )

    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=iterations, eta_min=lr / 10
    # )

    # optimizer = optim.SGD(lr=lr, params=generator.parameters())
    optimizer = optim.Adam(lr=lr, params=generator.parameters())
    # optimizer = optim.SGD([{"params": generator.leaf_colors, "lr": lr}, {"params": generator.split_ratios, "lr": lr}])

    # Training loop
    best_loss = float("inf")
    best_image = None

    for i in range(iterations):
        optimizer.zero_grad()

        # Generate image
        generated_image = generator()

        # Main loss: MSE between generated image and circle target
        # loss = F.mse_loss(generated_image, circle_target)
        loss = torch.mean((generated_image - circle_target) ** 2)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

        # Apply optimizer step
        optimizer.step()
        # scheduler.step()

        # Track best result
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_image = generated_image.clone().detach()

        # Detailed monitoring
        if i % int(iterations / 40) == 0 or i == iterations - 1:
            print(f"Iteration {i}")
            print(f"  Loss: {loss.item():.4f}")

            # Visualize current state vs target
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))
            ax1.imshow(generated_image.permute(1, 2, 0).detach().cpu().numpy())
            ax1.set_title(f"Iteration {i}: Generated")
            ax1.axis("off")

            ax2.imshow(circle_target.permute(1, 2, 0).detach().cpu().numpy())
            ax2.set_title("Target Circle")
            ax2.axis("off")

            plt.tight_layout()
            plt.show()

    # Return results
    return generator.export_html(), best_image


def optimize(
    generator: FractalFrameGenerator,
    siglip_model,
    siglip_processor,
    prompt: str,
    iterations: int = 500,
    lr: float = 0.005,
    grad_clip: float = 1.0,
):
    # Set models to eval mode but enable gradients
    siglip_model.eval()
    generator.train()

    # Get text embedding using processor
    text_inputs = siglip_processor(
        text=[prompt], padding="max_length", max_length=64, return_tensors="pt"
    ).to(siglip_model.device)

    with torch.no_grad():
        text_embedding = siglip_model.get_text_features(**text_inputs)
        text_embedding = F.normalize(text_embedding, dim=1)

    # Create separate optimizer groups with different learning rates
    # Extremely high learning rate for split_ratios to overcome tiny gradients
    optimizer = optim.SGD(lr=lr, params=generator.parameters())
    # optimizer = optim.AdamW([
    #    {'params': generator.leaf_colors, 'lr': lr},
    #    {'params': generator.split_ratios, 'lr': lr * 50},  # Very high LR
    #    {'params': generator.frame_selection, 'lr': lr * 0.5}
    # ], weight_decay=1e-6)  # Lower weight decay

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=lr / 10
    )

    # SigLIP2 normalization constants
    mean = (
        torch.tensor([0.48145466, 0.4578275, 0.40821073])
        .view(1, 3, 1, 1)
        .to(generator.device)
    )
    std = (
        torch.tensor([0.26862954, 0.26130258, 0.27577711])
        .view(1, 3, 1, 1)
        .to(generator.device)
    )

    # Training loop
    best_similarity = -1
    best_image = None

    # Track parameters across iterations
    split_ratio_history = []

    # Initial parameter values
    print("Initial parameters:")
    with torch.no_grad():
        split_values = torch.sigmoid(generator.split_ratios).detach().cpu().numpy()
        print(f"  split_ratios: {split_values}")
        split_ratio_history.append(split_values.copy())

    for i in range(iterations):
        optimizer.zero_grad()

        # Forward pass
        generated_image = generator()

        # Resize and normalize for SigLIP
        input_tensor = F.interpolate(
            generated_image.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        normalized_tensor = (input_tensor - mean) / std

        # Get image embeddings
        image_embedding = siglip_model.get_image_features(
            pixel_values=normalized_tensor
        )
        image_embedding = F.normalize(image_embedding, dim=1)

        # Compute similarity and loss
        raw_similarity = torch.matmul(image_embedding, text_embedding.t())[0][0]
        logit_scale = siglip_model.logit_scale.exp()
        logit_bias = siglip_model.logit_bias
        scaled_similarity = raw_similarity * logit_scale + logit_bias
        similarity_prob = torch.sigmoid(scaled_similarity)
        loss = -torch.log(similarity_prob.clamp(min=1e-6, max=1 - 1e-6))

        # Backward pass
        loss.backward()

        # Log split ratio gradients
        # with torch.no_grad():
        if False:
            if generator.split_ratios.grad is not None:
                sr_grad = generator.split_ratios.grad.clone().detach()
                sr_grad_norm = sr_grad.norm().item()

                # Print gradient info
                print(f"Iteration {i} - Split ratio gradients:")
                for j, grad_val in enumerate(sr_grad):
                    print(f"  Split {j}: grad = {grad_val.item():.10f}")
                print(f"  Norm: {sr_grad_norm:.10f}")

                # Even tiny gradients are useful with high enough learning rate
                if sr_grad.abs().max() < 1e-10:
                    print("  WARNING: Extremely small gradients detected")
            else:
                print(f"Iteration {i} - Split ratio grad is None!")

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

        # Apply optimizer step
        optimizer.step()
        scheduler.step()

        # Track best result
        current_similarity = similarity_prob.item()
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_image = generated_image.clone().detach()

        # Track parameters
        with torch.no_grad():
            split_values = torch.sigmoid(generator.split_ratios).detach().cpu().numpy()
            split_ratio_history.append(split_values.copy())

        # Detailed monitoring
        if i % 50 == 0 or i == iterations - 1:
            print(f"Iteration {i}")
            print(f"  Loss: {loss.item():.4f}, Similarity: {current_similarity:.4f}")
            print(f"  split_ratios: {split_values}")

            # Visualize current state
            plt.figure(figsize=(4, 4))
            plt.imshow(generated_image.permute(1, 2, 0).detach().cpu().numpy())
            plt.title(f"Iteration {i}: '{prompt}' (sim: {current_similarity:.3f})")
            plt.axis("off")
            plt.show()

    # Plot the split ratio history
    plt.figure(figsize=(10, 5))
    split_ratio_history = np.array(split_ratio_history)
    for j in range(split_ratio_history.shape[1]):
        plt.plot(split_ratio_history[:, j], label=f"Split {j}")
    plt.title("Split Ratio Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Split Ratio Value (sigmoid)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return results
    return generator.export_html(), best_image


def optimize_with_siglip(
    generator: FractalFrameGenerator,
    siglip_model,
    siglip_processor,
    prompt: str,
    iterations: int = 200,
    initial_lr: float = 0.1,
    grad_clip: float = 1.0,
    initial_temp: float = 5.0,
    final_temp: float = 0.5,
    device="cuda",
    log_every=10,
):
    """
    Optimize the fractal generator to match a text prompt using SigLIP2 similarity.
    """
    # Set models to appropriate modes
    siglip_model.eval()  # Keep in eval mode but ensure gradients flow
    generator.train()

    # Process the text prompt once
    text_inputs = siglip_processor(
        text=[prompt], padding="max_length", max_length=64, return_tensors="pt"
    ).to(device)

    # Get the text features (no need for gradients here)
    with torch.no_grad():
        text_features = siglip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=1)

    # SigLIP2 normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    # Create optimizer
    optimizer = optim.Adam(
        [p for p in generator.parameters() if p.requires_grad], lr=initial_lr
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=initial_lr / 10
    )

    # Track metrics
    losses = []
    similarities = []
    similarities_argmax = []
    best_similarity = -float("inf")
    best_similarity_argmax = -float("inf")
    best_image = None
    best_image_argmax = None

    for i in range(iterations):
        # Calculate current temperature (anneal from initial to final)
        temperature = initial_temp - (initial_temp - final_temp) * (i / iterations)

        # Zero gradients
        optimizer.zero_grad()

        # Generate image with current temperature (soft version for training)
        generated_image = generator(temperature=temperature, is_training=True)

        # Also generate the argmax version for comparison
        with torch.no_grad():
            generated_image_argmax = generator(
                temperature=temperature, is_training=False
            )

        # Ensure the image has the correct shape (B, C, H, W)
        image_tensor = generated_image.unsqueeze(0)

        # Resize to 224x224 if needed
        if image_tensor.shape[2] != 224 or image_tensor.shape[3] != 224:
            image_tensor = F.interpolate(
                image_tensor, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Normalize the image
        normalized_image = (image_tensor - mean) / std

        # Get image features
        image_features = siglip_model.get_image_features(pixel_values=normalized_image)
        image_features = F.normalize(image_features, dim=1)

        # Calculate similarity
        similarity = torch.matmul(image_features, text_features.t())[0][0]

        # Apply scaling and bias as in SigLIP2
        logit_scale = siglip_model.logit_scale.exp()
        logit_bias = siglip_model.logit_bias
        scaled_similarity = similarity * logit_scale + logit_bias

        # Convert to probability with sigmoid
        similarity_prob = torch.sigmoid(scaled_similarity)

        # Loss is negative similarity (we want to maximize similarity)
        loss = -similarity

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

        # Step optimizer and scheduler
        optimizer.step()
        scheduler.step()

        # Evaluate the argmax version
        with torch.no_grad():
            # Process the argmax image
            image_tensor_argmax = generated_image_argmax.unsqueeze(0)
            if (
                image_tensor_argmax.shape[2] != 224
                or image_tensor_argmax.shape[3] != 224
            ):
                image_tensor_argmax = F.interpolate(
                    image_tensor_argmax,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            normalized_image_argmax = (image_tensor_argmax - mean) / std

            # Get features and similarity
            image_features_argmax = siglip_model.get_image_features(
                pixel_values=normalized_image_argmax
            )
            image_features_argmax = F.normalize(image_features_argmax, dim=1)
            similarity_argmax = torch.matmul(image_features_argmax, text_features.t())[
                0
            ][0]
            scaled_similarity_argmax = similarity_argmax * logit_scale + logit_bias
            similarity_prob_argmax = torch.sigmoid(scaled_similarity_argmax)

        # Track metrics
        current_loss = loss.item()
        current_similarity = similarity_prob.item()
        current_similarity_argmax = similarity_prob_argmax.item()

        losses.append(current_loss)
        similarities.append(current_similarity)
        similarities_argmax.append(current_similarity_argmax)

        # Save best results for both versions
        if current_similarity > best_similarity:
            best_similarity = current_similarity
            best_image = generated_image.detach().clone()

        if current_similarity_argmax > best_similarity_argmax:
            best_similarity_argmax = current_similarity_argmax
            best_image_argmax = generated_image_argmax.detach().clone()

        # Print progress
        if i % log_every == 0 or i == iterations - 1:
            print(f"Iteration {i}")
            print(f"  Loss: {current_loss:.4f}")
            print(f"  Similarity (soft): {current_similarity:.4f}")
            print(f"  Similarity (argmax): {current_similarity_argmax:.4f}")
            print(f"  Temperature: {temperature:.2f}")

            # Visualize current state - show both soft and argmax versions
            fig, axes = plt.subplots(1, 3, figsize=(6, 3))

            # Soft version
            axes[0].imshow(
                generated_image.permute(1, 2, 0).cpu().detach().numpy().clip(0, 1)
            )
            axes[0].set_title(f"Soft (sim: {current_similarity:.3f})")
            axes[0].axis("off")

            # Argmax version
            axes[1].imshow(
                generated_image_argmax.permute(1, 2, 0)
                .cpu()
                .detach()
                .numpy()
                .clip(0, 1)
            )
            axes[1].set_title(
                f"Argmax (sim: {current_similarity_argmax:.3f})"
            )
            axes[1].axis("off")

            # Plot similarity progress
            axes[2].plot(similarities, label="Soft")
            axes[2].plot(similarities_argmax, label="Argmax")
            axes[2].set_title("Similarity Progress")
            axes[2].set_xlabel("Iteration")
            axes[2].set_ylabel("Similarity")
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            plt.show()

    # Final visualization of best results
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # Best soft result
    axes[0].imshow(best_image.permute(1, 2, 0).cpu().detach().numpy().clip(0, 1))
    axes[0].set_title(f"Best soft result (sim: {best_similarity:.4f})")
    axes[0].axis("off")

    # Best argmax result
    axes[1].imshow(best_image_argmax.permute(1, 2, 0).cpu().detach().numpy().clip(0, 1))
    axes[1].set_title(
        f"Best argmax result (sim: {best_similarity_argmax:.4f})"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Return the best images and other metrics
    return {
        "best_image_soft": best_image,
        "best_image_argmax": best_image_argmax,
        "best_similarity_soft": best_similarity,
        "best_similarity_argmax": best_similarity_argmax,
        "similarity_history_soft": similarities,
        "similarity_history_argmax": similarities_argmax,
        "loss_history": losses,
        "html_files": generator.export_html(),
    }


# Example usage:
#   ckpt = "google/siglip2-base-patch16-224"
#   model = AutoModel.from_pretrained(ckpt).eval().to("cuda")
#   processor = AutoProcessor.from_pretrained(ckpt)
#   generator = FractalFrameGenerator(num_frames, frame_size).to(model.device)

# html_files, final_image = optimize_frames_to_prompt("a flower", num_frames=12, iterations=300)
