import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm


class FractalFrameEnv(EnvBase):
    """
    A custom TorchRL environment for fractal frame generation.

    This environment allows optimization of fractal frame parameters using RL.
    """

    def __init__(
        self,
        num_frames: int,
        frame_size: tuple[int, int],
        max_depth: int = 5,
        siglip_model=None,
        siglip_processor=None,
        prompt: str = "",
        device="cuda",
    ):
        # Initialize base environment
        super().__init__(device=device, batch_size=[])

        # Store parameters
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.max_depth = max_depth
        self.siglip_model = siglip_model
        self.siglip_processor = siglip_processor
        self.prompt = prompt

        # Process the text prompt to get text features
        if siglip_model is not None and siglip_processor is not None:
            with torch.no_grad():
                text_inputs = siglip_processor(
                    text=[prompt],
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                ).to(device)
                text_features = siglip_model.get_text_features(**text_inputs)
                self.text_features = torch.nn.functional.normalize(text_features, dim=1)
        else:
            self.text_features = None

        # Initialize environment parameters
        self.frame_colors = torch.rand(num_frames, 3, device=device)
        self.split_directions = [i % 2 for i in range(num_frames)]
        self.split_ratios = torch.ones(num_frames, device=device) * 0.5
        self.frame_selection = torch.zeros(num_frames, 2, num_frames, device=device)

        # Initialize with sequential mapping for frames
        for i in range(num_frames):
            self.frame_selection[i, 0, (i + 1) % num_frames] = 1.0
            self.frame_selection[i, 1, (i + 2) % num_frames] = 1.0

        # Episode state
        self.current_step = 0
        self.max_episode_steps = 20

        # RNG for reproducibility
        self.rng = torch.Generator(device=self.device)
        self._set_seed(42)  # Default seed

        # Setup observation and action spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Setup the observation and action spaces"""
        from torchrl.data import (
            BoundedTensorSpec,
            CompositeSpec,
            UnboundedContinuousTensorSpec,
            DiscreteTensorSpec,
        )

        # Observation space: flattened frame parameters
        obs_dim = self.num_frames * (3 + 2 * self.num_frames)
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=torch.Size([obs_dim]), device=self.device
            ),
            step_count=DiscreteTensorSpec(
                n=self.max_episode_steps + 1,
                shape=torch.Size([1]),
                dtype=torch.int64,
                device=self.device,
            ),
            shape=torch.Size([]),
            device=self.device,
        )

        # Action space: Controls colors and frame selection
        action_dim = self.num_frames * (3 + 2 * self.num_frames)
        self.action_spec = BoundedTensorSpec(
            low=-2.0, high=2.0, shape=torch.Size([action_dim]), device=self.device
        )

    def _set_seed(self, seed: Optional[int] = None):
        """Set the seed for this environment's random number generator(s)."""
        if seed is not None:
            self.rng.manual_seed(seed)
        return seed

    def _get_obs(self) -> TensorDict:
        """Return current state as observation"""
        # Flatten frame parameters into a vector
        colors_flat = self.frame_colors.reshape(-1)
        selection_flat = self.frame_selection.reshape(-1)
        obs_vector = torch.cat([colors_flat, selection_flat])

        return TensorDict(
            {
                "observation": obs_vector,
                "step_count": torch.tensor(
                    [self.current_step], dtype=torch.int64, device=self.device
                ),
            },
            batch_size=[],
        )

    def render_frame(
        self,
        frame_idx: int,
        depth: int = 0,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        """Render a frame recursively based on current parameters"""
        # Default dimensions
        if height is None:
            height = self.frame_size[0]
        if width is None:
            width = self.frame_size[1]

        # Base color for this frame
        bg_color = self.frame_colors[frame_idx]

        # If at max depth or too small, return solid color
        if depth >= self.max_depth or width < 2 or height < 2:
            return bg_color.view(3, 1, 1).expand(3, height, width)

        # Get split information
        is_vertical = self.split_directions[frame_idx]
        split_ratio = 0.5  # Fixed split ratio for simplicity

        # Initialize output tensor
        output = torch.zeros(3, height, width, device=self.device)

        # Determine which children to render based on probabilities
        left_idx = torch.argmax(self.frame_selection[frame_idx, 0]).item()
        right_idx = torch.argmax(self.frame_selection[frame_idx, 1]).item()

        # Calculate dimensions based on split direction
        if is_vertical:
            # Vertical split
            left_width = max(1, int(width * split_ratio))
            right_width = max(1, width - left_width)

            # Render left child
            left_child = self.render_frame(left_idx, depth + 1, height, left_width)
            output[:, :, :left_width] = left_child

            # Render right child
            right_child = self.render_frame(right_idx, depth + 1, height, right_width)
            output[:, :, left_width:] = right_child
        else:
            # Horizontal split
            top_height = max(1, int(height * split_ratio))
            bottom_height = max(1, height - top_height)

            # Render top child
            top_child = self.render_frame(left_idx, depth + 1, top_height, width)
            output[:, :top_height, :] = top_child

            # Render bottom child
            bottom_child = self.render_frame(right_idx, depth + 1, bottom_height, width)
            output[:, top_height:, :] = bottom_child

        return output

    def compute_similarity(self, image: torch.Tensor) -> float:
        """Compute similarity between generated image and text prompt using SigLIP"""
        assert self.siglip_model
        assert self.text_features

        # SigLIP normalization constants
        mean = (
            torch.tensor([0.48145466, 0.4578275, 0.40821073])
            .view(1, 3, 1, 1)
            .to(self.device)
        )
        std = (
            torch.tensor([0.26862954, 0.26130258, 0.27577711])
            .view(1, 3, 1, 1)
            .to(self.device)
        )

        # Ensure image has batch dimension and correct size
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Resize to 224x224 for SigLIP
        image_resized = torch.nn.functional.interpolate(
            image, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalize the image
        normalized_image = (image_resized - mean) / std

        # Get image features
        with torch.no_grad():
            image_features = self.siglip_model.get_image_features(
                pixel_values=normalized_image
            )
            image_features = torch.nn.functional.normalize(image_features, dim=1)

        # Calculate similarity
        similarity = torch.matmul(image_features, self.text_features.t())[0][0]

        # Apply scaling and bias
        logit_scale = self.siglip_model.logit_scale.exp()
        logit_bias = self.siglip_model.logit_bias
        scaled_similarity = similarity * logit_scale + logit_bias

        # Convert to probability with sigmoid
        similarity_prob = torch.sigmoid(scaled_similarity)

        return similarity_prob.item()

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Take a step in the environment based on the action"""
        # Extract and apply action
        action = tensordict.get("action")

        # Process action to update frame parameters
        action = action.squeeze()
        color_values = action[: self.num_frames * 3].reshape(self.num_frames, 3)
        selection_values = action[self.num_frames * 3 :].reshape(
            self.num_frames, 2, self.num_frames
        )

        # Update colors with sigmoid to keep in [0,1]
        self.frame_colors = torch.sigmoid(color_values)

        # Update frame selection with softmax
        for i in range(self.num_frames):
            self.frame_selection[i, 0] = torch.softmax(selection_values[i, 0], dim=0)
            self.frame_selection[i, 1] = torch.softmax(selection_values[i, 1], dim=0)

        # Generate the image and compute reward
        image = self.render_frame(0)
        reward = self.compute_similarity(image)

        # Increment step counter
        self.current_step += 1

        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        done = terminated or truncated

        # Create output tensordict with both current and next state
        next_td = TensorDict(
            {
                **self._get_obs(),
                "reward": torch.tensor([reward], device=self.device),
                "terminated": torch.tensor([terminated], device=self.device),
                "truncated": torch.tensor([truncated], device=self.device),
                "done": torch.tensor([done], device=self.device),
                "image": image,  # Include image for visualization
            },
            batch_size=[],
        )

        result_td = TensorDict(
            {
                "next": next_td,
                **self._get_obs(),
                "action": action,
                "terminated": torch.tensor([terminated], device=self.device),
                "truncated": torch.tensor([truncated], device=self.device),
                "done": torch.tensor([done], device=self.device),
                "image": image,
            },
            batch_size=[],
        )

        return result_td

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:
        """Reset the environment to start a new episode"""
        self.current_step = 0

        # Randomly initialize colors for exploration
        self.frame_colors = torch.rand(
            self.num_frames, 3, device=self.device, generator=self.rng
        )

        # Initialize frame selection with some randomization
        for i in range(self.num_frames):
            self.frame_selection[i, 0] = torch.softmax(
                torch.randn(self.num_frames, device=self.device, generator=self.rng),
                dim=0,
            )
            self.frame_selection[i, 1] = torch.softmax(
                torch.randn(self.num_frames, device=self.device, generator=self.rng),
                dim=0,
            )

        # Generate initial image
        image = self.render_frame(0)

        # Return initial observation
        return TensorDict(
            {
                **self._get_obs(),
                "image": image,
                "terminated": torch.tensor([False], device=self.device),
                "truncated": torch.tensor([False], device=self.device),
                "done": torch.tensor([False], device=self.device),
            },
            batch_size=[],
        )


def create_policy_and_value_networks(env, hidden_size=256):
    """
    Create policy and value networks for PPO training

    Args:
        env: FractalFrameEnv instance
        hidden_size: Size of hidden layers

    Returns:
        policy_module: ProbabilisticActor policy
        value_module: ValueOperator for critic
    """
    # Create the policy network (actor)
    actor_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 2 * env.action_spec.shape[0]),
        NormalParamExtractor(),
    )

    # Wrap in TensorDictModule
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    # Create a ProbabilisticActor that will convert loc and scale to an action distribution
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )

    # Create the value network (critic)
    value_net = nn.Sequential(
        nn.Linear(env.observation_spec["observation"].shape[0], hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    )

    # Wrap in ValueOperator
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return policy_module, value_module


def train_with_ppo(
    env,
    policy_module,
    value_module,
    total_frames=10000,
    frames_per_batch=1000,
    num_epochs=10,
    lr=3e-4,
    gamma=0.99,
    lmbda=0.95,
    clip_epsilon=0.2,
    entropy_eps=1e-4,
    device="cuda",
):
    """
    Train the fractal frame generator using PPO algorithm

    PPO (Proximal Policy Optimization) is a policy gradient method that:
    1. Collects batches of experience using the current policy
    2. Computes "advantage" estimates (how much better/worse actions were than expected)
    3. Updates the policy multiple times on the same batch, but with a clipping mechanism
       that prevents too large policy changes

    Args:
        env: The FractalFrameEnv
        policy_module: The policy network (actor)
        value_module: The value network (critic)
        total_frames: Total frames to collect during training
        frames_per_batch: Number of frames to collect before each optimization
        num_epochs: Number of optimization passes over each batch
        lr: Learning rate
        gamma: Discount factor for future rewards
        lmbda: GAE lambda parameter (controls bias/variance tradeoff)
        clip_epsilon: PPO clipping parameter
        entropy_eps: Entropy bonus coefficient
        device: Device to use

    Returns:
        Dict with training results
    """
    print("Starting PPO training...")

    # Set up the advantage module (GAE - Generalized Advantage Estimation)
    # GAE helps compute a more stable estimate of the advantage by blending
    # TD(0) and TD(∞) estimates based on the lambda parameter
    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True,
    )

    # Create the PPO loss module
    # This handles both the policy loss (with clipping) and the value loss
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=lr,
    )

    # Create scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        total_frames // frames_per_batch,
        0.0,
    )

    # Set up data collector
    # This handles collecting experience by running the policy in the environment
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    # Create replay buffer for sampling during optimization
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # Training loop variables
    logs = {"reward": [], "eval_reward": [], "images": []}
    best_reward = -float("inf")
    best_image = None

    # Main training loop with progress bar
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # Iterate through batches of collected data
    for i, batch_data in enumerate(collector):
        # For each batch, we run multiple epochs of optimization
        for _ in range(num_epochs):
            # Compute advantages for the whole batch
            # This updates the batch_data with advantage and value_target
            advantage_module(batch_data)

            # Reshape data for the replay buffer
            data_view = batch_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            # Sub-batch size for optimization steps
            sub_batch_size = min(64, frames_per_batch)

            # Multiple optimization steps per epoch
            for _ in range(frames_per_batch // sub_batch_size):
                # Sample a sub-batch without replacement
                sub_batch = replay_buffer.sample(sub_batch_size)

                # Compute losses
                loss_vals = loss_module(sub_batch.to(device))

                # Combined loss
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization step
                optimizer.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optimizer.step()

        # Record metrics
        avg_reward = batch_data["next", "reward"].mean().item()
        logs["reward"].append(avg_reward)

        # Update progress bar
        pbar.update(batch_data.numel())
        pbar.set_description(f"Reward: {avg_reward:.4f}")

        # Periodically evaluate with deterministic policy
        if i % 5 == 0:
            # Evaluate the current policy without exploration
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_td = env.reset()

                # Run one full episode
                done = False
                episode_reward = 0

                while not done:
                    # Get action from policy
                    action_td = policy_module(eval_td)

                    # Step the environment
                    next_td = env.step(action_td)

                    # Accumulate reward
                    episode_reward += next_td["next", "reward"].item()

                    # Save image for visualization
                    if len(logs["images"]) < 10:  # Limit number of images saved
                        logs["images"].append(next_td["image"].detach().cpu())

                    # Check if done
                    done = next_td["next", "done"].item()
                    eval_td = next_td

            # Record evaluation metrics
            logs["eval_reward"].append(episode_reward)

            # Track best image
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_image = eval_td["image"].detach().cpu()

            # Update progress bar description
            eval_str = f"Eval reward: {episode_reward:.4f}, Best: {best_reward:.4f}"
            pbar.set_description(f"{eval_str} | Train reward: {avg_reward:.4f}")

        # Step the learning rate scheduler
        scheduler.step()

    pbar.close()

    # Return training results
    return {
        "reward_history": logs["reward"],
        "eval_reward_history": logs["eval_reward"],
        "best_reward": best_reward,
        "best_image": best_image,
        "sample_images": logs["images"],
        "policy": policy_module,
        "value": value_module,
        "env": env,
    }


def visualize_results(results):
    """Visualize training results"""
    # Plot rewards
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(results["reward_history"])
    plt.title("Training Rewards")
    plt.xlabel("Batch")
    plt.ylabel("Average Reward")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(results["eval_reward_history"])
    plt.title("Evaluation Rewards")
    plt.xlabel("Evaluation")
    plt.ylabel("Episode Reward")
    plt.grid(True)

    # Show best image
    plt.subplot(1, 3, 3)
    plt.imshow(results["best_image"].permute(1, 2, 0).numpy())
    plt.title(f"Best Image (reward: {results['best_reward']:.4f})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Show progression of generated images
    if len(results["sample_images"]) > 0:
        n_images = min(5, len(results["sample_images"]))
        plt.figure(figsize=(15, 3))
        for i in range(n_images):
            plt.subplot(1, n_images, i + 1)
            img_idx = i * len(results["sample_images"]) // n_images
            plt.imshow(results["sample_images"][img_idx].permute(1, 2, 0).numpy())
            plt.title(f"Sample {img_idx}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


def optimize_fractal_frames_with_ppo(
    siglip_model,
    siglip_processor,
    prompt: str,
    num_frames: int = 10,
    frame_size: tuple[int, int] = (128, 128),
    max_depth: int = 5,
    total_frames: int = 10000,
    device: str = "cuda",
):
    """
    Main function to optimize fractal frames with PPO

    Args:
        siglip_model: SigLIP model for computing similarity
        siglip_processor: SigLIP processor for text inputs
        prompt: Text prompt to optimize towards
        num_frames: Number of frames in the fractal
        frame_size: Size of the frames
        max_depth: Maximum recursion depth for rendering
        total_frames: Total frames to collect during training
        device: Device to use

    Returns:
        Dict with training results
    """
    print(f"Creating environment for prompt: '{prompt}'")

    # Create the environment
    env = FractalFrameEnv(
        num_frames=num_frames,
        frame_size=frame_size,
        max_depth=max_depth,
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
        prompt=prompt,
        device=device,
    )

    # Create policy and value networks
    policy_module, value_module = create_policy_and_value_networks(env)

    # Train with PPO
    results = train_with_ppo(
        env=env,
        policy_module=policy_module,
        value_module=value_module,
        total_frames=total_frames,
        device=device,
    )

    # Visualize results
    visualize_results(results)

    # Export HTML files
    html_files = export_html_files(env)
    results["html_files"] = html_files

    return results


def export_html_files(env):
    """Export the optimized frames as HTML files"""
    html_files = []

    # Generate HTML for each frame
    for i in range(env.num_frames):
        is_vertical = env.split_directions[i]
        split_ratio = 50  # Fixed at 50%

        # Get the background color
        color = env.frame_colors[i].detach().cpu()
        r, g, b = [int(c * 255) for c in color]

        # Get the child frame indices
        left_idx = torch.argmax(env.frame_selection[i, 0]).item()
        right_idx = torch.argmax(env.frame_selection[i, 1]).item()

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


# Example usage:
"""
from transformers import AutoModel, AutoProcessor

# Initialize SigLIP model
ckpt = "google/siglip2-base-patch16-224"
siglip_model = AutoModel.from_pretrained(ckpt).eval().to("cuda")
siglip_processor = AutoProcessor.from_pretrained(ckpt)

# Run optimization
results = optimize_fractal_frames_with_ppo(
    siglip_model=siglip_model,
    siglip_processor=siglip_processor,
    prompt="a beautiful flower",
    num_frames=10,
    frame_size=(128, 128),
    max_depth=5,
    total_frames=10000,
    device="cuda",
)

# Access the optimized HTML files
html_files = results["html_files"]
"""
