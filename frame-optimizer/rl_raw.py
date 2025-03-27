import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random


class PolicyNetwork(nn.Module):
    """Policy network for selecting frames and split parameters"""
    def __init__(self, num_frames: int, hidden_size: int = 128):
        super().__init__()
        self.num_frames = num_frames

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(num_frames, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy heads for frame selection (left and right child for each frame)
        self.frame_selection_left = nn.Linear(hidden_size, num_frames)
        self.frame_selection_right = nn.Linear(hidden_size, num_frames)

        # Color policy (for each frame's RGB values)
        self.color_policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_frames * 3),
            nn.Sigmoid()  # Colors in [0,1] range
        )

    def forward(self, state):
        """
        Forward pass to generate action distributions

        Args:
            state: Current state of the fractal generator (one-hot encoding)

        Returns:
            Distributions for frame selections and colors
        """
        features = self.shared(state)

        # Get frame selection logits
        left_logits = self.frame_selection_left(features)
        right_logits = self.frame_selection_right(features)

        # Get frame colors
        colors = self.color_policy(features).view(-1, self.num_frames, 3)

        return {
            'left_logits': left_logits,
            'right_logits': right_logits,
            'colors': colors
        }

    def get_action(self, state, temperature=1.0):
        """Sample actions from the policy"""
        outputs = self.forward(state)

        # Apply temperature and sample
        left_probs = F.softmax(outputs['left_logits'] / temperature, dim=-1)
        right_probs = F.softmax(outputs['right_logits'] / temperature, dim=-1)

        # Sample actions
        left_actions = torch.multinomial(left_probs, 1)
        right_actions = torch.multinomial(right_probs, 1)
        colors = outputs['colors']

        # Calculate log probabilities
        left_log_probs = F.log_softmax(outputs['left_logits'], dim=-1)
        right_log_probs = F.log_softmax(outputs['right_logits'], dim=-1)

        # Get the log probs of the sampled actions
        left_log_prob = left_log_probs.gather(1, left_actions)
        right_log_prob = right_log_probs.gather(1, right_actions)

        return {
            'left_actions': left_actions,
            'right_actions': right_actions,
            'colors': colors,
            'left_log_probs': left_log_prob,
            'right_log_probs': right_log_prob,
            'left_probs': left_probs,
            'right_probs': right_probs
        }


class ValueNetwork(nn.Module):
    """Value network to estimate how good a state is"""
    def __init__(self, num_frames: int, hidden_size: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_frames, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.model(state)


class RLFractalFrameGenerator(nn.Module):
    """FractalFrameGenerator optimized with reinforcement learning"""
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

        # Initialize with random parameters for exploration
        self.frame_colors = torch.rand(num_frames, 3, device=device)

        # Frame selection as categorical distributions
        self.frame_selection = torch.zeros(num_frames, 2, num_frames, device=device)

        # Fixed split direction (alternating horizontal/vertical)
        self.split_directions = [i % 2 for i in range(num_frames)]

        # Fixed split ratios (50/50 splits)
        self.split_ratios = torch.ones(num_frames, device=device) * 0.5

    def update_from_actions(self, actions):
        """Update generator parameters from policy actions"""
        # Update frame colors
        self.frame_colors = actions['colors'].squeeze(0)

        # Create selection matrices based on sampled actions
        left_actions = actions['left_actions'].squeeze(0)
        right_actions = actions['right_actions'].squeeze(0)

        # Create new frame selection tensors
        new_frame_selection = torch.zeros_like(self.frame_selection)

        # Update each frame's child selections
        for i in range(self.num_frames):
            new_frame_selection[i, 0, left_actions[i]] = 1.0
            new_frame_selection[i, 1, right_actions[i]] = 1.0

        self.frame_selection = new_frame_selection

    def get_state_representation(self):
        """Get a flat state representation for the policy network"""
        # Simple one-hot encoding of the current state
        return torch.ones(1, self.num_frames, device=self.device)

    def render_frame(
        self,
        frame_idx: int,
        depth: int = 0,
        start_y: int = 0,
        start_x: int = 0,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        """Render a frame based on current parameters"""
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
        split_ratio = self.split_ratios[frame_idx].item()

        # Initialize output tensor
        output = torch.zeros(3, height, width, device=self.device)

        # Determine which children to render based on frame_selection
        left_idx = torch.argmax(self.frame_selection[frame_idx, 0]).item()
        right_idx = torch.argmax(self.frame_selection[frame_idx, 1]).item()

        # Calculate dimensions based on split direction
        if is_vertical:
            # Vertical split
            left_width = max(1, int(width * split_ratio))
            right_width = max(1, width - left_width)

            # Render left child
            left_child = self.render_frame(
                left_idx, depth + 1, start_y, start_x,
                height, left_width
            )
            output[:, :, :left_width] = left_child

            # Render right child
            right_child = self.render_frame(
                right_idx, depth + 1, start_y, start_x + left_width,
                height, right_width
            )
            output[:, :, left_width:] = right_child
        else:
            # Horizontal split
            top_height = max(1, int(height * split_ratio))
            bottom_height = max(1, height - top_height)

            # Render top child
            top_child = self.render_frame(
                left_idx, depth + 1, start_y, start_x,
                top_height, width
            )
            output[:, :top_height, :] = top_child

            # Render bottom child
            bottom_child = self.render_frame(
                right_idx, depth + 1, start_y + top_height, start_x,
                bottom_height, width
            )
            output[:, top_height:, :] = bottom_child

        return output

    def forward(self) -> torch.Tensor:
        """Generate the full fractal image"""
        h, w = self.frame_size
        return self.render_frame(0, 0, 0, 0, h, w)

    def export_html(self) -> list[tuple[str, str]]:
        """Export the optimized frames as HTML files"""
        html_files = []

        # Generate HTML for each frame
        for i in range(self.num_frames):
            is_vertical = self.split_directions[i]
            split_ratio = self.split_ratios[i].item() * 100  # Convert to percentage

            # Get the background color
            color = self.frame_colors[i].detach().cpu()
            r, g, b = [int(c * 255) for c in color]

            # Get the child frame indices
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


class ExperienceBuffer:
    """Simple experience buffer for PPO"""
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


def compute_siglip_similarity(
    image: torch.Tensor,
    text_features: torch.Tensor,
    siglip_model,
    device: str,
    image_size: tuple[int, int] = (224, 224)
):
    """Compute similarity between an image and text using SigLIP"""
    # SigLIP normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    # Ensure image has batch dimension and correct size
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Resize if needed
    if image.shape[2:] != image_size:
        image = F.interpolate(
            image,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )

    # Normalize the image
    normalized_image = (image - mean) / std

    # Get image features
    with torch.no_grad():
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

    return similarity_prob.item()


def collect_experiences(
    policy_net: PolicyNetwork,
    generator: RLFractalFrameGenerator,
    siglip_model,
    text_features: torch.Tensor,
    episodes: int,
    temperature: float,
    device: str
) -> tuple[list[dict], float, torch.Tensor]:
    """
    Collect experiences by running the current policy on the fractal generator.

    In PPO, we first collect a batch of experiences (state, action, reward) by executing
    the current policy, before making any policy updates.

    Args:
        policy_net: The current policy network
        generator: The fractal generator to optimize
        siglip_model: SigLIP model for reward computation
        text_features: Precomputed text features for similarity
        episodes: Number of episodes to run
        temperature: Temperature for exploration (higher = more random actions)
        device: Device to use

    Returns:
        experiences: List of experience dictionaries
        best_reward: Best reward achieved in this batch
        best_image: Best image generated in this batch
    """
    experiences = []
    episode_rewards = []
    best_reward = -float('inf')
    best_image = None

    for _ in range(episodes):
        # Get current state
        state = generator.get_state_representation()

        # Sample actions from the policy (higher temperature = more exploration)
        actions = policy_net.get_action(state, temperature=temperature)

        # Update generator with the sampled actions
        generator.update_from_actions(actions)

        # Generate the image
        generated_image = generator()

        # Compute reward (similarity with the text prompt)
        reward = compute_siglip_similarity(
            generated_image,
            text_features,
            siglip_model,
            device
        )

        # Estimate value of this state
        with torch.no_grad():
            value = policy_net.get_value(state).item()

        # Store the experience
        experience = {
            'state': state,
            'actions': actions,
            'reward': reward,
            'value': value
        }
        experiences.append(experience)
        episode_rewards.append(reward)

        # Track best result
        if reward > best_reward:
            best_reward = reward
            best_image = generated_image.detach().clone()

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    return experiences, avg_reward, best_reward, best_image


def compute_advantages(
    experiences: list[dict],
    gamma: float,
    gae_lambda: float,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages and returns using Generalized Advantage Estimation (GAE).

    PPO, like many actor-critic methods, uses advantage estimates to reduce variance.
    The advantage tells us how much better an action was compared to the average action
    in that state. GAE helps create a balance between bias and variance in these estimates.

    Args:
        experiences: List of experiences
        gamma: Discount factor for future rewards (0-1)
        gae_lambda: GAE lambda parameter (0-1)
        device: Computation device

    Returns:
        advantages: Tensor of advantage estimates
        returns: Tensor of estimated returns
    """
    rewards = torch.tensor([exp['reward'] for exp in experiences], device=device)
    values = torch.tensor([exp['value'] for exp in experiences], device=device)

    # Compute advantages using GAE
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    # We process backwards through the episodes
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # For the last step, there is no next value
            nextnonterminal = 0.0
            nextvalues = 0.0
        else:
            nextnonterminal = 1.0  # Since we don't have terminal episodes
            nextvalues = values[t + 1]

        # GAE formula: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]

        # A_t = δ_t + γλA_{t+1}
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    # Returns = advantages + values
    returns = advantages + values

    # Normalize advantages to improve training stability
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def update_policy(
    policy_net: PolicyNetwork,
    policy_optimizer: torch.optim.Optimizer,
    experiences: list[dict],
    advantages: torch.Tensor,
    clip_ratio: float,
    entropy_coef: float,
    num_epochs: int = 4
) -> tuple[float, float]:
    """
    Update the policy network using the PPO algorithm.

    In PPO, we update the policy multiple times using the same batch of experiences.
    To prevent the policy from changing too much (which could lead to instability),
    we use a clipped surrogate objective.

    Args:
        policy_net: The policy network to update
        policy_optimizer: Optimizer for the policy network
        experiences: List of experiences
        advantages: Precomputed advantage estimates
        clip_ratio: Clipping parameter for PPO (typically 0.1-0.3)
        entropy_coef: Coefficient for entropy bonus (encourages exploration)
        num_epochs: Number of optimization epochs

    Returns:
        avg_policy_loss: Average policy loss across epochs
        avg_entropy: Average entropy across epochs
    """
    # Collect states and old log probs from experiences
    states = torch.cat([exp['state'] for exp in experiences])

    old_left_log_probs = torch.cat([exp['actions']['left_log_probs'] for exp in experiences])
    old_right_log_probs = torch.cat([exp['actions']['right_log_probs'] for exp in experiences])

    # Initialize tracking variables
    policy_losses = []
    entropies = []

    # Perform multiple update epochs
    for _ in range(num_epochs):
        # Forward pass through policy network
        policy_outputs = [policy_net(s) for s in states]

        # Get new log probs for the actions we took
        new_left_log_probs = []
        new_right_log_probs = []
        entropy_sum = 0

        for i, exp in enumerate(experiences):
            # Get logits from policy network output
            left_logits = policy_outputs[i]['left_logits']
            right_logits = policy_outputs[i]['right_logits']

            # Convert to probability distributions
            left_log_probs = F.log_softmax(left_logits, dim=-1)
            right_log_probs = F.log_softmax(right_logits, dim=-1)

            # Calculate entropy (higher entropy = more exploration)
            left_probs = F.softmax(left_logits, dim=-1)
            right_probs = F.softmax(right_logits, dim=-1)
            entropy = -(left_probs * left_log_probs).sum() - (right_probs * right_log_probs).sum()
            entropy_sum += entropy

            # Get log probs of the actions we actually took
            left_actions = exp['actions']['left_actions']
            right_actions = exp['actions']['right_actions']

            left_log_prob = left_log_probs.gather(1, left_actions)
            right_log_prob = right_log_probs.gather(1, right_actions)

            new_left_log_probs.append(left_log_prob)
            new_right_log_probs.append(right_log_prob)

        # Concatenate log probs
        new_left_log_probs = torch.cat(new_left_log_probs)
        new_right_log_probs = torch.cat(new_right_log_probs)

        # Calculate probability ratios (π_new / π_old)
        left_ratio = torch.exp(new_left_log_probs - old_left_log_probs)
        right_ratio = torch.exp(new_right_log_probs - old_right_log_probs)

        # Combined ratio (average of left and right)
        ratio = (left_ratio + right_ratio) / 2

        # Key PPO trick 1: Clipped surrogate objective
        # Limits how much the policy can change in one update
        surrogate1 = ratio * advantages  # Unclipped objective
        surrogate2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages  # Clipped objective

        # Take the minimum to create a pessimistic bound
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Key PPO trick 2: Entropy bonus
        # Encourages exploration by rewarding policies with high entropy
        entropy = entropy_sum / len(experiences)
        policy_loss = policy_loss - entropy_coef * entropy

        # Policy gradient step
        policy_optimizer.zero_grad()
        policy_loss.backward()

        # Key PPO trick 3: Gradient clipping
        # Prevents large policy updates that could destabilize training
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        policy_optimizer.step()

        # Track metrics
        policy_losses.append(policy_loss.item())
        entropies.append(entropy.item())

    return sum(policy_losses) / len(policy_losses), sum(entropies) / len(entropies)


def update_value_function(
    value_net: ValueNetwork,
    value_optimizer: torch.optim.Optimizer,
    experiences: list[dict],
    returns: torch.Tensor,
    value_coef: float,
    num_epochs: int = 4
) -> float:
    """
    Update the value network to better predict expected returns.

    In actor-critic methods like PPO, the value function (critic) helps reduce variance
    by providing a baseline for advantage estimation. The better our value estimates,
    the more stable our policy updates will be.

    Args:
        value_net: The value network to update
        value_optimizer: Optimizer for the value network
        experiences: List of experiences
        returns: Target returns for value prediction
        value_coef: Coefficient for value loss
        num_epochs: Number of optimization epochs

    Returns:
        avg_value_loss: Average value loss across epochs
    """
    # Collect states from experiences
    states = torch.cat([exp['state'] for exp in experiences])

    # Initialize tracking
    value_losses = []

    # Perform multiple epochs
    for _ in range(num_epochs):
        # Get value predictions for all states
        predicted_values = torch.cat([value_net(s) for s in states])

        # Value loss is a simple MSE between predicted values and returns
        value_loss = F.mse_loss(predicted_values, returns) * value_coef

        # Gradient step
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        value_optimizer.step()

        # Track value loss
        value_losses.append(value_loss.item())

    return sum(value_losses) / len(value_losses)


def visualize_progress(
    iteration: int,
    num_iterations: int,
    avg_reward: float,
    best_reward: float,
    current_temp: float,
    rewards_history: list[float],
    current_image: torch.Tensor,
    best_image: torch.Tensor
):
    """
    Visualize the current training progress.

    Args:
        iteration: Current iteration
        num_iterations: Total iterations
        avg_reward: Average reward in the last batch
        best_reward: Best reward seen so far
        current_temp: Current temperature value
        rewards_history: History of rewards
        current_image: Currently generated image
        best_image: Best image seen so far
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Current image
    axes[0].imshow(current_image.permute(1, 2, 0).cpu().numpy().clip(0, 1))
    axes[0].set_title(f"Current (temp={current_temp:.2f})")
    axes[0].axis("off")

    # Best image so far
    if best_image is not None:
        axes[1].imshow(best_image.permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[1].set_title(f"Best (reward={best_reward:.4f})")
        axes[1].axis("off")

    # Plot rewards over time
    axes[2].plot(rewards_history)
    axes[2].set_title("Rewards Over Time")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Avg Reward")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Iteration {iteration + 1}/{num_iterations}")
    print(f"  Avg Reward: {avg_reward:.4f}, Best Reward: {best_reward:.4f}")
    print(f"  Temperature: {current_temp:.4f}")


class PolicyValueNetwork(nn.Module):
    """
    Combined policy and value network with shared feature extraction.

    In PPO, we need both:
    1. A policy network (actor) that decides which actions to take
    2. A value network (critic) that estimates how good each state is

    Sharing early layers between these networks is common and efficient.
    """
    def __init__(self, num_frames: int, hidden_size: int = 128):
        super().__init__()
        self.num_frames = num_frames

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(num_frames, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Policy heads for frame selection
        self.frame_selection_left = nn.Linear(hidden_size, num_frames)
        self.frame_selection_right = nn.Linear(hidden_size, num_frames)

        # Color policy head
        self.color_policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_frames * 3),
            nn.Sigmoid()  # Colors in [0,1] range
        )

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Get both policy distributions and value estimate"""
        features = self.shared(state)

        # Get frame selection logits
        left_logits = self.frame_selection_left(features)
        right_logits = self.frame_selection_right(features)

        # Get frame colors
        colors = self.color_policy(features).view(-1, self.num_frames, 3)

        # Get value estimate
        value = self.value_head(features)

        return {
            'left_logits': left_logits,
            'right_logits': right_logits,
            'colors': colors,
            'value': value
        }

    def get_value(self, state):
        """Get only the value estimate for a state"""
        features = self.shared(state)
        return self.value_head(features)

    def get_action(self, state, temperature=1.0):
        """Sample actions from the policy"""
        outputs = self.forward(state)

        # Apply temperature and sample
        left_probs = F.softmax(outputs['left_logits'] / temperature, dim=-1)
        right_probs = F.softmax(outputs['right_logits'] / temperature, dim=-1)

        # Sample actions
        left_actions = torch.multinomial(left_probs, 1)
        right_actions = torch.multinomial(right_probs, 1)
        colors = outputs['colors']

        # Calculate log probabilities
        left_log_probs = F.log_softmax(outputs['left_logits'], dim=-1)
        right_log_probs = F.log_softmax(outputs['right_logits'], dim=-1)

        # Get the log probs of the sampled actions
        left_log_prob = left_log_probs.gather(1, left_actions)
        right_log_prob = right_log_probs.gather(1, right_actions)

        return {
            'left_actions': left_actions,
            'right_actions': right_actions,
            'colors': colors,
            'left_log_probs': left_log_prob,
            'right_log_probs': right_log_prob,
            'left_probs': left_probs,
            'right_probs': right_probs
        }


def optimize_with_ppo(
    generator: RLFractalFrameGenerator,
    siglip_model,
    siglip_processor,
    prompt: str,
    num_iterations: int = 100,
    episodes_per_update: int = 5,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    temperature: float = 1.0,
    temperature_decay: float = 0.99,
    device: str = "cuda",
    log_every: int = 5
):
    """
    Optimize fractal generator with PPO algorithm

    PPO (Proximal Policy Optimization) is a popular RL algorithm that offers:
    1. Stability: Controlled policy updates that don't change too drastically
    2. Sample efficiency: Multiple training epochs with the same experience batch
    3. Balance of exploration and exploitation: Using entropy bonus

    Args:
        generator: The fractal generator to optimize
        siglip_model: Pre-trained SigLIP model for similarity computation
        siglip_processor: SigLIP processor for text encoding
        prompt: Text prompt to optimize towards
        num_iterations: Number of PPO updates
        episodes_per_update: Number of episodes to collect before each update
        learning_rate: Learning rate for policy and value networks
        gamma: Discount factor for future rewards (closer to 1 = care more about future)
        gae_lambda: Lambda parameter for GAE (closer to 1 = care more about true returns)
        clip_ratio: PPO clipping parameter (larger = allow bigger policy changes)
        value_coef: Value loss coefficient (larger = focus more on value accuracy)
        entropy_coef: Entropy bonus coefficient (larger = encourage more exploration)
        temperature: Initial temperature for exploration
        temperature_decay: Temperature decay rate
        device: Device to use
        log_every: How often to log and visualize progress
    """
    # Set models to appropriate modes
    siglip_model.eval()
    generator.train()

    # Process the text prompt once and get text features
    text_inputs = siglip_processor(
        text=[prompt], padding="max_length", max_length=64, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_features = siglip_model.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=1)

    # Initialize combined policy and value network
    policy_net = PolicyValueNetwork(generator.num_frames).to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Tracking variables
    best_reward = -float('inf')
    best_image = None
    rewards_history = []
    current_temp = temperature

    for iteration in range(num_iterations):
        # Step 1: Collect experiences with the current policy
        experiences, avg_reward, iteration_best_reward, iteration_best_image = collect_experiences(
            policy_net=policy_net,
            generator=generator,
            siglip_model=siglip_model,
            text_features=text_features,
            episodes=episodes_per_update,
            temperature=current_temp,
            device=device
        )

        # Track rewards
        rewards_history.append(avg_reward)

        # Update best result if we found a better one
        if iteration_best_reward > best_reward:
            best_reward = iteration_best_reward
            best_image = iteration_best_image

        # Step 2: Compute advantages and returns
        advantages, returns = compute_advantages(
            experiences=experiences,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device
        )

        # Step 3: Update policy
        avg_policy_loss, avg_entropy = update_policy(
            policy_net=policy_net,
            policy_optimizer=optimizer,
            experiences=experiences,
            advantages=advantages,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef
        )

        # Step 4: Update value function
        avg_value_loss = update_value_function(
            value_net=policy_net,  # We use the value head of the combined network
            value_optimizer=optimizer,
            experiences=experiences,
            returns=returns,
            value_coef=value_coef
        )

        # Decay temperature for less exploration over time
        current_temp *= temperature_decay

        # Step 5: Visualization and logging
        if iteration % log_every == 0 or iteration == num_iterations - 1:
            # Sample new actions with current policy
            state = generator.get_state_representation()
            actions = policy_net.get_action(state, temperature=current_temp)
            generator.update_from_actions(actions)
            current_image = generator()

            visualize_progress(
                iteration=iteration,
                num_iterations=num_iterations,
                avg_reward=avg_reward,
                best_reward=best_reward,
                current_temp=current_temp,
                rewards_history=rewards_history,
                current_image=current_image,
                best_image=best_image
            )

    # Generate final result
    state = generator.get_state_representation()
    actions = policy_net.get_action(state, temperature=0.5)  # Lower temperature for deterministic output
    generator.update_from_actions(actions)
    final_image = generator()

    return {
        'best_image': best_image,
        'final_image': final_image,
        'best_reward': best_reward,
        'reward_history': rewards_history,
        'html_files': generator.export_html(),
        'policy_net': policy_net
    }


# Example usage:
"""
# Initialize models
ckpt = "google/siglip2-base-patch16-224"
siglip_model = AutoModel.from_pretrained(ckpt).eval().to("cuda")
siglip_processor = AutoProcessor.from_pretrained(ckpt)

# Create RL-based fractal generator
num_frames = 10
frame_size = (256, 256)
generator = RLFractalFrameGenerator(num_frames, frame_size, device="cuda")

# Optimize with PPO
result = optimize_with_ppo(
    generator=generator,
    siglip_model=siglip_model,
    siglip_processor=siglip_processor,
    prompt="a beautiful flower",
    num_iterations=50,
    episodes_per_update=10,
    learning_rate=3e-4,
    temperature=2.0,
    temperature_decay=0.95,
    device="cuda"
)

# Export the HTML files
html_files = result['html_files']
"""
