import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import math
from dataclasses import dataclass

# Environment wrappers (copied from your training code)
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward):
        return np.sign(float(reward))

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

# Network Architecture
class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = env.action_space.n
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = 3136

        self.value_head = nn.Sequential(NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms))
        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms * self.n_actions)
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

def make_env(env_id, seed=1, capture_video=True, video_folder="videos"):
    def thunk():
        if capture_video:
            env = gym.make(env_id, obs_type="grayscale", render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder)
        else:
            env = gym.make(env_id, obs_type="grayscale")
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 4)
        
        env.action_space.seed(seed)
        return env
    return thunk

def evaluate_model(
    model_path: str,
    env_id: str = "ALE/Pong-v5",
    n_episodes: int = 5,
    n_atoms: int = 51,
    v_min: float = -10,
    v_max: float = 10,
    capture_video: bool = True,
    video_folder: str = "evaluation_videos",
    epsilon: float = 0.0,  # Set to 0 for purely greedy actions
    device: str = "auto"
):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to the saved model (.cleanrl_model file)
        env_id: Environment ID (should match training environment)
        n_episodes: Number of episodes to run
        n_atoms: Number of atoms for distributional DQN (should match training)
        v_min: Minimum value for distributional DQN (should match training)
        v_max: Maximum value for distributional DQN (should match training)
        capture_video: Whether to record videos
        video_folder: Folder to save videos
        epsilon: Epsilon for epsilon-greedy action selection (0 = purely greedy)
        device: Device to run on ('auto', 'cpu', 'cuda')
    """
    
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes to run: {n_episodes}")
    
    # Create environment
    os.makedirs(video_folder, exist_ok=True)
    env = make_env(env_id, capture_video=capture_video, video_folder=video_folder)()
    
    # Load model
    model = NoisyDuelingDistributionalNetwork(env, n_atoms, v_min, v_max).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode (important for noisy networks)
    
    print("Model loaded successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")
    
    # Run episodes
    episodic_returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while True:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get action
            if random.random() < epsilon:
                action = env.action_space.sample()
                print(f"Random action: {action}")
            else:
                with torch.no_grad():
                    q_dist = model(obs_tensor)
                    # Compute Q-values from distribution
                    q_values = (q_dist * model.support).sum(dim=2)
                    action = q_values.argmax().item()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            
            # Print some info
            if episode_length % 100 == 0:
                print(f"Step {episode_length}, Reward: {episode_return:.2f}")
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        episodic_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Return: {episode_return:.2f}")
        print(f"  Length: {episode_length}")
        
        if "episode" in info:
            print(f"  Episode info: {info['episode']}")
    
    env.close()
    
    # Print summary
    print(f"\n--- Evaluation Summary ---")
    print(f"Episodes: {n_episodes}")
    print(f"Average Return: {np.mean(episodic_returns):.2f} ± {np.std(episodic_returns):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min Return: {np.min(episodic_returns):.2f}")
    print(f"Max Return: {np.max(episodic_returns):.2f}")
    
    if capture_video:
        print(f"\nVideos saved in: {video_folder}")
    
    return episodic_returns, episode_lengths

if __name__ == "__main__":
    # Configuration - Update these paths and parameters as needed
    MODEL_PATH = "notebook_run_RAINBOW.cleanrl_model"  # Update this path
    ENV_ID = "ALE/Pong-v5"  # Should match your training environment
    N_EPISODES = 3
    
    # Run evaluation
    returns, lengths = evaluate_model(
        model_path=MODEL_PATH,
        env_id=ENV_ID,
        n_episodes=N_EPISODES,
        capture_video=True,
        video_folder="evaluation_videos",
        epsilon=0.0  # Purely greedy actions
    )