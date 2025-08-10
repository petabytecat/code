import os
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections.abc import Callable
from typing import SupportsFloat
import ale_py

# Custom Wrappers (same as your training code)
class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward: SupportsFloat) -> float:
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
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = random.randint(1, self.noop_max + 1)
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, _ = self.env.reset(**kwargs)
        return obs, {}

# EXACT SAME NETWORK ARCHITECTURE AS YOUR TRAINING CODE
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

# EXACT SAME NETWORK AS YOUR TRAINING CODE
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

# Environment Setup
def make_env(env_id: str, seed: int) -> Callable:
    def thunk():
        env = gym.make(env_id, render_mode="human")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk

def run_model(model_path: str, env_id: str = "ALE/Pong-v5", seed: int = 1, 
              n_atoms: int = 51, v_min: float = -10, v_max: float = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env(env_id, seed)()
    
    # Create model (same as training)
    q_network = NoisyDuelingDistributionalNetwork(env, n_atoms, v_min, v_max).to(device)
    
    # Load model
    try:
        state_dict = torch.load(model_path, map_location=device)
        q_network.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    q_network.eval()  # Set to evaluation mode
    
    print(f"Environment: {env_id}")
    print("Starting game... Press Ctrl+C to stop")
    
    # Run episodes
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_count = 0
    
    try:
        while True:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs.__array__()).unsqueeze(0).to(device)
            
            # Get action from model
            with torch.no_grad():
                # Don't reset noise during evaluation for deterministic behavior
                q_dist = q_network(obs_tensor)
                q_values = torch.sum(q_dist * q_network.support, dim=2)
                action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} finished with reward: {episode_reward}")
                episode_reward = 0
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        env.close()

if __name__ == "__main__":
    # Example usage - update the model path to your actual file
    model_path = "notebook_run_bestSWEEP.cleanrl_model"  # UPDATE THIS
    
    run_model(
        model_path=model_path,
        env_id="ALE/Pong-v5",  # Match your training environment
        seed=42,
        n_atoms=51,
        v_min=-10,
        v_max=10
    )