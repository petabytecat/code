# runner.py
import os
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from typing import SupportsFloat
import ale_py

# Custom Wrappers (copied from training code)
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

# QNetwork Definition (same as training)
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),  # Updated line
        )

    def forward(self, x):
        return self.network(x / 255.0)

# Environment Setup with Visualization
def make_env(env_id: str, seed: int) -> Callable:
    """Create environment with rendering and same preprocessing as training"""
    def thunk():
        env = gym.make(env_id, render_mode="human", obs_type="grayscale")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 4)  # Updated line
        env.action_space.seed(seed)
        return env
    return thunk

def run_model(model_path: str, env_id: str = "ALE/Breakout-v5", seed: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = make_env(env_id, seed)()
    
    # Load trained model
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()
    
    # Run with visualization
    obs, _ = env.reset()
    while True:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs.__array__()).unsqueeze(0).to(device)
            q_values = q_network(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to trained model (e.g., runs/BreakoutNoFrameskip-v4__.../exp_name.cleanrl_model)")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5", 
                        help="Environment ID (default: BreakoutNoFrameskip-v4)")
    parser.add_argument("--seed", type=int, default=1, 
                        help="Random seed (default: 1)")
    args = parser.parse_args()
    
    run_model(
        model_path=args.model_path,
        env_id=args.env_id,
        seed=args.seed
    )