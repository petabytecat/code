import torch
import numpy as np
from collections import deque
import gymnasium as gym
import cv2
import time
import sys  # Added for sys.exit

# ===== DEVICE CONFIGURATION =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== PREPROCESS FUNCTION =====
def preprocess(obs):
    """Convert RGB to grayscale, crop, and resize."""
    gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
    cropped = gray[95:195]  # Crop height
    resized = cv2.resize(cropped, dsize=(100, 50), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)

# ===== DQN MODEL DEFINITION =====
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 11 * 4, 256)
        self.fc2 = nn.Linear(256, actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ===== PLAY FUNCTION =====
def play_episodes(model_path, env_id="ALE/Gopher-v5", num_episodes=3, max_steps=1000, delay=0.02):  # Changed env_id
    """Play episodes using a trained model"""
    try:
        env = gym.make(env_id, render_mode="human")
    except gym.error.Error as e:
        print(f"Failed to create environment: {e}")
        print("Ensure you've installed required packages:")
        print("  pip install gymnasium[atari] gymnasium[accept-rom-license]")
        sys.exit(1)
    
    model = DQN(env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for episode in range(num_episodes):
        obs, info = env.reset()
        frame = preprocess(obs)

        frame_stack = deque(maxlen=4)
        for _ in range(4):
            frame_stack.append(frame)
        state = np.stack(frame_stack, axis=0)

        total_reward = 0

        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.argmax().item()

            next_obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            env.render()
            time.sleep(delay)

            next_frame = preprocess(next_obs)
            frame_stack.append(next_frame)
            next_state = np.stack(frame_stack, axis=0)

            if done or truncated:
                break
            state = next_state

        print(f"Episode {episode+1}: Total reward: {total_reward}, Steps: {step+1}")

    env.close()

# ===== EXECUTE PLAYBACK =====
if __name__ == "__main__":
    print("Playing with best model...")
    play_episodes("dqn_gopher_best.pth")