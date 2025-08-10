from __future__ import annotations

import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
import sys
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import gc
import tyro

import os
import random
import sys
import time
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple, SupportsFloat, List, Callable

import numpy as np
from gymnasium import spaces
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import ale_py

import argparse
from pathlib import Path
from pprint import pformat
from tenacity import retry, stop_after_attempt, wait_fixed
import random
import math
import collections
from collections import deque
import gc

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = False, # capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

# hugging face

HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME = "replay.mp4"
HUGGINGFACE_README_FILE_NAME = "README.md"


@retry(stop=stop_after_attempt(10), wait=wait_fixed(3))
def push_to_hub(
    args: argparse.Namespace,
    episodic_returns: List,
    repo_id: str,
    algo_name: str,
    folder_path: str,
    video_folder_path: str = "",
    revision: str = "main",
    create_pr: bool = False,
    private: bool = False,
):
    # Step 1: lazy import and create / read a huggingface repo
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        private=private,
    )
    # parse the default entity
    entity, repo = repo_url.split("/")[-2:]
    repo_id = f"{entity}/{repo}"

    # Step 2: clean up data
    # delete previous tfevents and mp4 files
    operations = [
        CommitOperationDelete(path_in_repo=file)
        for file in api.list_repo_files(repo_id=repo_id)
        if ".tfevents" in file or file.endswith(".mp4")
    ]

    # Step 3: Generate the model card
    algorithm_variant_filename = sys.argv[0].split("/")[-1]
    model_card = f"""
# (CleanRL) **{algo_name}** Agent Playing **{args.env_id}**

This is a trained model of a {algo_name} agent playing {args.env_id}.
The model was trained by using [CleanRL](https://github.com/vwxyzjn/cleanrl) and the most up-to-date training code can be
found [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/{args.exp_name}.py).

## Get Started

To use this model, please install the `cleanrl` package with the following command:

```
pip install "cleanrl[{args.exp_name}]"
python -m cleanrl_utils.enjoy --exp-name {args.exp_name} --env-id {args.env_id}
```

Please refer to the [documentation](https://docs.cleanrl.dev/get-started/zoo/) for more detail.


## Command to reproduce the training

```bash
curl -OL https://huggingface.co/{repo_id}/raw/main/{algorithm_variant_filename}
curl -OL https://huggingface.co/{repo_id}/raw/main/pyproject.toml
curl -OL https://huggingface.co/{repo_id}/raw/main/poetry.lock
poetry install --all-extras
python {algorithm_variant_filename} {" ".join(sys.argv[1:])}
```

# Hyperparameters
```python
{pformat(vars(args))}
```
    """
    readme_path = Path(folder_path) / HUGGINGFACE_README_FILE_NAME
    readme = model_card

    # metadata
    metadata = {}
    metadata["tags"] = [
        args.env_id,
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
    ]
    metadata["library_name"] = "cleanrl"
    eval = metadata_eval_result(
        model_pretty_name=algo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{np.average(episodic_returns):.2f} +/- {np.std(episodic_returns):.2f}",
        dataset_pretty_name=args.env_id,
        dataset_id=args.env_id,
    )
    metadata = {**metadata, **eval}

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    metadata_save(readme_path, metadata)

    # fetch mp4 files
    if video_folder_path:
        # Push all video files
        video_files = list(Path(video_folder_path).glob("*.mp4"))
        operations += [CommitOperationAdd(path_or_fileobj=str(file), path_in_repo=str(file)) for file in video_files]
        # Push latest one in root directory
        latest_file = max(video_files, key=lambda file: int("".join(filter(str.isdigit, file.stem))))
        operations.append(
            CommitOperationAdd(path_or_fileobj=str(latest_file), path_in_repo=HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME)
        )

    # fetch folder files
    operations += [
        CommitOperationAdd(path_or_fileobj=str(item), path_in_repo=str(item.relative_to(folder_path)))
        for item in Path(folder_path).glob("*")
    ]

    # fetch source code
    operations.append(CommitOperationAdd(path_or_fileobj=sys.argv[0], path_in_repo=sys.argv[0].split("/")[-1]))

    # upload poetry files at the root of the repository
    git_root = Path(__file__).parent.parent
    operations.append(CommitOperationAdd(path_or_fileobj=str(git_root / "pyproject.toml"), path_in_repo="pyproject.toml"))
    operations.append(CommitOperationAdd(path_or_fileobj=str(git_root / "poetry.lock"), path_in_repo="poetry.lock"))

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="pushing model",
        revision=revision,
        create_pr=create_pr,
    )
    print(f"Model pushed to {repo_url}")
    return repo_url

# from cleanrl_utils.buffers import ReplayBuffer

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


def get_device(device: torch.device | str = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")



class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples | RolloutBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))


class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info

class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
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
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: dict = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info




def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, obs_type="grayscale") # env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, obs_type="grayscale") # env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        # env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4) # env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1): # CHANGE std_init FROM 0.5 to 0.1
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
        # factorized gaussian noise
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


# ALGO LOGIC: initialize agent here:
class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = env.single_action_space.n
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


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def total(self):
        return self.tree[0]

    def retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


# adapted from: https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.full(self.tree_size, float("inf"), dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def min(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, device, n_step, gamma, alpha=0.6, beta=0.4, eps=1e-6):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.buffer_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_next_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.buffer_actions = np.zeros(capacity, dtype=np.int64)
        self.buffer_rewards = np.zeros(capacity, dtype=np.float32)
        self.buffer_dones = np.zeros(capacity, dtype=np.bool_)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

        # For n-step returns
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        reward = 0.0
        next_obs = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        for i in range(len(self.n_step_buffer)):
            reward += self.gamma**i * self.n_step_buffer[i][2]
            if self.n_step_buffer[i][4]:
                next_obs = self.n_step_buffer[i][3]
                done = True
                break
        return reward, next_obs, done

    def add(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_next_obs[idx] = next_obs
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_dones[idx] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        indices = []
        p_total = self.sum_tree.total()

        # Safety check for total sum
        if p_total <= 0:
            p_total = 1.0

        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            # Ensure valid range
            if b - a <= 0:
                b = a + 1e-6

            # Clamp to valid bounds
            a = max(a, 0)
            b = min(b, p_total)

            try:
                upperbound = np.random.uniform(a, b)
                idx = self.sum_tree.retrieve(upperbound)

                # Ensure valid index
                idx = max(0, min(idx, self.size - 1))
                indices.append(idx)

            except (OverflowError, ValueError):
                # Fallback to random sampling
                idx = np.random.randint(0, self.size)
                indices.append(idx)

        samples = {
            "observations": torch.from_numpy(self.buffer_obs[indices]).to(self.device),
            "actions": torch.from_numpy(self.buffer_actions[indices]).to(self.device).unsqueeze(1),
            "rewards": torch.from_numpy(self.buffer_rewards[indices]).to(self.device).unsqueeze(1),
            "next_observations": torch.from_numpy(self.buffer_next_obs[indices]).to(self.device),
            "dones": torch.from_numpy(self.buffer_dones[indices]).to(self.device).unsqueeze(1),
        }

        # Get priorities with safety checks
        probs = []
        for idx in indices:
            try:
                prob = self.sum_tree.tree[idx + self.capacity - 1]
                probs.append(max(prob, self.eps))  # Ensure no zeros
            except IndexError:
                probs.append(self.eps)

        probs = np.array(probs, dtype=np.float64)

        # Calculate weights with safety checks
        # Ensure probs are normalized and not zero
        probs = np.maximum(probs, self.eps)
        probs_normalized = probs / p_total
        probs_normalized = np.maximum(probs_normalized, 1e-8)

        # Calculate importance sampling weights
        weights = np.power(self.size * probs_normalized, -self.beta)

        # Handle potential overflow/underflow
        weights = np.clip(weights, 1e-8, 1e8)

        # Normalize weights
        max_weight = np.max(weights)
        if max_weight > 0:
            weights = weights / max_weight
        else:
            weights = np.ones_like(weights)

        samples["weights"] = torch.from_numpy(weights).to(self.device).unsqueeze(1)
        samples["indices"] = indices

        return PrioritizedBatch(**samples)

    def update_priorities(self, indices, priorities):
        # Ensure priorities are valid
        priorities = np.abs(priorities) + self.eps
        priorities = np.clip(priorities, self.eps, 1e6)  # Cap maximum priority

        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indices, priorities):
            priority = priority**self.alpha
            self.sum_tree.update(idx, priority)
            self.min_tree.update(idx, priority)

def main():
    @dataclass
    class Args:
        exp_name: str = "notebook_run"
        """the name of this experiment"""
        seed: int = 42
        """seed of the experiment"""
        torch_deterministic: bool = True
        """if toggled, `torch.backends.cudnn.deterministic=False`"""
        cuda: bool = True
        """if toggled, cuda will be enabled by default"""
        track: bool = True
        """if toggled, this experiment will be tracked with Weights and Biases"""
        wandb_project_name: str = "cleanRL"
        """the wandb's project name"""
        wandb_entity: str = None
        """the entity (team) of wandb's project"""
        capture_video: bool = False
        """whether to capture videos of the agent performances (check out `videos` folder)"""
        save_model: bool = True
        """whether to save model into the `runs/{run_name}` folder"""
        upload_model: bool = False
        """whether to upload the saved model to huggingface"""
        hf_entity: str = ""
        """the user or org name of the model repository from the Hugging Face Hub"""

        env_id: str = "ALE/Pong-v5"
        """the id of the environment"""
        total_timesteps: int = 10_000_000
        """total timesteps of the experiments"""
        learning_rate: float = 0.00024159304323406795
        """the learning rate of the optimizer"""
        num_envs: int = 1
        """the number of parallel game environments"""
        buffer_size: int = 100_000
        """the replay memory buffer size"""
        gamma: float = 0.9848904522431526
        """the discount factor gamma"""
        tau: float = 0.007881662783282053
        """the target network update rate"""
        target_network_frequency: int = 500
        """the timesteps it takes to update the target network"""
        batch_size: int = 64
        """the batch size of sample from the replay memory"""
        start_e: float = 0
        """the starting epsilon for exploration"""
        end_e: float = 0
        """the ending epsilon for exploration"""
        exploration_fraction: float = 0
        """the fraction of `total-timesteps` it takes from start-e to go end-e"""
        learning_starts: int = 80000
        """timestep to start learning"""
        train_frequency: int = 4
        """the frequency of training"""
        n_step: int = 3
        """the number of steps to look ahead for n-step Q learning"""
        prioritized_replay_alpha: float = 0.5090792103106702
        """alpha parameter for prioritized replay buffer"""
        prioritized_replay_beta: float = 0.5472741836625601
        """beta parameter for prioritized replay buffer"""
        prioritized_replay_eps: float = 0.000001
        """epsilon parameter for prioritized replay buffer"""
        n_atoms: int = 51
        """the number of atoms"""
        v_min: float = -10
        """the return lower bound"""
        v_max: float = 10
        """the return upper bound"""

    if "ipykernel" in sys.argv[0] or "colab_kernel_launcher" in sys.argv[0]:
        args = Args()
    else:
        args = tyro.cli(Args)
        
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    
    # Setup run name and logging
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Initialize tensorboard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Initialize environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize networks
    q_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)

    # Initialize replay buffer
    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
        args.n_step,
        args.gamma,
        args.prioritized_replay_alpha,
        args.prioritized_replay_beta,
        args.prioritized_replay_eps,
    )

    # Training loop
    start_time = time.time()
    episodic_returns = np.zeros((envs.num_envs,))
    episodic_lengths = np.zeros((envs.num_envs,))
    
    obs, _ = envs.reset(seed=args.seed)
    
    try:
        for global_step in range(args.total_timesteps):
            # [Your existing training loop code]
            # ... [previous code remains unchanged]
            
            if global_step % 1000 == 0:
                print(f"Step {global_step}/{args.total_timesteps}")
                
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Cleanup
        envs.close()
        writer.close()
        
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(q_network.state_dict(), model_path)
            print(f"model saved to {model_path}")

if __name__ == "__main__":
    main()
