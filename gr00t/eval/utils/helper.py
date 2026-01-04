from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
from loguru import logger
import numpy as np
from tqdm import tqdm


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    # n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class EvalConfig:
    embodiment_tag: str
    model_path: str | None = None
    max_episode_steps: int = 504
    n_episodes: int = 50
    policy_client_host: str = "loalhost"
    policy_client_port: int = 8888
    n_envs: int = 1
    # n_action_steps: int = 8
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    def __post_init__(self):
        if self.max_episode_steps != self.multistep.max_episode_steps:
            logger.warning(
                f"Overriding multistep.max_episode_steps ({self.multistep.max_episode_steps}) "
                f"with eval.max_episode_steps ({self.max_episode_steps})"
            )
            self.multistep.max_episode_steps = self.max_episode_steps
        if self.max_episode_steps != self.video.max_episode_steps:
            logger.warning(
                f"Overriding video.max_episode_steps ({self.video.max_episode_steps}) "
                f"with eval.max_episode_steps ({self.max_episode_steps})"
            )
            self.video.max_episode_steps = self.max_episode_steps


def wrap_env(env_fn: Callable[[], gym.Env], **wrapper_configs) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """
    env = env_fn()
    video: VideoConfig = wrapper_configs.pop("video", VideoConfig())
    if video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=video.fps,
            codec=video.codec,
            input_pix_fmt=video.input_pix_fmt,
            crf=video.crf,
            thread_type=video.thread_type,
            thread_count=video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(video.video_dir),
            steps_per_render=video.steps_per_render,
            max_episode_steps=video.max_episode_steps,
            overlay_text=video.overlay_text,
        )

    multistep: MultiStepConfig = wrapper_configs.pop("multistep", MultiStepConfig())
    env = MultiStepWrapper(
        env,
        video_delta_indices=multistep.video_delta_indices,
        state_delta_indices=multistep.state_delta_indices,
        n_action_steps=multistep.n_action_steps,
        max_episode_steps=multistep.max_episode_steps,
        terminate_on_success=multistep.terminate_on_success,
    )

    if len(wrapper_configs) > 0:
        raise ValueError(f"Unknown wrapper configs: {wrapper_configs.keys()}")

    return env


def run_rollout_gymnasium_policy(
    env_fn: Callable[[], gym.Env], policy: BasePolicy, n_episodes: int = 10, n_envs: int = 1
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy_fn: Function that creates a policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes with {n_envs} vec envs")

    _envs = [env_fn for _ in range(n_envs)]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(_envs)  # type: ignore
    else:
        env = gym.vector.AsyncVectorEnv(_envs, shared_memory=False, context="spawn")  # type: ignore

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)

    # Initial reset
    observations, _ = env.reset()
    policy.reset()
    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        actions, _ = policy.get_action(observations)
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_successes.append(current_successes[env_idx])
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return episode_successes, episode_infos


def create_gr00t_sim_policy(
    embodiment_tag: EmbodimentTag,
    model_path: str | None = None,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        policy = Gr00tSimPolicyWrapper(
            Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=0)
        )  # type: ignore
    return policy


def run_gr00t_sim_policy(env_fn: Callable[[], gym.Env], config: EvalConfig):
    if config.video.video_dir is None:
        config.video.video_dir = f"/tmp/sim_eval_{config.embodiment_tag}_videos_ac{config.multistep.n_action_steps}_{uuid.uuid4()}"

    policy = create_gr00t_sim_policy(
        EmbodimentTag(config.embodiment_tag),
        config.model_path,
        config.policy_client_host,
        config.policy_client_port,
    )
    wrapped_env_fn = partial(
        wrap_env,
        env_fn,
        video=config.video,
        multistep=config.multistep,
    )

    results = run_rollout_gymnasium_policy(
        env_fn=wrapped_env_fn, policy=policy, n_episodes=config.n_episodes, n_envs=config.n_envs
    )
    print("Video saved to: ", config.video.video_dir)
    return results
