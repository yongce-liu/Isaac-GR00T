from collections import defaultdict
from functools import partial
from pathlib import Path
import time
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.eval.utils.config import ClientConfig
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm


'''
def get_robocasa_env_fn(
    env_name: str,
):
    def env_fn():
        import os

        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        return gym.make(env_name, enable_render=True)

    return env_fn


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_behavior_env_fn(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
):
    def env_fn():
        from gr00t.eval.sim.BEHAVIOR.behavior_env import register_behavior_envs

        register_behavior_envs()
        return gym.make(env_name, env_idx=env_idx, total_n_envs=total_n_envs)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)

    if env_embodiment in (
        EmbodimentTag.GR1,
        EmbodimentTag.ROBOCASA_PANDA_OMRON,
    ):
        env_fn = get_robocasa_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.UNITREE_G1,):
        env_fn = get_groot_locomanip_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.OXE_GOOGLE, EmbodimentTag.OXE_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.BEHAVIOR_R1_PRO,):
        env_fn = get_behavior_env_fn(env_name, env_idx, total_n_envs)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()
'''


def create_eval_env(
    env: gym.Env,
    config: ClientConfig,
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    if config.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
            max_episode_steps=config.video.max_episode_steps,
            overlay_text=config.video.overlay_text,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
        terminate_on_success=config.multistep.terminate_on_success,
    )
    return env


def run_rollout_gymnasium_policy(
    env: gym.Env,
    policy: BasePolicy,
    n_episodes: int,
    n_envs: int,
    config: ClientConfig,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy_fn: Function that creates a policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
    Returns:
        Collection results from running the episodes
    """

    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env.spec.id} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env=env,
            config=config,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

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

    return env.spec.id, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str,
    policy_client_port: int,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        policy = Gr00tSimPolicyWrapper(
            Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=0,
            )
        )
    return policy


def run_gr00t_sim_policy(
    env: gym.Env,
    embodiment_tag: EmbodimentTag,
    n_episodes: int,
    model_path: str,
    policy_client_host: str,
    policy_client_port: int,
    n_envs: int,
    config: ClientConfig,
):
    embodiment_tag = EmbodimentTag(embodiment_tag)

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    results = run_rollout_gymnasium_policy(
        env=env,
        policy=policy,
        n_episodes=n_episodes,
        n_envs=n_envs,
        config=config,
    )
    return results
