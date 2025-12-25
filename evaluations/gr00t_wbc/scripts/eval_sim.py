import os
from pathlib import Path

from gr00t.eval.utils.config import ClientConfig, MultiStepConfig, VideoConfig
from gr00t.eval.utils.helper import run_gr00t_sim_policy
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf

# Also triggers task registration with gym.registry
from robot_sim.adapters.gr00t import Gr00tTaskConfig  # noqa: F401
from robot_sim.utils.helper import setup_logger


def get_env_fn(path: str) -> gym.Env:
    PROJECT_ROOT = Path(__file__).parents[1].resolve()
    os.chdir(PROJECT_ROOT)
    with hydra.initialize(
        config_path="../../../../robot_sim/examples/gr00t/configs", version_base=None
    ):
        cfg = hydra.compose(config_name=path, overrides=[])
    setup_logger(log_file=f"logs/{Path(__file__).stem}.loguru.log")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    task_cfg = Gr00tTaskConfig.from_dict(cfg)
    task = gym.make(task_cfg.task, config=task_cfg.environment, **task_cfg.params)
    return task


if __name__ == "__main__":
    # eval_config = tyro.cli(ClientConfig)
    eval_config = ClientConfig(
        task_config="configs/tasks/pick_place.yaml",
        n_episodes=2,
        max_episode_steps=1440,
        n_envs=1,
        policy_client_port=8888,
        policy_client_host="192.168.123.55",
        video=VideoConfig(),
        multistep=MultiStepConfig(n_action_steps=20, terminate_on_success=False),
    )
    # eval_config.video = VideoConfig()
    # eval_config.multistep = MultiStepConfig()

    eval_config.n_episodes = 1
    eval_config.max_episode_steps = 500
    eval_config.task_config = "tasks/pick_place.yaml"
    eval_config.multistep.n_action_steps = 20
    eval_config.multistep.terminate_on_success
    eval_config.n_envs = 1
    eval_config.policy_client_port = 8888
    eval_config.policy_client_host = "192.168.123.55"

    env = get_env_fn(eval_config.task_config)

    results = run_gr00t_sim_policy(
        env=env,
        embodiment_tag="unitree_g1",
        n_episodes=eval_config.n_episodes,
        model_path=eval_config.model_path,
        policy_client_host=eval_config.policy_client_host,
        policy_client_port=eval_config.policy_client_port,
        n_envs=eval_config.n_envs,
        config=eval_config,
    )

    print("=" * 100)
    print("Video saved to: ", eval_config.video.video_dir)
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
    print("=" * 100)
