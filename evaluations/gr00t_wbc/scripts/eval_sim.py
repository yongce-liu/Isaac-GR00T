from pathlib import Path

from gr00t.eval.utils.config import ClientConfig
from gr00t.eval.utils.helper import run_gr00t_sim_policy
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf
from robot_sim.configs import MapTaskConfig
from robot_sim.utils.helper import setup_logger
import tyro


def get_env_fn(path: str) -> gym.Env:
    PROJECT_ROOT = Path(__file__).parents[1]
    OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT))
    cfg = OmegaConf.load(path)
    setup_logger(log_file="output_path")
    cfg = hydra.utils.instantiate(cfg, _recursive_=True)
    task_cfg = MapTaskConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
    task_cfg.print()
    task = gym.make(task_cfg.task, env_config=task_cfg.env_config, **task_cfg.params)

    return task


if __name__ == "__main__":
    eval_config = tyro.cli(ClientConfig)
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
