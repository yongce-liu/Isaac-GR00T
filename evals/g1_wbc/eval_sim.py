from functools import partial
import os
from pathlib import Path

from gr00t.eval.utils.helper import run_gr00t_sim_policy
import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from robot_sim.adapters.gr00t import Gr00tTaskConfig  # noqa: F401
from robot_sim.utils.helper import setup_logger


PROJECT_DIR = Path(__file__).parents[0].resolve()
os.chdir(PROJECT_DIR)


def make_env(task_cfg: Gr00tTaskConfig) -> gym.Env:
    """Create Gr00t gym environment."""
    task = gym.make(task_cfg.task, config=task_cfg.simulator, maps=task_cfg.maps, **task_cfg.params)
    return task


@hydra.main(
    version_base=None, config_path=str(PROJECT_DIR / "configs"), config_name="tasks/pick_place"
)
def main(cfg: DictConfig) -> None:
    """Run Gr00t simulation.

    Args:
        cfg: Hydra configuration containing simulator_config, observation_mapping, action_mapping
    """
    cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    setup_logger(f"{HydraConfig.get().runtime.output_dir}/{HydraConfig.get().job.name}.loguru.log")
    logger.info("Starting Gr00t simulation...")
    eval_cfg_path = PROJECT_DIR / f"configs/{cfg.pop('eval', 'default/eval.yaml')}"
    eval_cfg = hydra.utils.instantiate(OmegaConf.load(eval_cfg_path), _recursive_=True)
    eval_cfg.video.video_dir = (
        f"{HydraConfig.get().runtime.output_dir}/{HydraConfig.get().job.name}/videos"
    )
    task_cfg: Gr00tTaskConfig = Gr00tTaskConfig.from_dict(cfg)

    # Initialize Gr00tEnv
    res = run_gr00t_sim_policy(partial(make_env, task_cfg), eval_cfg)  # type: ignore
    logger.info("Gr00t simulation completed.")
    logger.info(f"Results: {res}")


if __name__ == "__main__":
    main()
