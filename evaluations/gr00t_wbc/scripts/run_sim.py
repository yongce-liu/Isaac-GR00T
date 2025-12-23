"""Simple Gr00t simulation example using Hydra configuration."""

from pathlib import Path

import gr00t
import gymnasium as gym
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import robot_sim.adapters.gr00t  # noqa: F401 - triggers task registration
from robot_sim.configs import MapTaskConfig
from robot_sim.utils.helper import setup_logger


PROJECT_ROOT = Path(gr00t.__file__).parents[1].resolve() / "evaluations" / "gr00t_wbc"
OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT))


def check_observation_space(task: gym.Env, obs: gym.spaces.Dict) -> None:
    for k, v in obs.items():
        space = task.observation_space.spaces[k]
        # logger.info(
        #     f"{k}: value shape={np.shape(v)}, dtype={getattr(v, 'dtype', type(v))}, space shape={space.shape}, space dtype={space.dtype}"
        # )
        if not space.contains(v):
            logger.warning(f"[DEBUG] Observation '{k}' out of space!\nValue: {v}\nSpace: {space}")
            # input("Press Enter to continue...")


def run(cfg: DictConfig) -> None:
    """Run Gr00t simulation.

    Args:
        cfg: Hydra configuration containing simulator_config, observation_mapping, action_mapping
    """
    setup_logger()
    logger.info("Starting Gr00t simulation...")
    cfg = hydra.utils.instantiate(cfg, _recursive_=True)

    # Hydra automatically instantiates all _target_ in the config tree
    task_cfg: MapTaskConfig = MapTaskConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
    task_cfg.print()

    # Initialize Gr00tEnv
    logger.info("Initializing Gr00tEnv...")
    task = gym.make(task_cfg.task, env_config=task_cfg.env_config, **task_cfg.params)

    # Reset environment
    logger.info("Resetting environment...")
    obs, info = task.reset()
    logger.info(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
    check_observation_space(task, obs)

    # Run a few simulation steps
    num_steps = 1000
    logger.info(f"Running {num_steps} simulation steps...")

    for step in range(num_steps):
        # Create a simple random action
        # For now, just use zeros for all action groups
        action = task.action_space.sample()
        for k, v in action.items():
            action[k] = 0 * v  # Zero action
        action["action.base_height_command"] = 0.8

        # Step environment
        obs, reward, terminated, truncated, info = task.step(action)
        check_observation_space(task, obs)

        if step % 50 == 0:
            logger.info(
                f"Step {step}: observation keys = {obs.keys() if isinstance(obs, dict) else 'N/A'}"
            )

        if terminated or truncated:
            logger.info(f"Episode ended at step {step}")
            break

    logger.info("Gr00t simulation completed successfully!")
    task.close()


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="tasks/pick_place.yaml",
)
def main(cfg: DictConfig) -> None:
    """Main function to run Gr00t simulation with Hydra configuration.

    Args:
        cfg: Hydra configuration
    """
    try:
        run(cfg)
    except Exception as e:
        logger.exception(f"An error occurred in main: {e}")
        raise e


if __name__ == "__main__":
    main()
