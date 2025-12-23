from dataclasses import dataclass, field
import uuid

import numpy as np


##########################################
######## Config
##########################################
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
    n_action_steps: int = 8


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
class ClientConfig:
    """A client config.
    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
        max_episode_steps: Maximum number of steps per episode
        n_episodes: Number of episodes to run
        model_path: Path to the model
        policy_client_host: Host address for policy client
        policy_client_port: Port number for policy client
        task: Name of the environment
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps
    """

    task_config: str = ""
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    max_episode_steps: int = 504
    n_episodes: int = 50
    model_path: str = ""
    policy_client_host: str = ""
    policy_client_port: int | None = None
    n_envs: int = 1

    def __post_init__(self):
        # validate policy configuration
        assert (self.model_path and not (self.policy_client_host or self.policy_client_port)) or (
            not self.model_path and self.policy_client_host and self.policy_client_port is not None
        ), (
            "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
            "If all 3 arguments are provided, explicitly choose one:\n"
            '  - To use policy client: set --policy_client_host and --policy_client_port, and set --model_path ""\n'
            '  - To use model path: set --model_path, and set --policy_client_host "" (and leave --policy_client_port unset)'
        )

        self.video.max_episode_steps = self.max_episode_steps
        self.multistep.max_episode_steps = self.max_episode_steps
        # self.multistep.terminate_on_success = True
        if self.video.video_dir is None and self.model_path:
            video_dir = f"/tmp/sim_eval_videos_{self.model_path.split('/')[-3]}_ac{self.multistep.n_action_steps}_{uuid.uuid4()}"
        elif self.video.video_dir is None:
            video_dir = f"/tmp/sim_eval_videos_{self.task_config.split('/')[-1]}_ac{self.multistep.n_action_steps}_{uuid.uuid4()}"
        elif self.task_config.startswith("sim_behavior_r1_pro"):
            # BEHAVIOR sim will crash if decord is imported in video_utils.py
            video_dir = None
        self.video.video_dir = video_dir
