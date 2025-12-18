from dataclasses import dataclass
import json
import os
import random
from typing import Any, Literal
import warnings

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import numpy as np
import torch
import tyro


warnings.simplefilter("ignore", category=FutureWarning)


###############################################################################
# TENSORRT Module Wrappers
###############################################################################


def set_seed(seed: int = 0):
    """
    Set seed for all random number generators.
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA ops
    torch.use_deterministic_algorithms(True, warn_only=True)

    # For cuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch requires this to be set for some CUDA kernels
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TensorRTDiTWrapper:
    """Wrapper for TensorRT DiT engine."""

    def __init__(self, engine_path: str, device: int = 0):
        import tensorrt as trt

        self.device = device

        # Ensures CUDA driver is properly loaded
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(device)  # Set the specified CUDA device
            print(f"CUDA initialized via PyTorch: device {device}")
        else:
            raise RuntimeError("CUDA not available for TensorRT")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        print(f"TensorRT engine loaded: {engine_path}")

    def __call__(self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None):
        """Forward pass through TensorRT DiT."""
        # Setup context bindings
        sa_embs = sa_embs.to(f"cuda:{self.device}").contiguous()
        vl_embs = vl_embs.to(f"cuda:{self.device}").contiguous()
        timestep = timestep.to(f"cuda:{self.device}").contiguous()  # Keep as int64

        if image_mask is not None:
            image_mask = image_mask.to(f"cuda:{self.device}").contiguous()
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(f"cuda:{self.device}").contiguous()

        self.context.set_input_shape("sa_embs", sa_embs.shape)
        self.context.set_input_shape("vl_embs", vl_embs.shape)
        self.context.set_input_shape("timestep", timestep.shape)
        if image_mask is not None:
            self.context.set_input_shape("image_mask", image_mask.shape)
        if backbone_attention_mask is not None:
            self.context.set_input_shape("backbone_attention_mask", backbone_attention_mask.shape)

        self.context.set_tensor_address("sa_embs", sa_embs.data_ptr())
        self.context.set_tensor_address("vl_embs", vl_embs.data_ptr())
        self.context.set_tensor_address("timestep", timestep.data_ptr())
        if image_mask is not None:
            self.context.set_tensor_address("image_mask", image_mask.data_ptr())
        if backbone_attention_mask is not None:
            self.context.set_tensor_address(
                "backbone_attention_mask", backbone_attention_mask.data_ptr()
            )

        # Output in BF16 (matches ONNX export and engine precision)
        output_shape = self.context.get_tensor_shape("output")
        output = torch.empty(
            tuple(output_shape), dtype=torch.bfloat16, device=f"cuda:{self.device}"
        )
        self.context.set_tensor_address("output", output.data_ptr())

        success = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not success:
            raise RuntimeError("TensorRT inference failed")

        return output


def replace_dit_with_tensorrt(policy: Gr00tPolicy | Any, trt_engine_path: str, device: int = 0):
    """Replace the DiT forward method with TensorRT inference."""
    trt_dit = TensorRTDiTWrapper(trt_engine_path, device=device)

    def trt_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        """
        TensorRT wrapper matching DiT forward signature.

        Maps DiT parameter names to ONNX export names:
        - hidden_states -> sa_embs
        - encoder_hidden_states -> vl_embs
        - timestep -> timestep
        - image_mask, backbone_attention_mask passed through
        """
        output = trt_dit(
            sa_embs=hidden_states,
            vl_embs=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

        # DiT returns (output, all_hidden_states) when return_all_hidden_states=True
        if return_all_hidden_states:
            # TensorRT only returns the final output, not intermediate states
            # For inference, we don't need intermediate states, so raise
            # as this seems invalid config for inference
            raise RuntimeError("TensorRT only returns the final output. Check inference config")
        else:
            return output

    policy.model.action_head.model.forward = trt_forward
    print(" DiT replaced with TensorRT engine")


###############################################################################
# MAIN
###############################################################################


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    seed: int = 0
    """Seed to use for reproducibility."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "127.0.0.1"
    """Host address for the server"""

    port: int = 8888
    """Port number for the server"""

    strict: bool = False
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """Inference mode: 'pytorch' (default) or 'tensorrt'."""

    trt_engine_path: str = None
    """Path to TensorRT engine file (.trt). Used only when inference_mode='tensorrt'."""


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Setting seed: {config.seed}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )

        # Apply inference mode: TensorRT or PyTorch
        if config.inference_mode == "tensorrt":
            replace_dit_with_tensorrt(policy, config.trt_engine_path)
        else:
            # PyTorch mode with torch.compile
            policy.model.action_head.model.forward = torch.compile(
                policy.model.action_head.model.forward, mode="max-autotune"
            )

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
