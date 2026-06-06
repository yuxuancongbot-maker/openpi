import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format.

    LeRobot stores images as float32 (C, H, W), so we convert to uint8 (H, W, C)
    for the model.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Converts Franka FR3 observations to the model input format.

    This transform is used for both training and inference. It maps the Franka
    dataset keys (observation/image, observation/wrist_image, observation/state,
    actions, prompt) to the standard model input format.

    The Franka FR3 robot has:
      - 7 joint angles
      - 2 finger joint positions
      - 2 cameras: cam_high (third-person view) and cam_wrist (wrist-mounted view)
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images from LeRobot format (float32 CHW) to model format (uint8 HWC).
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict with model-expected image keys.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Franka only has one wrist camera; pad the right wrist with zeros.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Only mask padding images for pi0 model, not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the Franka action format.

    Used during inference to extract the correct action dimensions from the
    (potentially padded) model output.

    The model outputs actions with shape (action_horizon, model_action_dim),
    where model_action_dim may be larger than Franka's 9 dims (e.g., 32 for pi0).
    This transform slices the action dimension down to the real 9 dims
    (7 joint angles + 2 finger joints).
    """

    def __call__(self, data: dict) -> dict:
        # Slice the last axis (action_dim) to keep only the first 9 dimensions,
        # removing model padding. Keeps all action_horizon steps.
        return {"actions": np.asarray(data["actions"][:, :9])}
