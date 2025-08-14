import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robomemory_example() -> dict:
    """Creates a random input example for the RoboMemory policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboMemoryInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        print("----------ROBOMEMORY DATA", data)
        assert "prompt" in data, "prompt must be in data for either action prediction or subtask prediction"
        prompt = data["prompt"] if isinstance(data["prompt"], str) else data["prompt"].decode("utf-8")

        # action prediction task only
        if "gripper_position" in data:
            gripper_pos = np.asarray(data["gripper_position"])
            if gripper_pos.ndim == 0:
                # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
                gripper_pos = gripper_pos[np.newaxis]
            state = np.concatenate([data["joint_position"], gripper_pos])
            
            base_image = _parse_image(data["exterior_image_1_left"])
            wrist_image = _parse_image(data["wrist_image_left"])

            match self.model_type: 
                case _model.ModelType.PI0 | _model.ModelType.PI05:
                    names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                    images = (base_image, wrist_image, np.zeros_like(base_image))
                    image_masks = (np.True_, np.True_, np.False_)
                case _model.ModelType.PI0_FAST:
                    names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                    # We don't mask out padding images for FAST models.
                    images = (base_image, np.zeros_like(base_image), wrist_image)
                    image_masks = (np.True_, np.True_, np.True_)

            inputs = {
                "prompt": prompt,
                "state": state,
                "image": dict(zip(names, images, strict=True)),
                "image_mask": dict(zip(names, image_masks, strict=True)),
                "actions": np.array(data["actions"]),
            }

        # subtask prediction task only
        else:
            assert "subtask_target" in data, "subtask_target must be in data for subtask prediction task"
            assert self.model_type == _model.ModelType.PI05, "subtask prediction task only supported for PI05 model"

            keyframe_1 = _parse_image(data["keyframe_1"])
            recent_1 = _parse_image(data["recent_frame_1"])

            names = ("keyframe_1", "recent_frame_1", "right_wrist_0_rgb") # TODO: need to make this longer
            images = (keyframe_1, recent_1, np.zeros_like(keyframe_1))
            image_masks = (np.True_, np.True_, np.False_)
            
            inputs = {
                "prompt": prompt,
                "image": dict(zip(names, images, strict=True)),
                "image_mask": dict(zip(names, image_masks, strict=True)),
                "subtask_target": data["subtask_target"] if isinstance(data["subtask_target"], str) else data["subtask_target"].decode("utf-8"),
            }
        
        print("----------ROBOMEMORY INPUTS", inputs)
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMemoryOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        return {"actions": np.asarray(data["actions"][:, :8])}