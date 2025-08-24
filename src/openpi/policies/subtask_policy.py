import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_subtask_prediction_example() -> dict:
    """Creates a random input example for subtask prediction."""
    example = {"prompt": "do something"}
    
    for i in range(1, _model.KEYFRAMES + 1):
        example[f"keyframe_{i}"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    
    for i in range(1, _model.RECENT_FRAMES + 1):
        example[f"recent_frame_{i}"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    
    return example

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SubtaskPredictionInputs(transforms.DataTransformFn):
    """
    Input transform specifically for subtask prediction tasks.
    This handles datasets that contain keyframes and recent frames for predicting subtasks.
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        assert "prompt" in data, "prompt must be in data for subtask prediction"
        assert self.model_type == _model.ModelType.PI05, "subtask prediction task only supported for PI05 model"
        
        prompt = data["prompt"] if isinstance(data["prompt"], str) else data["prompt"].decode("utf-8")

        keyframe_names = [f"keyframe_{i}" for i in range(1, _model.KEYFRAMES + 1)]
        keyframes = [_parse_image(data[name]) for name in keyframe_names]
        
        recent_names = [f"recent_frame_{i}" for i in range(1, _model.RECENT_FRAMES + 1)]
        recent_frames = [_parse_image(data[name]) for name in recent_names]

        names = tuple(keyframe_names + recent_names)
        images = tuple(keyframes + recent_frames)
        image_masks = tuple(np.True_ for _ in range(len(names)))
        
        inputs = {
            "prompt": prompt,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        
        # Include subtask target if available (for training)
        if "subtask_target" in data:
            inputs["subtask_target"] = data["subtask_target"] if isinstance(data["subtask_target"], str) else data["subtask_target"].decode("utf-8")
    
        return inputs


@dataclasses.dataclass(frozen=True)
class SubtaskPredictionOutputs(transforms.DataTransformFn):
    """
    Output transform for subtask prediction.
    Since subtask prediction doesn't produce actions, this is mainly a pass-through.
    """

    def __call__(self, data: dict) -> dict:
        # For subtask prediction, we typically just pass through the predictions
        # The actual subtask predictions are handled by the model's subtask head
        return data 