import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_subtask_prediction_example() -> dict:
    """Creates a random input example for subtask prediction."""
    return {
        "prompt": "do something",
        "keyframe_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "keyframe_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "keyframe_3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "keyframe_4": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "keyframe_5": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "keyframe_6": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_4": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_5": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_6": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_7": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_8": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_9": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "recent_frame_10": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    }

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

        keyframe_1 = _parse_image(data["keyframe_1"])            
        keyframe_2 = _parse_image(data["keyframe_2"])
        keyframe_3 = _parse_image(data["keyframe_3"])
        keyframe_4 = _parse_image(data["keyframe_4"])
        keyframe_5 = _parse_image(data["keyframe_5"])
        keyframe_6 = _parse_image(data["keyframe_6"])
        recent_1 = _parse_image(data["recent_frame_1"])
        recent_2 = _parse_image(data["recent_frame_2"])
        recent_3 = _parse_image(data["recent_frame_3"])
        recent_4 = _parse_image(data["recent_frame_4"])
        recent_5 = _parse_image(data["recent_frame_5"])
        recent_6 = _parse_image(data["recent_frame_6"])
        recent_7 = _parse_image(data["recent_frame_7"])
        recent_8 = _parse_image(data["recent_frame_8"])
        recent_9 = _parse_image(data["recent_frame_9"])
        recent_10 = _parse_image(data["recent_frame_10"])

        names = (
            "keyframe_1", "keyframe_2", "keyframe_3", "keyframe_4", "keyframe_5", "keyframe_6",
            "recent_frame_1", "recent_frame_2", "recent_frame_3", "recent_frame_4", "recent_frame_5", 
            "recent_frame_6", "recent_frame_7", "recent_frame_8", "recent_frame_9", "recent_frame_10"
        )
        images = (
            keyframe_1, keyframe_2, keyframe_3, keyframe_4, keyframe_5, keyframe_6,
            recent_1, recent_2, recent_3, recent_4, recent_5, recent_6, 
            recent_7, recent_8, recent_9, recent_10
        )
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