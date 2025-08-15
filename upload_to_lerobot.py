import numpy as np
import pandas as pd
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import shutil
from pathlib import Path
import os
import argparse

def create_action_pred_dataset(data, repo_id, robot_type, fps, dof):
    """Create and populate LeRobot dataset for action prediction task."""
    ds = LeRobotDataset.create(
        repo_id   = repo_id,    
        robot_type= robot_type,
        fps       = fps,
        features  = {
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float64",
                "shape": (7,),              # observation/joint positions + observation/gripper position
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float64",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "prompt": {
                "dtype": "string",
                "shape": (1,),
                "names": ["prompt"],
            },
            "actions": {
                "dtype" : "float64",
                "shape" : (DOF + 1,),        # action/joint velocity + action/gripper position
                "names" : ["actions"],   
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    current_prompt = None
    for _, row in data.iterrows():
        prompt = row.prompt
        joint_position = row.joint_position
        gripper_position = row.gripper_position
        wrist_image = row.wrist_image
        exterior_image = row.exterior_image
        actions = row.actions
        
        # If prompt changes or this is the first row, save previous episode (if any) and start new one
        if current_prompt is not None and current_prompt != prompt:
            print("Saving episode for prompt: ", current_prompt, ". New prompt is: ", prompt)
            ds.save_episode()
        
        # Add frame to current episode
        ds.add_frame(frame = {"exterior_image_1_left": exterior_image, "wrist_image_left": wrist_image, "joint_position": joint_position, "gripper_position": gripper_position, "actions": actions, "prompt": prompt}, task = prompt)
        
        current_prompt = prompt

    if current_prompt is not None:
        print("Saving final episode for prompt: ", current_prompt, ". FINISHED!")
        ds.save_episode()
    
def create_subtask_pred_dataset(data, repo_id, robot_type, fps, dof):
    """Create and populate LeRobot dataset for subtask prediction task."""
    ds = LeRobotDataset.create(
        repo_id   = repo_id,    
        robot_type= robot_type,
        fps       = fps,
        features  = {
            "recent_frame_1": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "keyframe_1": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "prompt": {
                "dtype": "string",
                "shape": (1,),
                "names": ["prompt"],
            },
            "subtask_target": {
                "dtype": "string",
                "shape": (1,),
                "names": ["subtask_target"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for _, row in data.iterrows():
        prompt = row.prompt
        keyframe_1 = row.keyframe_1
        recent_frame_1 = row.recent_frame_1
        subtask_target = row.subtask_target
        
        if type(keyframe_1) == float:
            keyframe_1 = np.zeros((180, 320, 3), dtype=np.uint8) # TODO: deal with nulls
        if type(recent_frame_1) == float:
            recent_frame_1 = np.zeros((180, 320, 3), dtype=np.uint8) # TODO: deal with nulls
            
        ds.add_frame(frame = {"recent_frame_1": recent_frame_1, "keyframe_1": keyframe_1, "prompt": prompt, "subtask_target": subtask_target}, task = prompt)
    ds.save_episode()
        
def main():
    parser = argparse.ArgumentParser(description='Upload data to LeRobot dataset')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the pickle file containing the data')
    parser.add_argument('--split', type=str, required=True, help='Dataset split (train, test)')
    parser.add_argument('--repo_id', type=str, required=True, help='HuggingFace repository ID')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--dof', type=int, default=7, help='Degrees of freedom')
    parser.add_argument('--robot_type', type=str, default='panda', help='Robot type')
    parser.add_argument('--push', action='store_true', help='Push to HuggingFace hub')
    parser.add_argument('--task', type=str, required=True, choices=['action_pred', 'subtask_pred'], help='Task type: either action_pred or subtask_pred')
    
    args = parser.parse_args()
        
    DATA = pd.read_pickle(args.data_file)
    print("Keys: ", DATA.iloc[0].keys())
    print("Length: ", len(DATA))

    HF_LEROBOT_HOME = os.environ['HF_LEROBOT_HOME']
    REPO_ID     = args.repo_id
    FPS         = args.fps
    DOF         = args.dof
    ROBOT_TYPE  = args.robot_type
    PUSH        = args.push
    TASK        = args.task
    output_path = Path(HF_LEROBOT_HOME) / REPO_ID 

    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    if TASK == 'action_pred':
        create_action_pred_dataset(DATA, REPO_ID, ROBOT_TYPE, FPS, DOF)
    elif TASK == 'subtask_pred':
        create_subtask_pred_dataset(DATA, REPO_ID, ROBOT_TYPE, FPS, DOF)

if __name__ == "__main__":
    main()