import argparse
import logging
import time
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download

COLUMNS = [
        "prompt", "keyframe_1", "keyframe_2", "keyframe_3", "keyframe_4", 
        "keyframe_5", "keyframe_6", "recent_frame_1", "recent_frame_2", 
        "recent_frame_3", "recent_frame_4", "recent_frame_5", "recent_frame_6", 
        "recent_frame_7", "recent_frame_8", "recent_frame_9", "recent_frame_10"
    ]

def load_policy(config, checkpoint_dir: str, task):
    try:
        print("Loading PI05 policy...")
        config = _config.get_config(config)
        checkpoint_dir = download.maybe_download(checkpoint_dir)
        policy = _policy_config.create_trained_policy(
            config, 
            checkpoint_dir, 
            task=task
        )
        print(f"Policy loaded successfully. Input transform: {policy._input_transform}")
        return policy
    except Exception as e:
        logging.error(f"Failed to load policy: {e}")
        raise

def prepare_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df[COLUMNS].to_dict('records')


def generate_predictions(policy, examples: List[Dict[str, Any]], ground_truths: List[str], prompts: List[str]) -> List[str]:
    predictions = []
    
    for i, example in enumerate(tqdm(examples, desc="Generating predictions")):
        try:
            result = policy.infer_subtask(example)
            prediction = result.get("subtask_target", "")
            predictions.append(prediction)
            
            # Log each prediction as it's generated
            print(f"\n--- Example {i+1}/{len(examples)} ---")
            print(f"Prompt: {prompts[i][:150]}{'...' if len(prompts[i]) > 150 else ''}")
            print(f"Ground Truth: {ground_truths[i]}")
            print(f"Prediction: {prediction}")
            exact_match = prediction.strip().lower() == ground_truths[i].strip().lower()
            print(f"Exact Match: {'✓' if exact_match else '✗'}")
                
        except Exception as e:
            import traceback
            logging.error(f"Error generating prediction for example {i}: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            logging.error(f"Example data keys: {list(example.keys())}")
            predictions.append("")
    
    return predictions

def calculate_exact_match_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred.strip().lower() == gt.strip().lower())
    return exact_matches / len(predictions) if predictions else 0.0

def save_results(predictions: List[str], ground_truths: List[str], prompts: List[str], output_file: str):
    results_df = pd.DataFrame({
        'prompt': prompts,
        'ground_truth': ground_truths,
        'prediction': predictions,
        'exact_match': [pred.strip().lower() == gt.strip().lower() for pred, gt in zip(predictions, ground_truths)]
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def evaluate_subtask_predictions(
    data_file: str,
    checkpoint_dir: str,
    max_examples: int,    
    output_file: str,
    config: str,
    task: str,
):
    try:
        print(f"Loading data from {data_file}")
        df = pd.read_pickle(data_file)
        print(f"Loaded {len(df)} examples")
    except Exception as e:
        logging.error(f"Failed to load data from {data_file}: {e}")
        raise
    
    if max_examples:
        df = df.head(max_examples)
        print(f"Using first {len(df)} examples")

    # Replace any null/empty image columns with a null image
    # TODO: this is a hack to handle the fact that the images are sometimes null. this is dealt with when uploaded to lerobot but im using unprocessed pkl files here
    null_image = np.zeros((180, 320, 3), dtype=np.uint8)    
    for col in [c for c in COLUMNS if c != "prompt"]:
        df[col] = df[col].apply(lambda x: null_image if (isinstance(x, float) and pd.isna(x)) or x is None else x)

    policy = load_policy(config, checkpoint_dir, task)
    
    # Prepare all data at once
    examples = prepare_data(df)
    ground_truths = df["subtask_target"].tolist()
    prompts = df["prompt"].tolist()
    
    # Generate predictions for all examples
    start_time = time.time()
    predictions = generate_predictions(policy, examples, ground_truths, prompts)
    total_time = time.time() - start_time
    
    print(f"Generated {len(predictions)} predictions in {total_time:.2f}s")
    print(f"Average time per prediction: {total_time/len(predictions):.3f}s")
    
    # Show sample results
    if len(predictions) > 0:
        print(f"Sample - Prompt: {prompts[0][:100]}...")
        print(f"Sample - Ground Truth: {ground_truths[0]}")
        print(f"Sample - Prediction: {predictions[0]}")
    
    exact_match_acc = calculate_exact_match_accuracy(predictions, ground_truths)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Total examples processed: {len(predictions)}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f} ({exact_match_acc*100:.2f}%)")
    
    try:
        save_results(predictions, ground_truths, prompts, output_file)
    except Exception as e:
        logging.error(f"Failed to save results to {output_file}: {e}")
        raise
    
    return {
        "exact_match_accuracy": exact_match_acc,
        "total_examples": len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PI05 subtask predictions")
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="subtask_prediction_df_16_frames_test.pkl",
        help="Path to the pickle file containing test data"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str, 
        default="gs://openpi-assets-preview/checkpoints/pi05_droid",
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (for testing)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/iris/u/jrpan/openpi/subtask_eval_results.csv",
        help="Output file for detailed results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pi05_cotrain",
        help="Config to use for loading the policy"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="subtask_pred",
        choices=["subtask_pred", "action_pred"],
        help="Task to evaluate (only subtask_pred or action_pred possible)"
    )
    
    args = parser.parse_args()
    
    try:
        results = evaluate_subtask_predictions(
            data_file=args.data_file,
            checkpoint_dir=args.checkpoint_dir,
            max_examples=args.max_examples,
            output_file=args.output_file,
            config=args.config,
            task=args.task
        )
        print(results)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
