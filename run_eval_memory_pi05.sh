#!/bin/bash
#SBATCH --job-name=pi05_eval_subtask_pred
#SBATCH --output=/iris/u/jrpan/openpi/slurm_jobs/slurm_out/eval_subtask_pred_%j.out 
#SBATCH --error=/iris/u/jrpan/openpi/slurm_jobs/slurm_err/eval_subtask_pred_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=8           
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=jrpan@stanford.edu

source ~/.bashrc
conda activate openpi311
cd /iris/u/jrpan/openpi

DATA_FILE="/iris/u/jrpan/openpi/subtask_prediction_df_16_frames_test.pkl"
CHECKPOINT_DIR="/iris/u/jrpan/openpi/checkpoints/pi05_cotrain_subtask_only/pi05_cotrain_0820/9000"
CONFIG="pi05_cotrain"
TASK="subtask_pred"
OUTPUT_FILE="/iris/u/jrpan/openpi/subtask_pred_results_subtask_only_test.csv"
MAX_EXAMPLES=100

python /iris/u/jrpan/openpi/eval_memory_pi05.py \
  --data_file "$DATA_FILE" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --config "$CONFIG" \
  --task "$TASK" \
  --max_examples $MAX_EXAMPLES \
  --output_file "$OUTPUT_FILE"