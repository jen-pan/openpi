#!/bin/bash
#SBATCH --job-name=pi0_fast_robomemory_finetune
#SBATCH --output=/iris/u/jrpan/openpi/slurm_jobs/slurm_out/pi0_fast_robomemory_finetune_%j.out 
#SBATCH --error=/iris/u/jrpan/openpi/slurm_jobs/slurm_err/pi0_fast_robomemory_finetune_%j.err
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=512G         
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=16
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodelist=iris-hgx-2
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=jrpan@stanford.edu
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

echo "Testing GPU availability:"
nvidia-smi

export DEBUG_MODE="false"  # Disable debug for real training
export PATH="/iris/u/jrpan/miniconda3/bin:$PATH"
export HF_HOME="/iris/u/jrpan/huggingface"
export OPENPI_DATA_HOME="/iris/u/jrpan/openpi"
export GOOGLE_APPLICATION_CREDENTIALS="/iris/u/jrpan/openpi/openpi-preview.json"

source ~/.bashrc
conda activate openpi311
cd /iris/u/jrpan/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $SLURM_JOB_NAME --batch_size 128 --exp-name=full_10k_$(date +%m%d) --resume
