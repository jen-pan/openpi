#!/bin/bash
#SBATCH --job-name=pi05_base_robocerebra_gpu8
#SBATCH --output=/hai/scratch/ajaysri/openpi-j/slurm_jobs/slurm_out/pi05_base_robocerebra_gpu8_%j.out
#SBATCH --time=24:00:00              # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                    # Single node
#SBATCH --cpus-per-task=64           # CPU cores per task
#SBATCH --mem=512G                   # Memory
#SBATCH --partition=iris-hi          # HAI partition
#SBATCH --gres=gpu:h100:8            # Request 8 H100 GPUs
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=ajaysri@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

# Report GPU availability
echo "Testing GPU availability:"
nvidia-smi

source ~/.bashrc
CONFIG_ID=pi05_base_robocerebra_finetune_gpu8
cd /hai/scratch/ajaysri/openpi-j
GOOGLE_APPLICATION_CREDENTIALS=openpi-preview.json XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_ID --exp-name=ft_$CONFIG_ID --resume
