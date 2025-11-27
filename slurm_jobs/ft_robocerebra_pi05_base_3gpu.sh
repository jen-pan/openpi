#!/bin/bash
#SBATCH --job-name=pi05_base_robocerebra_gpu3
#SBATCH --output=/iliad/u/ajaysri/episodic_memory/openpi-j/slurm_jobs/slurm_out/pi05_base_robocerebra_gpu3_%j.out      # Output file (generic)
#SBATCH --time=24:00:00              # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                    # Single node
#SBATCH --cpus-per-task=16           # CPU cores per task
#SBATCH --mem=512G                   # Memory
#SBATCH --account=iliad               # Account
#SBATCH --partition iliad
#SBATCH --gres=gpu:h200:3           # Request 3 H200 GPUs
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
CONFIG_ID=pi05_base_robocerebra_finetune_gpu3
cd /iliad/u/ajaysri/episodic_memory/openpi-j
GOOGLE_APPLICATION_CREDENTIALS=openpi-preview.json XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_ID --exp-name=ft_$CONFIG_ID --resume
