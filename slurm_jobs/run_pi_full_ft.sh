#!/bin/bash
#SBATCH --job-name=pi0_fast_robomemory_finetune
#SBATCH --output=/iris/u/jrpan/slurm_jobs/slurm_out/pi0_fast_robomemory_finetune_%j.out 
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=16           
#SBATCH --mem=512G         
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodelist=iris-hgx-2
#SBATCH --gres=gpu:h200:2
#SBATCH --error=/iris/u/jrpan/slurm_jobs/slurm_err/pi0_fast_robomemory_finetune_%j.err
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=jrpan@stanford.edu
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

echo "Testing GPU availability:"
nvidia-smi

source ~/.bashrc
conda activate openpi311
cd /iris/u/jrpan/openpi
export DEBUG_MODE="false"  # Disable debug for real training

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $SLURM_JOB_NAME --batch_size 128 --exp-name=full_10k_$(date +%m%d) --resume