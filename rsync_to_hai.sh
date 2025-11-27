#!/bin/bash
# Rsync script to transfer openpi-j to HAI cluster for pi05_base_robocerebra training
#
# Usage: ./rsync_to_hai.sh [--dry-run]
#
# This syncs:
# - Source code (src/, scripts/)
# - Config files (pyproject.toml, etc.)
# - Norm stats (assets/pi05_base_robocerebra*/jennypan00/robocerebra/)
# - SLURM job scripts
# - GCS credentials (openpi-preview.json)
#
# This does NOT sync:
# - Model checkpoints (downloaded from GCS on HAI)
# - Training outputs (checkpoints/, wandb/)
# - Large data files (*.pkl, *.mp4, lerobot/)
# - Jupyter notebooks

set -e

# Configuration
SRC_DIR="/iliad2/u/ajaysri/episodic_memory/openpi-j"
HAI_HOST="ajaysri@hai.stanford.edu"
HAI_DEST="/hai/scratch/ajaysri/openpi-j"

# Check for dry-run flag
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
fi

echo "Syncing openpi-j to HAI cluster..."
echo "Source: $SRC_DIR"
echo "Destination: $HAI_HOST:$HAI_DEST"
echo ""

# Main rsync command
rsync -avz --progress $DRY_RUN \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude '.mypy_cache' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '.uv' \
    --exclude 'wandb/' \
    --exclude 'checkpoints' \
    --exclude 'iliad_checkpoints/' \
    --exclude 'checkpoint_videos/' \
    --exclude 'data/' \
    --exclude 'lerobot/' \
    --exclude '*.pkl' \
    --exclude '*.mp4' \
    --exclude '*.ipynb' \
    --exclude 'dagger_processing/' \
    --exclude '*.egg-info' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '.eggs/' \
    "$SRC_DIR/" "$HAI_HOST:$HAI_DEST/"

echo ""
echo "=== Sync complete ==="
echo ""
echo "Next steps on HAI cluster:"
echo "1. SSH to HAI: ssh $HAI_HOST"
echo "2. cd $HAI_DEST"
echo "3. Create a SLURM script for HAI (partition/account may differ)"
echo "4. Submit job: sbatch slurm_jobs/ft_robocerebra_pi05_base_hai.sh"
