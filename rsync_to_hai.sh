#!/bin/bash
# Rsync script to transfer datasets and norm stats to HAI cluster
#
# Usage: ./rsync_to_hai.sh [--dry-run]
#
# This syncs:
# - LeRobot dataset from /iliad2/u/ajaysri/lerobot_data/jennypan00/robocerebra
# - Norm stats (assets/)

set -e

# Configuration
HAI_HOST="ajaysri@hai.stanford.edu"
LEROBOT_SRC="/iliad2/u/ajaysri/lerobot_data/jennypan00/robocerebra"
LEROBOT_DEST="/hai/scratch/ajaysri/lerobot_data/jennypan00/robocerebra"
ASSETS_SRC="/iliad2/u/ajaysri/episodic_memory/openpi-j/assets"
ASSETS_DEST="/hai/scratch/ajaysri/openpi-j/assets"

# Check for dry-run flag
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
fi

echo "Syncing datasets and norm stats to HAI cluster..."
echo ""

# Sync lerobot dataset
echo "=== Syncing robocerebra dataset ==="
echo "Source: $LEROBOT_SRC"
echo "Destination: $HAI_HOST:$LEROBOT_DEST"
rsync -avz --progress $DRY_RUN \
    "$LEROBOT_SRC/" "$HAI_HOST:$LEROBOT_DEST/"

# Sync norm stats in assets/
echo ""
echo "=== Syncing assets/ (norm stats) ==="
echo "Source: $ASSETS_SRC"
echo "Destination: $HAI_HOST:$ASSETS_DEST"
rsync -avz --progress $DRY_RUN \
    "$ASSETS_SRC/" "$HAI_HOST:$ASSETS_DEST/"

echo ""
echo "=== Sync complete ==="
