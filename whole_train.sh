#!/bin/bash
#SBATCH --job-name=whole_train
#SBATCH --cpus-per-task=2
#SBATCH --mem=16
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal


echo "Starting debug at: $(date)"

echo "Running on node: $SLURM_NODELIST"

echo "Moving to home"
cd /mnt/beegfs/home/leonardo.ulloa

echo "Mounting image "
sudo mount_image.py my_env.ext4 --rw

# Use single quotes for the definition to be safe
SOURCE_DATA='/mnt/beegfs/home/leonardo.ulloa/projects/bundle_chunk.tar'
DEST_DATA="$LOCAL_SCRATCH/data"

# Create the directory
mkdir -p "$DEST_DATA"

echo "Transferring data shards to local scratch..."

# Ensure we quote the variables in the command
rsync -avh --progress "$SOURCE_DATA" "$DEST_DATA/"

echo "Extracting data shards in local scratch..."

tar -xf "$DEST_DATA/bundle_chunk.tar" -C "$LOCAL_SCRATCH/data/" --checkpoint=10000 --checkpoint-action=echo="Extracted %u files"

echo "Conda INIT"
source /mnt/beegfs/home/leonardo.ulloa/miniconda3/etc/profile.d/conda.sh

echo "Activating Conda env"
conda activate scanpath

echo "Moving to project"
cd projects/EyeNet-Pipeline/

echo "Rendering cluster run config"
mkdir -p runs
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"
export LOCAL_SCRATCH RUN_DATETIME
RENDERED_CONFIG="$LOCAL_SCRATCH/eyenet_run_config.yaml"
envsubst < configs/cluster_run.yaml.template > "$RENDERED_CONFIG"

echo "Exporting WANDB_API_KEY"
# Never hardcode the key here -- it belongs in a file outside version control.
# Create it once with: echo <key> > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
export WANDB_API_KEY="$(cat ~/.wandb_api_key)"

echo "STARTING TRAINING"
python scripts/train.py --config "$RENDERED_CONFIG"

echo "Finished debug at: $(date)"

