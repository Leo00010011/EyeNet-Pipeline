#!/bin/bash
#SBATCH --job-name=whole_tune
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1


# Without this, a failed `conda activate` or a crashed tune.py is silently
# ignored and the script goes on to cat a best_params.yaml that never existed.
set -euo pipefail

echo "Starting hpo at: $(date)"

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
# Guard against a silently-failed activation leaving us on base's python.
echo "Using python: $(which python)"
python -c "import optuna, torch; print('optuna', optuna.__version__)"

echo "Moving to project"
cd projects/EyeNet-Pipeline/

echo "Rendering cluster hpo config"
mkdir -p runs
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"

# The study DB and best_params.yaml must OUTLIVE the job: $LOCAL_SCRATCH is wiped
# when it ends. Keeping them on beegfs is what lets a walltime-killed job resume
# (Optuna's load_if_exists) instead of restarting the search from trial 0.
HPO_OUTPUT_DIR="runs/hpo_${RUN_DATETIME}"
HPO_STORAGE_DIR="/mnt/beegfs/home/leonardo.ulloa/projects/EyeNet-Pipeline/${HPO_OUTPUT_DIR}"
mkdir -p "$HPO_STORAGE_DIR"

# Stop the study BELOW the 16h walltime so it finishes its current trial and
# writes best_params.yaml, rather than being SIGKILLed mid-trial with nothing to
# hand off. Optuna checks the timeout between trials, so leave room for one.
OPTUNA_TIMEOUT_SECONDS=$((15 * 3600))

export LOCAL_SCRATCH RUN_DATETIME HPO_OUTPUT_DIR HPO_STORAGE_DIR OPTUNA_TIMEOUT_SECONDS
RENDERED_CONFIG="$LOCAL_SCRATCH/eyenet_hpo_config.yaml"
envsubst < configs/cluster_hpo.yaml.template > "$RENDERED_CONFIG"

echo "Exporting WANDB_API_KEY"
# Never hardcode the key here -- it belongs in a file outside version control.
# Create it once with: echo <key> > ~/.wandb_api_key && chmod 600 ~/.wandb_api_key
export WANDB_API_KEY="$(cat ~/.wandb_api_key)"

echo "STARTING HYPERPARAMETER SEARCH"
python scripts/tune.py --config "$RENDERED_CONFIG"

echo "Best params written to: ${HPO_STORAGE_DIR}/best_params.yaml"
cat "${HPO_STORAGE_DIR}/best_params.yaml"

echo "Finished hpo at: $(date)"
