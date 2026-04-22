#!/bin/bash
# Shared Python-env setup for every sbatch job in this folder.
# Source (don't execute) this file from an sbatch script:
#     source "$SLURM_SUBMIT_DIR/cluster_runs/_setup.sh"
#
# The calling sbatch is responsible for:
#   1. Its own #SBATCH directives.
#   2. `module load` statements (python, cuda, cudnn, etc.).
# This helper handles conda, caches, and cd-to-repo-root.

set -euo pipefail

# Resolve this script's own location (works when sourced, regardless of cwd).
_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ACTIVATE CONDA ENV
source activate multitask
echo "Active env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "which python: $(command -v python || echo 'not found')"
python -c "import pytest; print('pytest', pytest.__version__)" 2>/dev/null || {
    echo "ERROR: pytest is not installed in the '${CONDA_DEFAULT_ENV:-?}' env." >&2
    echo "Install it with:  pip install -e '.[dev,huggingface]'" >&2
    exit 1
}

# Keep HF / TF caches off $HOME to avoid quota issues.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$SCRATCH/tfhub_cache}"
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$TRANSFORMERS_CACHE" "$TFHUB_CACHE_DIR"

# Run pytest from the repo root so testpaths + relative "data/" resolve.
# Repo root = parent of the dir containing this script (cluster_runs/).
REPO_DIR="${REPO_DIR:-$(cd "$_SETUP_DIR/.." && pwd)}"
cd "$REPO_DIR"
echo "Repo:   $(pwd)"
echo "Python: $(python -V)"
python -c "import tensorflow as tf, transformers; \
print('TF', tf.__version__, '| transformers', transformers.__version__); \
print('GPUs:', tf.config.list_physical_devices('GPU'))"
