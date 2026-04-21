#!/bin/bash
# Shared setup for every sbatch job in this folder.
# Source (don't execute) this file from an sbatch script:
#     source "$(dirname "$0")/_setup.sh"
#
# Caller is expected to have already set its own #SBATCH directives.

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# LOAD MODULES (Python + CUDA/cuDNN for TF GPU)
module load python
module load cuda/12.2
module load cudnn
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/software/cuda-12.2-el8-x86_64"

nvidia-smi || true
nvcc --version || true

# ACTIVATE CONDA ENV
source activate multitask

# Keep HF / TF caches off $HOME to avoid quota issues.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$SCRATCH/tfhub_cache}"
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$TRANSFORMERS_CACHE" "$TFHUB_CACHE_DIR"

# Run pytest from the repo root so testpaths + relative "data/" resolve.
# Default: parent of this script's directory.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR/..}"
cd "$REPO_DIR"
echo "Repo:   $(pwd)"
echo "Python: $(python -V)"
python -c "import tensorflow as tf, transformers; \
print('TF', tf.__version__, '| transformers', transformers.__version__); \
print('GPUs:', tf.config.list_physical_devices('GPU'))"
