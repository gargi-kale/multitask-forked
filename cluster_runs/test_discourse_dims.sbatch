#!/bin/bash
#SBATCH --job-name=nopo-discourse-dims
#SBATCH --output=nopo-discourse-dims.out
#SBATCH --error=nopo-discourse-dims.err

#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos ssd
#SBATCH -t 03:00:00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jgint@rcc.uchicago.edu

module load python
module load cuda/12.2
module load cudnn
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/software/cuda-12.2-el8-x86_64"

nvidia-smi
nvcc --version

# Slurm copies this script to a spool dir, so `$(dirname "$0")` is useless.
# Locate _setup.sh relative to $SLURM_SUBMIT_DIR (supports submission from
# the repo root OR from cluster_runs/).
_setup_sourced=
for _d in "$SLURM_SUBMIT_DIR/cluster_runs" "$SLURM_SUBMIT_DIR"; do
    if [[ -f "$_d/_setup.sh" ]]; then
        # shellcheck source=/dev/null
        source "$_d/_setup.sh"
        _setup_sourced=1
        break
    fi
done
if [[ -z "${_setup_sourced:-}" ]]; then
    echo "ERROR: cluster_runs/_setup.sh not found under SLURM_SUBMIT_DIR." >&2
    exit 1
fi

_repo_root="$SLURM_SUBMIT_DIR"
if [[ ! -f "${_repo_root}/tests/test_no_prop_overlap.py" && -f "${_repo_root}/multitask/tests/test_no_prop_overlap.py" ]]; then
    _repo_root="${_repo_root}/multitask"
fi
if [[ ! -f "${_repo_root}/tests/test_no_prop_overlap.py" ]]; then
    echo "ERROR: cannot locate tests/test_no_prop_overlap.py from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}." >&2
    exit 1
fi
cd "${_repo_root}"
export PYTHONPATH="${_repo_root}:${PYTHONPATH:-}"

export USE_TORCH=0
python tests/test_no_prop_overlap.py discourse_dims
