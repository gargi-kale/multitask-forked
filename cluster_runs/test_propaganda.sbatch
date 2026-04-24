#!/bin/bash
#SBATCH --job-name=nopo-propaganda
#SBATCH --output=nopo-propaganda.out
#SBATCH --error=nopo-propaganda.err

#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos ssd
#SBATCH -t 04:00:00
#SBATCH --mem=48g
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jgint@rcc.uchicago.edu

# RCC: conda needs the python module here; cluster_runs/_setup.sh then conda activates
# multitask so `python` is 3.11+ from the env (module python is only 3.9).
module load python
module load cuda/12.2
module load cudnn
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/software/cuda-12.2-el8-x86_64"

nvidia-smi
nvcc --version

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
python tests/test_no_prop_overlap.py propaganda
