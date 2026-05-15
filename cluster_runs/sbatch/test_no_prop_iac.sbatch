#!/bin/bash
#SBATCH --job-name=nopo-iac-grid
#SBATCH --output=cluster_runs/outputs/%x-%j.bootstrap.out
#SBATCH --error=cluster_runs/outputs/%x-%j.bootstrap.err

#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos ssd
#SBATCH -t 24:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jgint@rcc.uchicago.edu

_repo_root="$SLURM_SUBMIT_DIR"
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${_repo_root}/tests/test_no_prop_iac.py" ]]; then
    _candidate_root="$(cd "${_script_dir}/../.." && pwd)"
    if [[ -f "${_candidate_root}/tests/test_no_prop_iac.py" ]]; then
        _repo_root="${_candidate_root}"
    fi
fi
if [[ ! -f "${_repo_root}/tests/test_no_prop_iac.py" && -f "${_repo_root}/multitask/tests/test_no_prop_iac.py" ]]; then
    _repo_root="${_repo_root}/multitask"
fi
if [[ ! -f "${_repo_root}/tests/test_no_prop_iac.py" ]]; then
    echo "ERROR: cannot locate tests/test_no_prop_iac.py from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}." >&2
    exit 1
fi
_outputs_dir="${_repo_root}/cluster_runs/outputs"
mkdir -p "${_outputs_dir}"
_job_start_ts="$(date '+%Y%m%d-%H%M%S')"
_job_log_base="${SLURM_JOB_NAME:-job}-${_job_start_ts}"
exec >"${_outputs_dir}/${_job_log_base}.out" 2>"${_outputs_dir}/${_job_log_base}.err"

module load python
module load cuda/12.2
module load cudnn
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/software/cuda-12.2-el8-x86_64"

nvidia-smi
nvcc --version

_setup_sourced=
for _d in "$_repo_root/cluster_runs" "$_repo_root"; do
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

cd "${_repo_root}"
export PYTHONPATH="${_repo_root}:${PYTHONPATH:-}"

export USE_TORCH=0
python tests/test_no_prop_iac.py
