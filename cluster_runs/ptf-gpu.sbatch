#!/bin/bash
#SBATCH --job-name=ptf-gpu
#SBATCH --output=ptf-gpu.out
#SBATCH --error=ptf-gpu.err

#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos ssd
#SBATCH -t 24:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=jgint@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# LOAD MODULES (Python + CUDA/cuDNN for GPU jobs)
module load python
module load cuda/12.2
module load cudnn
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/software/cuda-12.2-el8-x86_64"

nvidia-smi
nvcc --version

# ACTIVATE CONDA ENV via shared setup
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

# Checking XLA_FLAGS
echo "XLA_FLAGS=$XLA_FLAGS"
ls /software/cuda-12.2-el8-x86_64/nvvm/libdevice/

# RUN SCRIPT (PyTorch frozen encoder option removed; use universal-sentence-encoder or openai)
python -m multitask.local_test_runs.run --universal-sentence-encoder