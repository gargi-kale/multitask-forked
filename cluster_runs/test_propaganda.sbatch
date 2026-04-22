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

# Slurm copies this script to a spool dir, so `$(dirname "$0")` is useless.
# Locate _setup.sh relative to $SLURM_SUBMIT_DIR (supports submission from
# the repo root OR from cluster_runs/).
for _d in "$SLURM_SUBMIT_DIR/cluster_runs" "$SLURM_SUBMIT_DIR"; do
    if [[ -f "$_d/_setup.sh" ]]; then source "$_d/_setup.sh"; break; fi
done

python -m pytest -v -s tests/test_no_prop_overlap.py::test_propaganda
