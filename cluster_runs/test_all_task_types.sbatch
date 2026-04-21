#!/bin/bash
#SBATCH --job-name=nopo-all-task-types
#SBATCH --output=nopo-all-task-types.out
#SBATCH --error=nopo-all-task-types.err

#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos ssd
#SBATCH -t 08:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jgint@rcc.uchicago.edu

source "$(dirname "$0")/_setup.sh"

pytest -v -s tests/test_no_prop_overlap.py::test_all_task_types
