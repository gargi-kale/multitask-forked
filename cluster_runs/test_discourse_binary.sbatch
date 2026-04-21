#!/bin/bash
#SBATCH --job-name=nopo-discourse-binary
#SBATCH --output=nopo-discourse-binary.out
#SBATCH --error=nopo-discourse-binary.err

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

source "$(dirname "$0")/_setup.sh"

pytest -v -s tests/test_no_prop_overlap.py::test_discourse_binary
