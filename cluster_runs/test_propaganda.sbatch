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

source "$(dirname "$0")/_setup.sh"

pytest -v -s tests/test_no_prop_overlap.py::test_propaganda
