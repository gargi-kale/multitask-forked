#!/usr/bin/env bash
set -euo pipefail

# Submit every .sbatch file in cluster_runs/sbatch, except ptf-gpu.sbatch.
# Usage:
#   bash cluster_runs/sbatch/submit_all_sbatch.sh
#   bash cluster_runs/sbatch/submit_all_sbatch.sh --dry-run

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shopt -s nullglob
sbatch_files=( "${script_dir}"/*.sbatch )
shopt -u nullglob

IFS=$'\n' sbatch_files=( $(printf '%s\n' "${sbatch_files[@]}" | sort) )

if [[ ${#sbatch_files[@]} -eq 0 ]]; then
    echo "No .sbatch files found in ${script_dir}"
    exit 0
fi

submitted=0
skipped=0

for sbatch_file in "${sbatch_files[@]}"; do
    sbatch_name="$(basename "${sbatch_file}")"

    if [[ "${sbatch_name}" == "ptf-gpu.sbatch" ]]; then
        echo "Skipping ${sbatch_name}"
        ((skipped += 1))
        continue
    fi

    if [[ ${dry_run} -eq 1 ]]; then
        echo "[dry-run] sbatch ${sbatch_file}"
    else
        echo "Submitting ${sbatch_name}"
        sbatch "${sbatch_file}"
    fi
    ((submitted += 1))
done

echo "Done. Submitted: ${submitted}, Skipped: ${skipped}"
