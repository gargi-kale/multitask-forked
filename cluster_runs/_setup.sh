#!/usr/bin/env bash
# Shared Slurm/login setup: use conda env "multitask" (Python 3.11+), not RCC module python.
#
# Note: ~/.condarc is a YAML config file (envs_dirs/pkgs_dirs), not conda's install root.
# Set CONDA_ROOT if auto-detection cannot locate your Miniforge/Miniconda base.

set -euo pipefail

if [[ -n "${CONDA_ROOT:-}" ]]; then
  _conda_root="${CONDA_ROOT}"
else
  _conda_root="$(conda info --base 2>/dev/null || true)"
fi
_conda_sh="${_conda_root}/etc/profile.d/conda.sh"
if [[ -z "${_conda_root}" || ! -f "${_conda_sh}" ]]; then
  echo "ERROR: could not locate conda.sh at ${_conda_sh:-<empty>}." >&2
  echo "Set CONDA_ROOT to your Miniforge/Miniconda install root (not ~/.condarc)." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${_conda_sh}"
conda activate multitask

if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  echo "ERROR: Python 3.11+ required; got $(python -V 2>&1) at $(command -v python). Recreate env: conda env remove -n multitask && conda env create -f environment.yml" >&2
  exit 1
fi
