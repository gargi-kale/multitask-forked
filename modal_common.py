"""
Shared Modal image and volume definitions for the `multitask` library.

Imported by `modal_app.py` (CPU entrypoints) and `modal_replicate.py` (GPU).
They are separate modules on purpose — see the note in `modal_app.py`.
"""

import os
import subprocess

import modal

REPO = "/root/repo"

# --------------------------------------------------------------------------
# Image
# --------------------------------------------------------------------------
# Python 3.12 because the library requires >=3.11,<3.13.
# TensorFlow carries its own CUDA wheels via the [and-cuda] extra. torch is
# installed CPU-only on purpose: it is used solely to convert the small-BERT
# PyTorch checkpoint (`from_pt=True`), never to train, and a CUDA torch build
# alongside CUDA TensorFlow invites cuDNN version conflicts.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "tensorflow[and-cuda]>=2.19,<2.20",
        "tensorflow-hub>=0.16,<1",
        "tensorflow-text>=2.19,<2.20",
        "numpy>=1.26,<2",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "scipy>=1.10",
        "transformers>=4.44,<5",
        "huggingface-hub>=0.36",
        "diskcache>=5.0",
        "pytest>=9.0",
    )
    .pip_install(
        "torch>=2.6,<3",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .env(
        {
            # Keep HF downloads on a persistent volume so repeat runs don't refetch.
            "HF_HOME": "/cache/hf",
            "TRANSFORMERS_CACHE": "/cache/hf",
            "TFHUB_CACHE_DIR": "/cache/tfhub",
            # Quieter logs; TF is very chatty at startup.
            "TF_CPP_MIN_LOG_LEVEL": "2",
        }
    )
    # Ship this module itself into the container. Modal auto-includes only the
    # module holding the App (modal_app.py / modal_replicate.py); a sibling
    # module they import is NOT included, and the copy landing under /root/repo
    # below is not on sys.path. Without this the container dies at import with
    # ModuleNotFoundError and Modal retries it — a crash loop, not a clean fail.
    .add_local_python_source("modal_common")
    # Mount the repo at runtime. Nothing here is baked into the image, so
    # editing local files takes effect on the next `modal run`.
    .add_local_dir(
        ".",
        remote_path=REPO,
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.venv/**",
            "**/htmlcov/**",
            "**/.pytest_cache/**",
            "data/**",  # data arrives via the volume, not the mount
            "**/multitask_data/**",
        ],
    )
)

data_vol = modal.Volume.from_name("multitask-data", create_if_missing=True)
cache_vol = modal.Volume.from_name("multitask-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("multitask-outputs", create_if_missing=True)

VOLUMES = {"/data": data_vol, "/cache": cache_vol, "/outputs": out_vol}


# Test files that run on purely synthetic data: no CSVs, no network, no model
# downloads. Deliberately excluded:
#   test_pooling_full.py    - reads no_prop_overlap_*.csv via csv.DictReader
#   test_encoder_*.py       - downloads from tfhub.dev / HuggingFace
#   test_hierarchical_encoder.py, test_persistence.py - same
# test_gpu_training.py self-skips when no GPU is visible, so it is a no-op on
# the CPU entrypoint and real coverage on the GPU one.
# test_regression_std.py and test_regression_extended.py are script-style: bare
# module-level asserts with no `def test_`, so pytest runs them at import and
# reports "no tests ran" on success, a collection error on failure.
SYNTHETIC_TESTS = [
    "tests/test_hierarchical.py",
    "tests/test_trainer.py",
    "tests/test_optimizers.py",
    "tests/test_distributed.py",
    "tests/test_pooling.py",
    "tests/test_pooling_hierarchical.py",
    "tests/test_pooling_trainer.py",
    "tests/test_regression_std.py",
    "tests/test_gpu_training.py",
    # testing-branch only — skipped automatically if absent
    "tests/test_task_types.py",
    "tests/test_masking.py",
    "tests/test_regression_extended.py",
    "tests/test_training_features.py",
]


def pytest_cmd(tests: list[str]) -> list[str]:
    """pytest invocation for the synthetic suite.

    `-o addopts=` blanks the addopts in pyproject.toml, which default to
    --cov=multitask --cov-report=html. Those need pytest-cov, which this image
    deliberately omits, and pytest rejects the whole run on an unrecognized
    flag -- including --no-cov, itself a pytest-cov flag.
    """
    return ["python", "-m", "pytest", *tests, "-v", "-o", "addopts=", "-p", "no:cacheprovider"]


def present_tests(repo: str = REPO) -> list[str]:
    """The SYNTHETIC_TESTS that exist on the checked-out branch."""
    return [t for t in SYNTHETIC_TESTS if os.path.exists(f"{repo}/{t}")]


def run(cmd: list[str], cwd: str = REPO) -> int:
    """Stream a subprocess straight to Modal's log output."""
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PYTHONPATH": cwd})
    return proc.returncode
