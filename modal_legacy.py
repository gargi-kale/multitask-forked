"""
Modal harness for the original pre-library "multitask_basic" paper code.

    modal run modal_legacy.py::replicate_legacy

The code lives in `replication/legacy_basic/`, restored from git (`legacy/` as
of dca88b4~1) with the paper config `configs/config_iac.yaml` (from 0ef634d~1)
pointed at the `multitask-data` volume. It needs a Keras 2 era stack
(`hub.KerasLayer`, `official.nlp.optimization`, `fit(workers=...)`), so it gets
its own TF 2.15 image instead of the shared one in `modal_common`.

Metrics and the report land in the `multitask-outputs` volume under
`legacy_basic/`.
"""

import os
import shutil
import subprocess

import modal

LEGACY = "/root/legacy_basic"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "tensorflow[and-cuda]==2.15.1",
        "tensorflow-text==2.15.0",
        "tensorflow-hub==0.16.1",
        "tf-models-official==2.15.0",
        "numpy<2",
        "pandas",
        "scikit-learn",
        "pyyaml",
    )
    .env({"TFHUB_CACHE_DIR": "/cache/tfhub", "TF_CPP_MIN_LOG_LEVEL": "2", "TF_FORCE_GPU_ALLOW_GROWTH": "true"})
    .add_local_dir(
        "replication/legacy_basic",
        remote_path=LEGACY,
        ignore=["**/__pycache__/**", "results/**"],
    )
)

app = modal.App("multitask-legacy")

data_vol = modal.Volume.from_name("multitask-data")
cache_vol = modal.Volume.from_name("multitask-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("multitask-outputs", create_if_missing=True)

GPU = os.environ.get("MULTITASK_GPU", "L4")


@app.function(
    image=image,
    volumes={"/data": data_vol, "/cache": cache_vol, "/outputs": out_vol},
    gpu=GPU,
    timeout=60 * 60 * 6,
    memory=32768,
)
def replicate_legacy(config: str = "configs/config_iac_modal.yaml") -> int:
    subprocess.run(["nvidia-smi", "-L"])

    # main.py writes ./results/ and a pickle into cwd; the mount is not a
    # place to write, so run from a scratch copy.
    work = "/tmp/legacy_basic"
    shutil.copytree(LEGACY, work, dirs_exist_ok=True)
    os.makedirs(f"{work}/results", exist_ok=True)

    cmd = ["python", "main.py", config]
    print(f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=work, env={**os.environ, "PYTHONPATH": work}).returncode

    dest = "/outputs/legacy_basic"
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(f"{work}/results"):
        shutil.copy(f"{work}/results/{name}", dest)
    for name in os.listdir(work):
        if name.endswith(".pkl"):
            shutil.copy(f"{work}/{name}", dest)
    out_vol.commit()
    print(f"Copied outputs to multitask-outputs:{dest.removeprefix('/outputs')}", flush=True)
    return rc
