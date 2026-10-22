"""
Modal harness for the `multitask` library — CPU entrypoints.

    modal run modal_app.py::smoke_tests      # synthetic test suite, no data needed
    modal run modal_app.py::inspect_data     # label coverage of the uploaded CSVs

The GPU replication lives in `modal_replicate.py`, deliberately in a separate
module: `modal run` validates every function registered in the module it loads,
so a single `gpu=` function in here would make even these CPU entrypoints fail
on an account without a payment method on file.

Nothing here needs an API key. The paper encoder
(`google/bert_uncased_L-2_H-128_A-2`) is a public HuggingFace model, and the
library reads OPENAI_API_KEY / COHERE_API_KEY / VOYAGE_API_KEY / HF_API_TOKEN
lazily, only inside the API-encoder classes the paper never uses.

Setup
-----
Upload the dataset once the authors send it:

    modal volume put multitask-data ./data/no_prop_overlap_train.csv /
    modal volume put multitask-data ./data/no_prop_overlap_val.csv   /
    modal volume put multitask-data ./data/no_prop_overlap_test.csv  /
"""

import os
import sys

import modal

from modal_common import (
    SYNTHETIC_TESTS,
    VOLUMES,
    image,
    present_tests,
    pytest_cmd,
    run,
)

app = modal.App("multitask")


@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8192)
def smoke_tests(extra_args: str = "") -> int:
    """Run the synthetic-data test suite on CPU. Needs no dataset and no GPU."""
    import tensorflow as tf

    print(f"TensorFlow {tf.__version__}  |  Python {sys.version.split()[0]}")
    print(f"GPUs visible: {tf.config.list_physical_devices('GPU')}\n")

    present = present_tests()
    missing = [t for t in SYNTHETIC_TESTS if t not in present]
    if missing:
        print(f"Not on this branch, skipping: {', '.join(missing)}\n")

    cmd = pytest_cmd(present)
    if extra_args:
        cmd += extra_args.split()
    return run(cmd)


@app.function(image=image, volumes=VOLUMES, timeout=60 * 10)
def inspect_data() -> None:
    """Print shape and label coverage of the uploaded CSVs.

    Worth running before `replicate`: the recorded cluster run had WA_bi
    entirely masked and four discourse columns missing a class, which is the
    likeliest reason its macro F1 came out at 0.217.
    """
    import pandas as pd

    if not os.path.isdir("/data"):
        print("No /data volume contents yet.")
        return

    files = sorted(f for f in os.listdir("/data") if f.endswith(".csv"))
    if not files:
        print("Volume 'multitask-data' is empty — upload the CSVs first.")
        return

    for name in files:
        df = pd.read_csv(f"/data/{name}")
        print(f"\n=== {name}: {len(df):,} rows x {len(df.columns)} cols ===")
        interesting = [
            c for c in df.columns
            if c.endswith("_bi") or c in ("WA", "WA_bi", "propaganda", "text")
        ]
        for col in interesting:
            s = pd.to_numeric(
                df[col].replace(["NA", "na", "NaN", ""], pd.NA), errors="coerce"
            )
            valid = int(s.notna().sum())
            uniq = sorted(s.dropna().unique())[:5] if valid else []
            flag = "  <-- ALL MASKED" if valid == 0 else ""
            print(f"  {col:<34} valid={valid:>7,}  values={uniq}{flag}")


@app.local_entrypoint()
def main():
    """Default target: the synthetic suite, which needs no dataset."""
    rc = smoke_tests.remote()
    print(f"\npytest exit code: {rc}")
