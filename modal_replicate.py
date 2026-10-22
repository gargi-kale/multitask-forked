"""
Modal harness for the `multitask` library — GPU paper replication.

    modal run modal_replicate.py::replicate
    modal run modal_replicate.py::replicate --experiment discourse_binary --epochs 5

Requires a payment method on your Modal account (GPU functions are gated behind
one even when you have credits; credits are still spent first). The CPU-only
entrypoints in `modal_app.py` work without one.

Prerequisites
-------------
1. The dataset, in the `multitask-data` volume:

       modal volume put multitask-data ./data/no_prop_overlap_train.csv /
       modal volume put multitask-data ./data/no_prop_overlap_val.csv   /

2. The replication script. On `main` it exists only in git history; on the
   `testing` branch it is checked in:

       mkdir -p replication
       git show origin/testing:tests/test_no_prop_overlap.py \
           > replication/test_no_prop_overlap.py

   This function looks in `replication/` first, then `tests/`.

No API key is needed: the encoder is the public `google/bert_uncased_L-2_H-128_A-2`.
"""

import os

import modal

from modal_common import (
    REPO,
    SYNTHETIC_TESTS,
    VOLUMES,
    image,
    out_vol,
    present_tests,
    pytest_cmd,
    run,
)

app = modal.App("multitask-replicate")

# L4 is ample for a 4.4M-parameter encoder. The cluster run measured 49 ms/step
# on an A100-class card at 1,712 steps/epoch (~84 s/epoch); expect roughly
# 3-4 min/epoch here. Switch to "A100" to close that gap.
GPU = os.environ.get("MULTITASK_GPU", "L4")


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 45, memory=16384)
def smoke_tests_gpu(extra_args: str = "") -> int:
    """The synthetic suite on a GPU container, to prove the training path works
    on CUDA before a real run. Same tests as `modal_app.py::smoke_tests`, but
    the four `test_gpu_training.py` cases that self-skip on CPU actually
    execute here: device detection, strategy selection, a one-epoch fit under
    the strategy, and prediction shapes.

        modal run modal_replicate.py::smoke_tests_gpu
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}  |  GPUs: {gpus}", flush=True)
    if not gpus:
        raise RuntimeError(
            "No GPU visible — the GPU-specific tests would silently skip, "
            "defeating the point of this entrypoint."
        )

    present = present_tests()
    missing = [t for t in SYNTHETIC_TESTS if t not in present]
    if missing:
        print(f"Not on this branch, skipping: {', '.join(missing)}\n")

    cmd = pytest_cmd(present)
    if extra_args:
        cmd += extra_args.split()
    return run(cmd)


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu=GPU,
    timeout=60 * 60 * 6,
    memory=32768,
)
def replicate(
    experiment: str = "paper_replication",
    encoder: str = "small-bert",
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    verbose: int = 2,
    data_dir: str = "/data",
    max_length: int = 512,
    seed: int = 42,
    tag: str = "",
    dropout: float = 0.4,
    pooled_loss: bool = False,
    weight_decay: float = -1.0,
    warmup_percent: float = -1.0,
    arch: str = "paper",
) -> int:
    """Run a replication experiment on a GPU.

    `data_dir` is a path inside the container; point it at a subdirectory of
    the `multitask-data` volume (e.g. /data/paper) to pick a dataset version.
    `max_length` defaults to the paper config's seq_length of 512. Metrics for
    `paper_replication` are written to the `multitask-outputs` volume as
    `<experiment>[_<tag>].json`.

    Defaults match `configs/config_iac.yaml` and the script's
    PAPER_TRAINING_CONFIG (lr 3e-4, batch 256, dropout 0.4, weight decay 0.01,
    5% warmup, adamw). `dropout` overrides the paper's 0.4 for
    `paper_replication`.
    `pooled_loss` trains the propaganda branch on the max-pooled output
    instead of the 18 technique heads (`--pooled-propaganda-loss`).
    `weight_decay` / `warmup_percent` override the paper's 0.01 / 5 when
    non-negative; `arch` picks the head sizes ("paper", "medium", "wide",
    "legacy" — see PAPER_ARCHITECTURES in the script).
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}  |  GPUs: {gpus}", flush=True)
    if not gpus:
        print("WARNING: no GPU visible — this will be extremely slow.", flush=True)

    # Locate the replication script: recovered copy first, then testing branch.
    candidates = [
        f"{REPO}/replication/test_no_prop_overlap.py",
        f"{REPO}/tests/test_no_prop_overlap.py",
    ]
    script = next((p for p in candidates if os.path.exists(p)), None)
    if script is None:
        raise FileNotFoundError(
            "Replication script not found. Recover it with:\n"
            "  mkdir -p replication && git show origin/testing:tests/"
            "test_no_prop_overlap.py > replication/test_no_prop_overlap.py"
        )
    print(f"Using {script}", flush=True)

    # Confirm the dataset is present before burning GPU time.
    required = [
        f"{data_dir}/no_prop_overlap_train.csv",
        f"{data_dir}/no_prop_overlap_val.csv",
        f"{data_dir}/no_prop_overlap_test.csv",
    ]
    absent = [p for p in required if not os.path.exists(p)]
    if absent:
        have = os.listdir(data_dir) if os.path.isdir(data_dir) else []
        raise FileNotFoundError(
            f"Missing from the multitask-data volume: {absent}\n"
            f"{data_dir} currently contains: {have}\n"
            "Upload with: modal volume put multitask-data "
            "./data/no_prop_overlap_train.csv /"
        )

    rc = run([
        "python", script, experiment,
        "--encoder", encoder,
        "--data-dir", data_dir,
        "--max-length", str(max_length),
        "--seed", str(seed),
        "--dropout", str(dropout),
        "--results-json", f"/outputs/{experiment}{'_' + tag if tag else ''}.json",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--verbose", str(verbose),
        "--arch", arch,
        *(["--weight-decay", str(weight_decay)] if weight_decay >= 0 else []),
        *(["--warmup-percent", str(warmup_percent)] if warmup_percent >= 0 else []),
        *(["--pooled-propaganda-loss"] if pooled_loss else []),
    ])
    out_vol.commit()
    return rc


# --------------------------------------------------------------------------
# Hyperparameter sweep (paper_replication with the pooled propaganda loss)
# --------------------------------------------------------------------------
# The paper's grid (commented out in legacy main.py) was lr {1.5e-4, 3e-4},
# decay {0.01, 0.05, 0.1}, warmup {5, 20}, dropout {0.1, 0.25, 0.4},
# batch {32, 64}. This space contains all of it, plus wider values and the
# head size. It is too big to run in full (4,608 points), so stage 1 samples
# it at random; stage 2 re-runs the best stage-1 settings on more seeds.
#
#     modal run --detach modal_replicate.py::sweep --stage 1 --n-random 80
#     modal run --detach modal_replicate.py::sweep --stage 2 --top 5 --seeds 1,2,3
#     modal run modal_replicate.py::sweep_report
#
# Each run writes /outputs/paper_replication_sweep/<tag>.json. Runs whose JSON
# already exists are skipped, so a sweep can be relaunched to resume it.
# Settings are ranked on the validation split only; test is reported, never
# used for selection.

SWEEP_DIR = "paper_replication_sweep"
SWEEP_SPACE = {
    "learning_rate": [5e-5, 1e-4, 1.5e-4, 3e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 0.01, 0.05, 0.1],
    "warmup_percent": [5, 10, 20],
    "dropout": [0.0, 0.1, 0.25, 0.4],
    "batch_size": [32, 64, 128, 256],
    "arch": ["paper", "medium", "wide", "legacy"],
}
# The ten outputs of the paper's Table 3 (PAPER_TABLE3_MULTITASK_F1 in the
# script, which can't be imported here without TensorFlow).
TABLE3_OUTPUTS = [
    "propaganda_max_pool", "disagree_agree_bi", "emotion_fact_bi",
    "attacking_respectful_bi", "nasty_nice_bi", "personal_audience_bi",
    "defeater_undercutter_bi", "negotiate_attack_bi",
    "questioning_asserting_bi", "WA_bi",
]
PAPER_TABLE3_MEAN = 66.7


def _sweep_name(cfg: dict, seed: int) -> str:
    return (
        f"lr{cfg['learning_rate']:g}_wd{cfg['weight_decay']:g}"
        f"_wu{cfg['warmup_percent']:g}_d{cfg['dropout']:g}"
        f"_b{cfg['batch_size']}_{cfg['arch']}_s{seed}"
    )


def _load_sweep() -> list[dict]:
    """Every finished sweep run on the outputs volume, with its mean F1/AUC."""
    import json

    try:
        entries = out_vol.listdir(SWEEP_DIR)
    except Exception:
        return []
    runs = []
    for entry in entries:
        if not entry.path.endswith(".json"):
            continue
        d = json.loads(b"".join(out_vol.read_file(entry.path)))
        a = d["args"]
        row = {
            "name": entry.path.rsplit("/", 1)[-1][:-5],
            "seed": a["seed"],
            "cfg": {
                "learning_rate": a["learning_rate"],
                "weight_decay": a["weight_decay"],
                "warmup_percent": a["warmup_percent"],
                "dropout": a["dropout"],
                "batch_size": a["batch_size"],
                "arch": a["arch"],
            },
        }
        for split in ("val", "test"):
            merged = {**d[split]["task_metrics"], **d[split]["pool_metrics"]}
            row[f"{split}_f1"] = 100 * sum(merged[k]["f1"] for k in TABLE3_OUTPUTS) / 10
            row[f"{split}_auc"] = 100 * sum(merged[k]["auc"] for k in TABLE3_OUTPUTS) / 10
        runs.append(row)
    return runs


def _launch(jobs: list[tuple[dict, int]]) -> None:
    done = {r["name"] for r in _load_sweep()}
    todo = [(cfg, seed) for cfg, seed in jobs if _sweep_name(cfg, seed) not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for cfg, seed in todo:
        name = _sweep_name(cfg, seed)
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2,
            tag=f"sweep/{name}", data_dir="/data/paper", **cfg,
        )))
    failed = []
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)
        if rc != 0:
            failed.append(name)
    print(f"Finished: {len(calls) - len(failed)} ok, {len(failed)} failed. "
          "Relaunch the same command to retry failures.")


@app.local_entrypoint()
def sweep(stage: int = 1, n_random: int = 80, sample_seed: int = 0,
          top: int = 5, seeds: str = "1,2,3", only: str = "", skip: str = ""):
    """Stage 1: the paper's settings at every head size, plus `n_random`
    random points of SWEEP_SPACE, all at seed 42. Stage 2: the `top` stage-1
    settings by validation F1, re-run on `seeds`. `only` restricts stage 1 to
    the comma-separated run names given (for a timing check); `skip` leaves
    out run names still in flight from an earlier launch."""
    import itertools
    import random

    if stage == 1:
        paper = {"learning_rate": 3e-4, "weight_decay": 0.01, "warmup_percent": 5,
                 "dropout": 0.0, "batch_size": 256}
        cfgs = [{**paper, "arch": arch} for arch in SWEEP_SPACE["arch"]]
        keys = list(SWEEP_SPACE)
        space = [dict(zip(keys, v)) for v in itertools.product(*SWEEP_SPACE.values())]
        rng = random.Random(sample_seed)
        cfgs += [c for c in rng.sample(space, n_random) if c not in cfgs]
        jobs = [(c, 42) for c in cfgs]
        if only:
            wanted = set(only.split(","))
            jobs = [j for j in jobs if _sweep_name(*j) in wanted]
    elif stage == 2:
        stage1 = sorted((r for r in _load_sweep() if r["seed"] == 42),
                        key=lambda r: r["val_f1"], reverse=True)
        if not stage1:
            raise SystemExit("No stage-1 results on the volume yet.")
        jobs = [(r["cfg"], int(s)) for r in stage1[:top] for s in seeds.split(",")]
    else:
        raise SystemExit("--stage must be 1 or 2")
    if skip:
        jobs = [j for j in jobs if _sweep_name(*j) not in set(skip.split(","))]
    _launch(jobs)


@app.local_entrypoint()
def sweep_report(show: int = 20, csv_path: str = "replication/sweep_results.csv"):
    """Rank settings by mean validation F1 over their seeds; write every run
    to `csv_path`."""
    import csv
    import statistics

    runs = _load_sweep()
    if not runs:
        raise SystemExit("No sweep results on the volume yet.")
    with open(csv_path, "w", newline="") as f:
        cols = ["name", "seed", *runs[0]["cfg"], "val_f1", "val_auc", "test_f1", "test_auc"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(runs, key=lambda r: r["val_f1"], reverse=True):
            w.writerow({k: r[k] for k in cols if k in r} | r["cfg"])
    print(f"{len(runs)} runs written to {csv_path}\n")

    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(_sweep_name(r["cfg"], 0)[:-3], []).append(r)

    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else float("nan")

    rows = []
    for key, rs in groups.items():
        v, t = [r["val_f1"] for r in rs], [r["test_f1"] for r in rs]
        rows.append((statistics.mean(v), key, len(rs), sd(v), statistics.mean(t), sd(t),
                     statistics.mean(r["test_auc"] for r in rs)))
    rows.sort(reverse=True)
    print(f"Top {show} settings by mean validation F1 (paper Table 3 mean: {PAPER_TABLE3_MEAN})")
    print(f"{'setting':<46}{'n':>3}{'val F1':>9}{'±':>6}{'test F1':>9}{'±':>6}{'test AUC':>10}")
    for vm, key, n, vs, tm, ts, auc in rows[:show]:
        print(f"{key:<46}{n:>3}{vm:>9.2f}{vs:>6.2f}{tm:>9.2f}{ts:>6.2f}{auc:>10.2f}")
    multi = [r for r in rows if r[2] > 1]
    if multi:
        vm, key, n, vs, tm, ts, auc = multi[0]
        print(f"\nBest setting run on several seeds: {key} "
              f"(n={n}): val {vm:.2f}, test {tm:.2f} ± {ts:.2f}, AUC {auc:.2f}")
