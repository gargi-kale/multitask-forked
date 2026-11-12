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
    data_vol,
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
# A comma-separated list is a fallback order: each run takes the first type
# with capacity, e.g. MULTITASK_GPU=H100,A100-80GB,L40S,L4.
GPU = [g.strip() for g in os.environ.get("MULTITASK_GPU", "L4").split(",")]
GPU = GPU[0] if len(GPU) == 1 else GPU


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
    task_type_grouping: bool = True,
    logit_adjust: bool = False,
    focal_gamma: float = 0.0,
    save_predictions: bool = False,
    loss_balancer: str = "",
    mlm_lambda: float = -1.0,
    mlm_mode: str = "fixed",
    tasks: str = "",
    drop_unlabeled_train: bool = False,
    grad_probe_every: int = 0,
    early_stop_metric: str = "",
    early_stop_patience: int = 3,
    main_task: str = "",
    self_distill: bool = False,
    self_distill_patience: int = 2,
    self_distill_signal: str = "f1",
    propaganda_zero_fill: bool = False,
    zero_fill_keep_class_weights: bool = False,
    stratify_batches: bool = False,
    masked_label_weight: float = 0.0,
    consistency_weight: float = 0.0,
    mask_sweep: str = "",
    mlm_mask_prob: float = 0.15,
    consistency_keep: float = 0.0,
    consistency_floor: float = 0.0,
    consistency_teacher: str = "",
    consistency_student_dropout: bool = True,
    ema_decay: float = 0.99,
    semi_supervised: bool = False,
    ssl_rates: str = "",
    ssl_rate_weight: float = 0.1,
    ssl_entropy_weight: float = 0.0,
    ssl_entropy_start_epoch: float = 2.0,
    propaganda_pool: str = "max",
    technique_loss_weight: float = 0.0,
    mask_news_discourse_train: bool = False,
    no_rule_labels: bool = False,
    hide_task: str = "",
    probe_tasks: str = "",
    unweighted_loss: bool = False,
    probe_untrained: bool = False,
    label_share: str = "",
    repeat_train: int = 1,
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
    `task_type_grouping=False` weights the loss per head instead of per
    task-type (`--no-task-type-grouping`).
    `logit_adjust` / `focal_gamma` change the Disagree/Agree loss
    (`--logit-adjust`, `--focal-gamma`; both need `pooled_loss`).
    `save_predictions` also writes `<experiment>[_<tag>]_predictions.npz`.
    `loss_balancer` ("famo" or "dbmtl") replaces the fixed task-type weights
    (`--loss-balancer`; needs `pooled_loss`).
    `mlm_lambda` >= 0 adds that weight x a masked-word loss on the encoder
    (`--mlm-lambda`; needs `pooled_loss`; 0 runs the same code with no effect).
    `mlm_mode` ("fixed", "task", "cosine") picks how that loss is weighted
    alongside `loss_balancer` (`--mlm-mode`).
    `tasks` trains a comma-separated subset of the ten outputs (`--tasks`),
    `drop_unlabeled_train` drops train rows with none of their labels, and
    `grad_probe_every` logs per-task gradient metrics every N steps.
    `early_stop_metric` ("loss", "f1", "auc") scores val and test after every
    epoch and keeps the best epoch by that val metric, stopping once all three
    have stalled for `early_stop_patience` epochs (`--early-stop-metric`).
    `propaganda_zero_fill` trains the propaganda heads with label 0 on the
    other sources' rows (`--propaganda-zero-fill`);
    `zero_fill_keep_class_weights` computes its class weights without the filled
    labels (`--zero-fill-keep-class-weights`).
    `mask_news_discourse_train` masks the 8 IAC labels on Propaganda rows in
    train only (`--mask-news-discourse-train`); `no_rule_labels` masks them in
    train, val and test (`--no-rule-labels`).
    `hide_task` hides one Table-3 output on every train row (`--hide-task`),
    and `probe_tasks` fits linear probes after training (`--probe-tasks`).
    `stratify_batches` keeps each train batch to the training set's mix of
    sources, labelled tasks and classes (`--stratify-batches`).
    `masked_label_weight` / `consistency_weight` > 0 also send the masked copy
    through the whole model, scored against the real labels or against the
    full-sentence answers (`--masked-label-weight`, `--consistency-weight`;
    both need `loss_balancer="famo"` and `mlm_lambda`).
    `mask_sweep` (comma-separated rates) scores val on masked copies after
    training (`--mask-sweep`; needs `early_stop_metric`).
    `mlm_mask_prob` sets the share of words hidden in the masked copy
    (`--mlm-mask-prob`, default 0.15); `consistency_keep` > 0 keeps only
    confident consistency targets, that share per side per head
    (`--consistency-keep`), and `consistency_floor` > 0 also requires those
    scores to be within that distance of 0 or 1 (`--consistency-floor`).
    `consistency_teacher` ("current" or "ema") takes the consistency target
    from a dropout-off teacher instead of a masked copy (`--consistency-teacher`;
    needs `loss_balancer="famo"`, no `mlm_lambda`), with the student pass's
    dropout on or off (`consistency_student_dropout`) and the EMA decay
    `ema_decay`.
    `semi_supervised` adds the rate and entropy terms on foreign cells
    (`--semi-supervised`; needs `pooled_loss`, no balancer or masking), with
    `ssl_rates` (head@source=rate list; "" = the script's defaults),
    `ssl_rate_weight`, `ssl_entropy_weight` and `ssl_entropy_start_epoch`.
    `propaganda_pool` ("max", "noisy_or", "lse") picks how the 18 technique
    heads are pooled (`--propaganda-pool`; needs `pooled_loss` unless max), and
    `technique_loss_weight` > 0 adds that weight x the per-technique loss to
    the pooled term (`--technique-loss-weight`; needs `pooled_loss`).
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
        *([] if task_type_grouping else ["--no-task-type-grouping"]),
        *(["--logit-adjust"] if logit_adjust else []),
        *(["--focal-gamma", str(focal_gamma)] if focal_gamma else []),
        *(["--save-predictions"] if save_predictions else []),
        *(["--loss-balancer", loss_balancer] if loss_balancer else []),
        *(["--mlm-lambda", str(mlm_lambda)] if mlm_lambda >= 0 else []),
        *(["--mlm-mode", mlm_mode] if mlm_mode != "fixed" else []),
        *(["--tasks", tasks] if tasks else []),
        *(["--drop-unlabeled-train"] if drop_unlabeled_train else []),
        *(["--grad-probe-every", str(grad_probe_every)] if grad_probe_every else []),
        *(["--early-stop-metric", early_stop_metric,
           "--early-stop-patience", str(early_stop_patience)] if early_stop_metric else []),
        *(["--main-task", main_task] if main_task else []),
        *(["--self-distill", "--self-distill-patience", str(self_distill_patience),
           "--self-distill-signal", self_distill_signal] if self_distill else []),
        *(["--propaganda-zero-fill"] if propaganda_zero_fill else []),
        *(["--zero-fill-keep-class-weights"] if zero_fill_keep_class_weights else []),
        *(["--stratify-batches"] if stratify_batches else []),
        *(["--mask-news-discourse-train"] if mask_news_discourse_train else []),
        *(["--no-rule-labels"] if no_rule_labels else []),
        *(["--unweighted-loss"] if unweighted_loss else []),
        *(["--hide-task", hide_task] if hide_task else []),
        *(["--probe-tasks", probe_tasks] if probe_tasks else []),
        *(["--probe-untrained"] if probe_untrained else []),
        *(["--label-share", label_share] if label_share else []),
        *(["--repeat-train", str(repeat_train)] if repeat_train > 1 else []),
        *(["--masked-label-weight", str(masked_label_weight)] if masked_label_weight else []),
        *(["--consistency-weight", str(consistency_weight)] if consistency_weight else []),
        *(["--mask-sweep", mask_sweep] if mask_sweep else []),
        *(["--mlm-mask-prob", str(mlm_mask_prob)] if mlm_mask_prob != 0.15 else []),
        *(["--consistency-keep", str(consistency_keep)] if consistency_keep else []),
        *(["--consistency-floor", str(consistency_floor)] if consistency_floor else []),
        *(["--propaganda-pool", propaganda_pool] if propaganda_pool != "max" else []),
        *(["--technique-loss-weight", str(technique_loss_weight)] if technique_loss_weight else []),
        *(["--consistency-teacher", consistency_teacher] if consistency_teacher else []),
        *([] if consistency_student_dropout else ["--consistency-student-dropout", "off"]),
        *(["--ema-decay", str(ema_decay)] if ema_decay != 0.99 else []),
        *(["--semi-supervised", "--ssl-rate-weight", str(ssl_rate_weight),
           "--ssl-entropy-weight", str(ssl_entropy_weight),
           "--ssl-entropy-start-epoch", str(ssl_entropy_start_epoch)]
          if semi_supervised else []),
        *(["--ssl-rates", ssl_rates] if semi_supervised and ssl_rates else []),
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
PAPER_TABLE3_MEAN = 66.74  # mean of the 10 multi-task F1s in Table 3 (667.36 / 10)


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


# --------------------------------------------------------------------------
# The sweep winner without task-type grouping, on the same four seeds as the
# winner's sweep numbers (42 from stage 1, 1-3 from stage 2).
#
#     modal run --detach modal_replicate.py::ungrouped
#
# Writes /outputs/paper_replication_ungrouped/<name>.json, kept apart from the
# sweep so sweep_report is unaffected.

WINNER = {"learning_rate": 3e-4, "weight_decay": 0.01, "warmup_percent": 5,
          "dropout": 0.0, "batch_size": 256, "arch": "wide"}


@app.local_entrypoint()
def ungrouped(seeds: str = "42,1,2,3"):
    import json
    import statistics

    calls = []
    for seed in (int(s) for s in seeds.split(",")):
        name = _sweep_name(WINNER, seed)
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, task_type_grouping=False,
            tag=f"ungrouped/{name}", data_dir="/data/paper", **WINNER,
        )))
    f1s, aucs = [], []
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)
        if rc != 0:
            continue
        d = json.loads(b"".join(out_vol.read_file(
            f"paper_replication_ungrouped/{name}.json")))
        merged = {**d["test"]["task_metrics"], **d["test"]["pool_metrics"]}
        f1s.append(100 * sum(merged[k]["f1"] for k in TABLE3_OUTPUTS) / 10)
        aucs.append(100 * sum(merged[k]["auc"] for k in TABLE3_OUTPUTS) / 10)
        print(f"     test F1 {f1s[-1]:.2f}  AUC {aucs[-1]:.2f}", flush=True)
    if len(f1s) > 1:
        print(f"Ungrouped: test F1 {statistics.mean(f1s):.1f} ± {statistics.stdev(f1s):.1f}, "
              f"AUC {statistics.mean(aucs):.1f} ± {statistics.stdev(aucs):.2f} "
              f"({len(f1s)} seeds). Grouped winner: 66.2 ± 1.0 / 68.3 ± 0.65; "
              f"paper {PAPER_TABLE3_MEAN}.")


# --------------------------------------------------------------------------
# Disagree/Agree imbalance ablation: the sweep winner with class weights
# (baseline), focal loss, logit adjustment, and both, on the winner's four
# seeds. Every variant is re-run, so all four share the GPU drift and report
# macro F1.
#
#     modal run --detach modal_replicate.py::imbalance
#
# Writes /outputs/paper_replication_imbalance/<variant>_<name>.json.

IMBALANCE_VARIANTS = {
    "baseline": {},
    "focal": {"focal_gamma": 2.0},
    "logit_adj": {"logit_adjust": True},
    "both": {"logit_adjust": True, "focal_gamma": 2.0},
}
PAPER_DISAGREE_AGREE_F1 = 71.38


@app.local_entrypoint()
def imbalance(seeds: str = "42,1,2,3", gamma: float = 2.0, variants: str = ""):
    import json
    import statistics

    chosen = variants.split(",") if variants else list(IMBALANCE_VARIANTS)
    calls = []
    for variant in chosen:
        opts = {k: (gamma if k == "focal_gamma" else v)
                for k, v in IMBALANCE_VARIANTS[variant].items()}
        for seed in (int(s) for s in seeds.split(",")):
            name = f"{variant}_{_sweep_name(WINNER, seed)}"
            calls.append((variant, name, replicate.spawn(
                seed=seed, pooled_loss=True, verbose=2,
                tag=f"imbalance/{name}", data_dir="/data/paper", **WINNER, **opts,
            )))
    rows: dict = {v: [] for v in chosen}
    for variant, name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)
        if rc != 0:
            continue
        d = json.loads(b"".join(out_vol.read_file(
            f"paper_replication_imbalance/{name}.json")))
        merged = {**d["test"]["task_metrics"], **d["test"]["pool_metrics"]}
        da = merged["disagree_agree_bi"]
        rows[variant].append((100 * da["f1"], 100 * da["f1_macro"], 100 * da["auc"],
                              100 * sum(merged[k]["f1"] for k in TABLE3_OUTPUTS) / 10))

    def ms(xs):
        return (f"{statistics.mean(xs):5.1f} ± {statistics.stdev(xs):4.1f}"
                if len(xs) > 1 else f"{xs[0]:5.1f}       " if xs else "    –       ")

    print(f"\nTest set, Disagree/Agree (paper Table 3 F1 {PAPER_DISAGREE_AGREE_F1}; "
          f"mean over 10 outputs, paper {PAPER_TABLE3_MEAN})")
    print(f"{'variant':<10} {'n':>2}  {'weighted F1':<13} {'macro F1':<13} "
          f"{'AUC':<13} {'mean of 10 F1':<13} {'Paper (Table 3 F1)':>18}")
    for variant in chosen:
        r = rows[variant]
        cols = [ms([x[i] for x in r]) for i in range(4)]
        print(f"{variant:<10} {len(r):>2}  " + " ".join(f"{c:<13}" for c in cols)
              + f" {PAPER_DISAGREE_AGREE_F1:>18}")


# --------------------------------------------------------------------------
# The sweep winner on its four seeds, saving every val/test prediction so
# cut-off rules and seed ensembles can be tried offline.
#
#     modal run --detach modal_replicate.py::predictions
#
# Writes /outputs/paper_replication_predictions/<name>.json and
# <name>_predictions.npz.


@app.local_entrypoint()
def predictions(seeds: str = "42,1,2,3"):
    calls = []
    for seed in (int(s) for s in seeds.split(",")):
        name = _sweep_name(WINNER, seed)
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            tag=f"predictions/{name}", data_dir="/data/paper", **WINNER,
        )))
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)
    print("Download with: modal volume get multitask-outputs "
          "paper_replication_predictions/ ./predictions/")


# --------------------------------------------------------------------------
# Adaptive task balancing: the sweep winner with FAMO or DB-MTL in place of
# the fixed task-type weights, on the winner's four seeds, saving predictions.
# Compared with the `predictions` runs (same settings, fixed weights).
#
#     modal run --detach modal_replicate.py::balancers
#
# Writes /outputs/paper_replication_balancers/<method>_<name>.json (+ .npz).

# Paper Table 3 multi-task weighted F1, as in the replication script.
TABLE3_LABELS = {
    "propaganda_max_pool": ("Propaganda", 61.74),
    "disagree_agree_bi": ("Disagree/Agree", 71.38),
    "emotion_fact_bi": ("Emotion/Fact", 63.93),
    "attacking_respectful_bi": ("Attacking/Respectful", 68.07),
    "nasty_nice_bi": ("Nasty/Nice", 73.69),
    "personal_audience_bi": ("Personal/Audience", 65.69),
    "defeater_undercutter_bi": ("Defeater/Undercutter", 55.65),
    "negotiate_attack_bi": ("Negotiate/Attack", 64.81),
    "questioning_asserting_bi": ("Questioning/Asserting", 63.23),
    "WA_bi": ("Argument Quality", 79.17),
}


# --------------------------------------------------------------------------
# Shared-depth study: the winner's training settings with the encoder
# fine-tuned, varying only the layer layout (DEPTH_ARCHS, defined in the
# replication script's PAPER_ARCHITECTURES). "wide" is s2_t2_k2.
#
#     modal run --detach modal_replicate.py::depth
#
# Writes /outputs/paper_replication_depth/<arch>_s<seed>.json.

DEPTH_ARCHS = [
    "wide", "s1_t2_k2", "s3_t2_k2", "s4_t2_k2", "s6_t2_k2",
    "s2_t2_k2_narrow", "s2_t2_k2_wider",
    "s3_t1_k2", "s3_t2_k1", "s4_t1_k1", "s1_t3_k2",
    "s2_t1_k1", "s2_t3_k3",
]


@app.local_entrypoint()
def depth(seeds: str = "42,1,2", archs: str = ""):
    done = set()
    try:
        done = {e.path.rsplit("/", 1)[-1][:-5] for e in out_vol.listdir("paper_replication_depth")
                if e.path.endswith(".json")}
    except Exception:
        pass
    calls = []
    for arch in (archs.split(",") if archs else DEPTH_ARCHS):
        for seed in (int(s) for s in seeds.split(",")):
            name = f"{arch}_s{seed}"
            if name in done:
                continue
            cfg = {**WINNER, "arch": arch}
            calls.append((name, replicate.spawn(
                seed=seed, pooled_loss=True, verbose=2,
                tag=f"depth/{name}", data_dir="/data/paper", **cfg,
            )))
    print(f"Launching {len(calls)} runs ({len(done)} already done).", flush=True)
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def balancers(seeds: str = "42,1,2,3", methods: str = "famo,dbmtl"):
    import json
    import statistics

    calls = []
    for method in methods.split(","):
        for seed in (int(s) for s in seeds.split(",")):
            name = f"{method}_{_sweep_name(WINNER, seed)}"
            calls.append((method, name, replicate.spawn(
                seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
                loss_balancer=method, tag=f"balancers/{name}",
                data_dir="/data/paper", **WINNER,
            )))
    for _, name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)

    def load(path):
        try:
            d = json.loads(b"".join(out_vol.read_file(path)))
        except Exception:  # missing or failed run
            return None
        return {**d["test"]["task_metrics"], **d["test"]["pool_metrics"]}

    seed_list = [int(s) for s in seeds.split(",")]
    runs = {"fixed weights": [load(f"paper_replication_predictions/{_sweep_name(WINNER, s)}.json")
                              for s in seed_list]}
    for method in methods.split(","):
        runs[method] = [load(f"paper_replication_balancers/{method}_{_sweep_name(WINNER, s)}.json")
                        for s in seed_list]
    runs = {k: [r for r in v if r] for k, v in runs.items()}

    def ms(xs):
        if not xs:
            return "–"
        return f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"

    cols = list(runs)
    print("\nTest weighted F1 x100, mean ± sd over seeds "
          f"(n = {', '.join(f'{k} {len(v)}' for k, v in runs.items())})")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{c:>18}" for c in cols))
    for key, (label, paper_f1) in TABLE3_LABELS.items():
        print(f"{label:<24}{paper_f1:>20.2f}"
              + "".join(f"{ms([100 * r[key]['f1'] for r in runs[c]]):>18}" for c in cols))
    for metric, title in (("f1", "Mean F1 over 10"), ("auc", "Mean AUC over 10")):
        paper = f"{PAPER_TABLE3_MEAN:.2f}" if metric == "f1" else "–"
        print(f"{title:<24}{paper:>20}" + "".join(
            f"{ms([100 * sum(r[k][metric] for k in TABLE3_OUTPUTS) / 10 for r in runs[c]]):>18}"
            for c in cols))


# --------------------------------------------------------------------------
# Masked-word auxiliary loss: total = supervised + lambda x MLM loss on the
# encoder, with the sweep winner's settings (encoder fine-tuned).
#
#     modal run modal_replicate.py::masked_lm_tests       # CPU, needs no GPU
#     modal run --detach modal_replicate.py::masked_lm
#     modal run modal_replicate.py::masked_lm_report
#
# Stage 1: every lambda in MLM_LAMBDAS on seeds 42 and 1, plus lambda 0 (the
# baseline through the same code) on seeds 2 and 3. Stage 2, launched as soon
# as stage 1 is done: the best lambda > 0 by mean validation F1 on seeds 2, 3.
# Writes /outputs/paper_replication_mlm/lam<lambda>_s<seed>.json (+ .npz).
# Runs whose JSON already exists are skipped, so relaunching resumes.

MLM_DIR = "paper_replication_mlm"
MLM_LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0]


@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, memory=8192)
def masked_lm_tests(extra_args: str = "") -> int:
    cmd = pytest_cmd(["replication/test_masked_lm_loss.py",
                      "replication/test_mlm_balancing.py",
                      "replication/test_gradient_probe.py",
                      "replication/test_epoch_scorer.py",
                      "replication/test_zero_fill.py",
                      "replication/test_stratify_batches.py",
                      "replication/test_label_aware_masking.py",
                      "replication/test_teacher_consistency.py",
                      "replication/test_semi_supervised.py",
                      "replication/test_pooled_output_loss.py",
                      "replication/test_mask_news_labels.py",
                      "replication/test_hide_one_label.py",
                      "replication/test_self_distill.py"])
    return run(cmd + (extra_args.split() if extra_args else []))


def _mlm_runs() -> dict:
    """{(lambda, seed): run JSON} for every finished masked-word run."""
    import json

    try:
        entries = [e.path for e in out_vol.listdir(MLM_DIR) if e.path.endswith(".json")]
    except Exception:
        return {}
    runs = {}
    for path in entries:
        d = json.loads(b"".join(out_vol.read_file(path)))
        runs[(float(d["args"]["mlm_lambda"]), int(d["args"]["seed"]))] = d
    return runs


def _mean10(d: dict, split: str, metric: str) -> float:
    merged = {**d[split]["task_metrics"], **d[split]["pool_metrics"]}
    return 100 * sum(merged[k][metric] for k in TABLE3_OUTPUTS) / 10


def _mlm_launch(jobs: list[tuple[float, int]]) -> None:
    done = set(_mlm_runs())
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for lam, seed in todo:
        name = f"lam{lam:g}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            mlm_lambda=lam, tag=f"mlm/{name}", data_dir="/data/paper", **WINNER,
        )))
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def masked_lm(stage: int = 0, lambdas: str = "", only: str = ""):
    """`stage` 0 runs stage 1 then stage 2; 1 or 2 runs just that stage.
    `only` = "lam<lambda>_s<seed>,..." runs exactly those (e.g. a smoke run)."""
    import statistics

    if only:
        jobs = [(float(n.split("_s")[0][3:]), int(n.split("_s")[1])) for n in only.split(",")]
        return _mlm_launch(jobs)
    grid = [float(x) for x in lambdas.split(",")] if lambdas else MLM_LAMBDAS
    if stage in (0, 1):
        _mlm_launch([(lam, s) for lam in grid for s in (42, 1)] + [(0.0, 2), (0.0, 3)])
    if stage in (0, 2):
        runs = _mlm_runs()
        val = {lam: [_mean10(runs[(lam, s)], "val", "f1") for s in (42, 1) if (lam, s) in runs]
               for lam in grid if lam > 0}
        val = {lam: v for lam, v in val.items() if len(v) == 2}
        if not val:
            print("No lambda > 0 has both stage-1 seeds; not launching stage 2.")
            return
        for lam, v in sorted(val.items()):
            print(f"  lambda {lam:g}: mean val F1 {statistics.mean(v):.2f} (seeds 42, 1)")
        best = max(val, key=lambda lam: statistics.mean(val[lam]))
        print(f"Best lambda on validation: {best:g}", flush=True)
        _mlm_launch([(best, 2), (best, 3)])


@app.local_entrypoint()
def masked_lm_report():
    import statistics

    runs = _mlm_runs()
    lams = sorted({lam for lam, _ in runs})

    def ms(xs):
        if not xs:
            return "–"
        return f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"

    def merged(d):
        return {**d["test"]["task_metrics"], **d["test"]["pool_metrics"]}

    print("\nRuns (lambda: seeds):", {lam: sorted(s for l, s in runs if l == lam) for lam in lams})
    print("\nMean over 10 outputs, x100, mean ± sd over seeds")
    print(f"{'lambda':<8}{'n':>3}{'val F1':>14}{'test F1':>14}{'test AUC':>14}"
          f"{'Paper (Table 3 F1)':>20}")
    for lam in lams:
        ds = [d for (l, _), d in runs.items() if l == lam]
        print(f"{lam:<8g}{len(ds):>3}{ms([_mean10(d, 'val', 'f1') for d in ds]):>14}"
              f"{ms([_mean10(d, 'test', 'f1') for d in ds]):>14}"
              f"{ms([_mean10(d, 'test', 'auc') for d in ds]):>14}{PAPER_TABLE3_MEAN:>20.2f}")
    print("\nTest weighted F1 per task, x100")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{f'lam {l:g}':>14}" for l in lams))
    for key, (label, paper_f1) in TABLE3_LABELS.items():
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(
            f"{ms([100 * merged(d)[key]['f1'] for (l, _), d in runs.items() if l == lam]):>14}"
            for lam in lams))
    print("\nPer-epoch losses (train sup / train MLM / val sup / val MLM acc):")
    for (lam, seed), d in sorted(runs.items()):
        h = d.get("history", {})
        rows = zip(*(h.get(k, []) for k in ("loss", "mlm_loss", "val_loss", "val_mlm_acc")))
        print(f"  lam {lam:g} s{seed}: " + "  ".join(
            f"[{a:.3f} {b:.2f} {c:.3f} {e:.2f}]" for a, b, c, e in rows))


# --------------------------------------------------------------------------
# Masked-word loss with loss balancing, all fine-tuned, sweep winner settings:
#   famo / dbmtl              - balancer over the 10 tasks, no masked-word loss
#   famo_fixed / dbmtl_fixed  - Option A: that + MLM_BALANCE_LAMBDA x MLM loss
#   famo_task / dbmtl_task    - Option B: MLM loss as an 11th balanced task
#   cosine                    - MLM_BALANCE_LAMBDA x MLM gradient, only on steps
#                               where it agrees (cosine > 0) with the tasks'
# Compared with lam0 / lam0.03 from paper_replication_mlm (same seeds).
#
#     modal run --detach modal_replicate.py::mlm_balance
#     modal run modal_replicate.py::mlm_balance_report
#
# Writes /outputs/paper_replication_mlm_balance/<config>_s<seed>.json (+ .npz);
# finished runs are skipped on relaunch.

MLM_BALANCE_DIR = "paper_replication_mlm_balance"
MLM_BALANCE_LAMBDA = 0.03  # best lambda on validation in the masked_lm study
MLM_BALANCE_CONFIGS = {  # name: (loss_balancer, mlm_lambda, mlm_mode)
    "famo": ("famo", -1.0, "fixed"),
    "dbmtl": ("dbmtl", -1.0, "fixed"),
    "famo_fixed": ("famo", MLM_BALANCE_LAMBDA, "fixed"),
    "dbmtl_fixed": ("dbmtl", MLM_BALANCE_LAMBDA, "fixed"),
    "famo_task": ("famo", 1.0, "task"),  # lambda is unused in task mode
    "dbmtl_task": ("dbmtl", 1.0, "task"),
    "cosine": ("", MLM_BALANCE_LAMBDA, "cosine"),
}


def _done(directory: str) -> set:
    try:
        return {e.path.rsplit("/", 1)[-1][:-5] for e in out_vol.listdir(directory)
                if e.path.endswith(".json")}
    except Exception:
        return set()


@app.local_entrypoint()
def mlm_balance(seeds: str = "42,1,2,3", configs: str = ""):
    names = configs.split(",") if configs else list(MLM_BALANCE_CONFIGS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(MLM_BALANCE_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        balancer, lam, mode = MLM_BALANCE_CONFIGS[config]
        name = f"{config}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer=balancer, mlm_lambda=lam, mlm_mode=mode,
            tag=f"mlm_balance/{name}", data_dir="/data/paper", **WINNER,
        )))
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def mlm_balance_report():
    import json
    import statistics

    from scipy import stats

    def load(path):
        return json.loads(b"".join(out_vol.read_file(path)))

    runs = {}  # config: {seed: run}
    for lam in (0.0, MLM_BALANCE_LAMBDA):
        runs[f"lam{lam:g}"] = {s: d for (l, s), d in _mlm_runs().items() if l == lam}
    for name in sorted(_done(MLM_BALANCE_DIR)):
        config, seed = name.rsplit("_s", 1)
        runs.setdefault(config, {})[int(seed)] = load(f"{MLM_BALANCE_DIR}/{name}.json")
    order = ["lam0", f"lam{MLM_BALANCE_LAMBDA:g}", *MLM_BALANCE_CONFIGS]
    runs = {c: runs[c] for c in order if runs.get(c)}

    def ms(xs):
        if not xs:
            return "–"
        return f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"

    def paired(a, b, split="test", metric="f1"):
        seeds = sorted(set(runs[a]) & set(runs[b]))
        d = [_mean10(runs[a][s], split, metric) - _mean10(runs[b][s], split, metric) for s in seeds]
        if len(d) < 2:
            return "–"
        m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        h = stats.t.ppf(0.975, len(d) - 1) * se
        return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}] n={len(d)}"

    print("\nMean over 10 outputs, x100, mean ± sd over seeds")
    print(f"{'config':<14}{'n':>3}{'val F1':>14}{'test F1':>14}{'test AUC':>14}"
          f"{'Paper (Table 3 F1)':>20}{'ΔF1 vs lam0':>26}{'ΔAUC vs lam0':>26}")
    for c, rs in runs.items():
        ds = list(rs.values())
        print(f"{c:<14}{len(ds):>3}{ms([_mean10(d, 'val', 'f1') for d in ds]):>14}"
              f"{ms([_mean10(d, 'test', 'f1') for d in ds]):>14}"
              f"{ms([_mean10(d, 'test', 'auc') for d in ds]):>14}{PAPER_TABLE3_MEAN:>20.2f}"
              f"{paired(c, 'lam0') if c != 'lam0' else '':>26}"
              f"{paired(c, 'lam0', metric='auc') if c != 'lam0' else '':>26}")
    print("\nPaired test F1 differences (row - column), 95% t CI")
    for a, b in (("famo_fixed", "famo"), ("dbmtl_fixed", "dbmtl"), ("famo_task", "famo"),
                 ("dbmtl_task", "dbmtl"), ("famo_fixed", "lam0.03"), ("dbmtl_fixed", "lam0.03"),
                 ("cosine", "lam0.03"), ("famo", "lam0"), ("dbmtl", "lam0")):
        if a in runs and b in runs:
            print(f"  {a:<12} - {b:<8} F1 {paired(a, b)}   AUC {paired(a, b, metric='auc')}")

    print("\nTest weighted F1 per task, x100")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{c:>14}" for c in runs))
    for key, (label, paper_f1) in TABLE3_LABELS.items():
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(
            f"{ms([100 * {**d['test']['task_metrics'], **d['test']['pool_metrics']}[key]['f1'] for d in rs.values()]):>14}"
            for rs in runs.values()))

    print("\nPer-epoch training logs (MLM loss / val MLM acc / extra):")
    for c, rs in runs.items():
        if c.startswith("lam") or c in ("famo", "dbmtl"):
            continue
        for seed, d in sorted(rs.items()):
            h = d.get("history", {})
            extra = [k for k in ("mlm_weight", "mlm_cos", "mlm_gate") if k in h]
            rows = zip(h.get("mlm_loss", []), h.get("val_mlm_acc", []), *(h[k] for k in extra))
            print(f"  {c} s{seed} [{', '.join(['mlm_loss', 'val_mlm_acc', *extra])}]: " + "  ".join(
                "[" + " ".join(f"{v:.3f}" for v in r) + "]" for r in rows))


# --------------------------------------------------------------------------
# Negative-transfer study: does multi-task learning help each task, and
# which tasks help or hurt which? All runs fine-tune the encoder with the
# sweep winner's settings, from scratch, on TRANSFER_SEEDS:
#   full  - all ten outputs, logging gradient metrics and task affinity
#   only_<t> - single-task baseline for output t
#   minus_<t> - leave-one-task-out: every output but t
# Every run drops train rows with none of its labels, so the full model and
# the subsets are treated alike.
#
# plan="reduced" (default): only_<t> on every seed, plus `full` on seeds 4
# and 42. The multi-task side of each seed's delta is the lambda-0 run of the
# masked-word study (same settings, same training) for seeds 42, 1, 2, 3 and
# full_s4 for seed 4. LOTO runs come later, for the pairs that stand out:
#     modal run --detach modal_replicate.py::transfer --plan all --only minus_WA_bi --seeds 42,1,2
# plan="all": every variant on every seed (105 runs).
#
#     modal run --detach modal_replicate.py::transfer
#     modal run modal_replicate.py::transfer_download && python replication/transfer_analysis.py
#
# Writes /outputs/paper_replication_transfer/<variant>_s<seed>.json. Runs whose
# JSON already exists are skipped, so relaunching resumes.

TRANSFER_DIR = "paper_replication_transfer"
TRANSFER_SEEDS = [42, 1, 2, 3, 4]
GRAD_PROBE_EVERY = 400


def _transfer_variants() -> dict:
    variants = {"full": ""}
    for t in TABLE3_OUTPUTS:
        variants[f"only_{t}"] = t
        variants[f"minus_{t}"] = ",".join(o for o in TABLE3_OUTPUTS if o != t)
    return variants


def _transfer_done() -> set:
    try:
        return {e.path.rsplit("/", 1)[-1][:-5] for e in out_vol.listdir(TRANSFER_DIR)
                if e.path.endswith(".json")}
    except Exception:
        return set()


@app.local_entrypoint()
def transfer(plan: str = "reduced", seeds: str = "", only: str = ""):
    """`only` = comma-separated variant names (e.g. "full,only_WA_bi")."""
    seed_list = [int(s) for s in seeds.split(",")] if seeds else TRANSFER_SEEDS
    variants = _transfer_variants()
    if only:
        variants = {k: v for k, v in variants.items() if k in only.split(",")}
    if plan == "reduced" and not only:
        wanted = [(v, s) for v in variants if v.startswith("only_") for s in seed_list]
        wanted += [("full", 4), ("full", 42)]
    elif plan in ("reduced", "all"):
        wanted = [(v, s) for v in variants for s in seed_list]
    else:
        raise SystemExit(f"Unknown plan {plan!r}")
    done = _transfer_done()
    jobs = [(v, s) for v, s in wanted if f"{v}_s{s}" not in done]
    print(f"{len(wanted)} runs requested, {len(wanted) - len(jobs)} already done, "
          f"launching {len(jobs)}.", flush=True)
    calls = []
    for variant, seed in jobs:
        name = f"{variant}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, tasks=variants[variant],
            drop_unlabeled_train=True,
            grad_probe_every=GRAD_PROBE_EVERY if variant == "full" else 0,
            tag=f"transfer/{name}", data_dir="/data/paper", **WINNER,
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
def transfer_download(out_dir: str = "replication/transfer_runs"):
    """Copy every finished transfer-run JSON, and the masked-word study's
    lambda-0 runs (as mtl_lam0_s<seed>.json), to `out_dir` for
    replication/transfer_analysis.py."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [(f"{TRANSFER_DIR}/{n}.json", f"{n}.json") for n in sorted(_transfer_done())]
    try:
        paths += [(e.path, "mtl_" + e.path.rsplit("/", 1)[-1]) for e in out_vol.listdir(MLM_DIR)
                  if e.path.rsplit("/", 1)[-1].startswith("lam0_s") and e.path.endswith(".json")]
    except Exception:
        pass
    for src, dst in paths:
        with open(f"{out_dir}/{dst}", "wb") as f:
            f.write(b"".join(out_vol.read_file(src)))
    print(f"Downloaded {len(paths)} runs to {out_dir}")


# --------------------------------------------------------------------------
# Fine-tuned learning-rate / stopping-rule study. The sweep winner was picked
# with a frozen encoder (before the Keras 3 fix), so this re-checks the
# learning rate with the encoder training, and compares stopping rules.
# Plain multi-task model, sweep winner settings except the learning rate.
# Every run scores val and test after each epoch and runs until val loss, F1
# and AUC have all stalled for RETUNE_PATIENCE epochs (max 10, the schedule's
# length), so each stopping rule is replayed from the same run: the epoch is
# always picked on val, and test is read at that epoch.
#
#     modal run --detach modal_replicate.py::retune
#     modal run modal_replicate.py::retune_report
#
# Writes /outputs/paper_replication_retune/lr<lr>_s<seed>.json (+ .npz);
# finished runs are skipped on relaunch.

RETUNE_DIR = "paper_replication_retune"
RETUNE_LRS = [3e-4, 1e-4, 5e-5, 2e-5]
RETUNE_PATIENCE = 3
# name: (val metric, patience). "loss p2" is the paper's rule, used so far.
RETUNE_RULES = {"loss p2": ("loss", 2), "loss p3": ("loss", 3), "F1 p2": ("f1", 2),
                "F1 p3": ("f1", 3), "AUC p3": ("auc", 3)}


@app.local_entrypoint()
def retune(seeds: str = "42,1,2,3", lrs: str = ""):
    lr_list = [float(x) for x in lrs.split(",")] if lrs else RETUNE_LRS
    jobs = [(lr, int(s)) for lr in lr_list for s in seeds.split(",")]
    done = _done(RETUNE_DIR)
    todo = [(lr, s) for lr, s in jobs if f"lr{lr:g}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for lr, seed in todo:
        name = f"lr{lr:g}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"retune/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": lr},
        )))
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


def _replay(records: list, metric: str, patience: int) -> dict:
    """The epoch record an early-stopping rule on val `metric` would keep."""
    key = "val_loss" if metric == "loss" else f"val_{metric}"
    sign = 1 if metric == "loss" else -1
    best = 0
    for i, r in enumerate(records):
        if sign * r[key] < sign * records[best][key]:
            best = i
        if i - best >= patience:
            break
    return records[best]


@app.local_entrypoint()
def retune_report():
    import json
    import statistics

    from scipy import stats

    runs = {}  # lr: {seed: run}
    for name in sorted(_done(RETUNE_DIR)):
        d = json.loads(b"".join(out_vol.read_file(f"{RETUNE_DIR}/{name}.json")))
        runs.setdefault(float(d["args"]["learning_rate"]), {})[int(d["args"]["seed"])] = d
    if not runs:
        raise SystemExit("No finished runs yet")

    def ms(xs):
        return (f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1
                else f"{xs[0]:.1f}")

    def picked(lr, rule):
        metric, patience = RETUNE_RULES[rule]
        return {s: _replay(d["epoch_scores"], metric, patience) for s, d in runs[lr].items()}

    base = picked(3e-4, "loss p2") if 3e-4 in runs else None
    print("\nMean over the 10 Table-3 outputs, x100, mean ± sd over seeds; the epoch is "
          "picked on val by each rule, test is read at that epoch.")
    print(f"{'lr':<8}{'rule':<10}{'n':>3}{'epoch':>7}{'val F1':>13}{'test F1':>13}{'test AUC':>13}"
          f"{'Paper (Table 3 F1)':>20}{'ΔF1 vs 3e-4 loss p2':>28}")
    for lr in sorted(runs, reverse=True):
        for rule in RETUNE_RULES:
            p = picked(lr, rule)
            diff = ""
            if base is not None and not (lr == 3e-4 and rule == "loss p2"):
                seeds = sorted(set(p) & set(base))
                d = [100 * (p[s]["test_f1"] - base[s]["test_f1"]) for s in seeds]
                if len(d) > 1:
                    m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
                    h = stats.t.ppf(0.975, len(d) - 1) * se
                    diff = f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]"
            print(f"{lr:<8g}{rule:<10}{len(p):>3}"
                  f"{statistics.mean(r['epoch'] for r in p.values()):>7.1f}"
                  f"{ms([100 * r['val_f1'] for r in p.values()]):>13}"
                  f"{ms([100 * r['test_f1'] for r in p.values()]):>13}"
                  f"{ms([100 * r['test_auc'] for r in p.values()]):>13}"
                  f"{PAPER_TABLE3_MEAN:>20.2f}{diff:>28}")
        print()

    print("Per-epoch curves, mean over seeds (val loss / val F1 / test F1):")
    for lr in sorted(runs, reverse=True):
        recs = [d["epoch_scores"] for d in runs[lr].values()]
        cells = []
        for e in range(max(len(r) for r in recs)):
            at = [r[e] for r in recs if len(r) > e]
            cells.append(f"e{e + 1}[{statistics.mean(x['val_loss'] for x in at):.3f} "
                         f"{100 * statistics.mean(x['val_f1'] for x in at):.1f} "
                         f"{100 * statistics.mean(x['test_f1'] for x in at):.1f} n={len(at)}]")
        print(f"  lr {lr:g}: " + " ".join(cells))

    best_lr = max(runs, key=lambda lr: statistics.mean(
        r["val_f1"] for r in picked(lr, "F1 p3").values()))
    rule = "F1 p3"
    print(f"\nBest lr on val (F1 p3): {best_lr:g}. Test weighted F1 per task, x100:")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}{'3e-4 loss p2':>16}{f'{best_lr:g} {rule}':>16}")
    best = picked(best_lr, rule)
    for key, (label, paper_f1) in TABLE3_LABELS.items():
        cols = [ms([100 * r["test"][key]["f1"] for r in sel.values()])
                for sel in (base or {}, best) if sel]
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(f"{c:>16}" for c in cols))


# --------------------------------------------------------------------------
# Balancers and the masked-word loss under the re-tuned setup: lr RETUNE_FAMO_LR and the epoch picked on
# val F1 (EpochScorer, patience RETUNE_PATIENCE), compared with the plain
# model from `retune` at the same lr, seeds and stopping rule.
#
#     modal run --detach modal_replicate.py::retune_famo
#     modal run modal_replicate.py::retune_famo_report
#
# Writes /outputs/paper_replication_retune_famo/<config>_s<seed>.json (+ .npz).

RETUNE_FAMO_DIR = "paper_replication_retune_famo"
RETUNE_FAMO_LR = 1e-4  # best lr on val F1 in `retune`
RETUNE_FAMO_CONFIGS = {
    **{k: MLM_BALANCE_CONFIGS[k] for k in ("famo", "famo_task", "famo_fixed", "dbmtl")},
    "mlm_fixed": ("", MLM_BALANCE_LAMBDA, "fixed"),  # fixed task weights + lambda x MLM
}


@app.local_entrypoint()
def retune_famo(seeds: str = "42,1,2,3", configs: str = ""):
    names = configs.split(",") if configs else list(RETUNE_FAMO_CONFIGS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        balancer, lam, mode = RETUNE_FAMO_CONFIGS[config]
        name = f"{config}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer=balancer, mlm_lambda=lam, mlm_mode=mode,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def retune_famo_report(rule: str = "F1 p3"):
    import json
    import statistics

    from scipy import stats

    def load(path):
        return json.loads(b"".join(out_vol.read_file(path)))

    metric, patience = RETUNE_RULES[rule]
    runs = {"plain": {}}  # config: {seed: picked epoch record}
    for name in _done(RETUNE_DIR):
        d = load(f"{RETUNE_DIR}/{name}.json")
        if float(d["args"]["learning_rate"]) == RETUNE_FAMO_LR:
            runs["plain"][int(d["args"]["seed"])] = _replay(d["epoch_scores"], metric, patience)
    for name in sorted(_done(RETUNE_FAMO_DIR)):
        config, seed = name.rsplit("_s", 1)
        d = load(f"{RETUNE_FAMO_DIR}/{name}.json")
        runs.setdefault(config, {})[int(seed)] = _replay(d["epoch_scores"], metric, patience)

    def ms(xs):
        return (f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1
                else f"{xs[0]:.1f}")

    def paired(a, b, key):
        seeds = sorted(set(runs[a]) & set(runs[b]))
        d = [100 * (runs[a][s][key] - runs[b][s][key]) for s in seeds]
        if len(d) < 2:
            return "–"
        m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        h = stats.t.ppf(0.975, len(d) - 1) * se
        return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]"

    print(f"\nlr {RETUNE_FAMO_LR:g}, epoch picked by '{rule}' on val; mean over the 10 "
          "Table-3 outputs, x100, mean ± sd over seeds")
    print(f"{'config':<12}{'n':>3}{'epoch':>7}{'val F1':>13}{'test F1':>13}{'test AUC':>13}"
          f"{'Paper (Table 3 F1)':>20}{'ΔF1 vs plain':>22}{'ΔAUC vs plain':>22}")
    for c, rs in runs.items():
        recs = list(rs.values())
        print(f"{c:<12}{len(recs):>3}{statistics.mean(r['epoch'] for r in recs):>7.1f}"
              f"{ms([100 * r['val_f1'] for r in recs]):>13}"
              f"{ms([100 * r['test_f1'] for r in recs]):>13}"
              f"{ms([100 * r['test_auc'] for r in recs]):>13}{PAPER_TABLE3_MEAN:>20.2f}"
              f"{paired(c, 'plain', 'test_f1') if c != 'plain' else '':>22}"
              f"{paired(c, 'plain', 'test_auc') if c != 'plain' else '':>22}")
    pairs = [(c, c.rsplit("_", 1)[0]) for c in runs
             if c.endswith(("_zf", "_zfw", "_strat")) and c.rsplit("_", 1)[0] in runs]
    if pairs:
        print("\nZero-fill or stratified batches vs the same config without it "
              "(paired by seed, 95% t CI)")
        for a, b in pairs:
            seeds = sorted(set(runs[a]) & set(runs[b]))
            prop = {m: statistics.mean(100 * (runs[a][s]["test"]["propaganda_max_pool"][m]
                                              - runs[b][s]["test"]["propaganda_max_pool"][m])
                                       for s in seeds) for m in ("f1", "auc")}
            print(f"  {a:<14} - {b:<10} mean F1 {paired(a, b, 'test_f1')}   mean AUC "
                  f"{paired(a, b, 'test_auc')}   Propaganda F1 {prop['f1']:+.1f} AUC {prop['auc']:+.1f}")
    print("\nTest weighted F1 per task, x100")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{c:>14}" for c in runs))
    for key, (label, paper_f1) in TABLE3_LABELS.items():
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(
            f"{ms([100 * r['test'][key]['f1'] for r in rs.values()]):>14}"
            for rs in runs.values()))


# Propaganda zero-fill: IAC and IBM rows count as "no technique" for the 18
# propaganda heads in train (val/test still scored on propaganda rows only),
# under the re-tuned setup. Runs go to RETUNE_FAMO_DIR as <config>_zf_s<seed>,
# so retune_famo_report lists them beside the same configs without the fill.
# keep_weights=True keeps the original class weights, to separate the new
# negatives from the reweighting; those runs are <config>_zfw_s<seed>.
#
#     modal run --detach modal_replicate.py::zero_fill
#     modal run --detach modal_replicate.py::zero_fill --keep-weights --configs plain
#     modal run modal_replicate.py::retune_famo_report

ZERO_FILL_CONFIGS = {"plain": ("", -1.0, "fixed"), "famo_fixed": MLM_BALANCE_CONFIGS["famo_fixed"]}


@app.local_entrypoint()
def zero_fill(seeds: str = "42,1,2,3", configs: str = "", keep_weights: bool = False):
    names = configs.split(",") if configs else list(ZERO_FILL_CONFIGS)
    suffix = "zfw" if keep_weights else "zf"
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_{suffix}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        balancer, lam, mode = ZERO_FILL_CONFIGS[config]
        name = f"{config}_{suffix}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer=balancer, mlm_lambda=lam, mlm_mode=mode,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            propaganda_zero_fill=True, zero_fill_keep_class_weights=keep_weights,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


# Stratified batches: each train batch keeps the training set's mix of sources,
# labelled tasks and classes, under the re-tuned setup. Runs go to RETUNE_FAMO_DIR
# as <config>_strat_s<seed>, so retune_famo_report lists them beside the same
# configs with shuffled batches.
#
#     modal run --detach modal_replicate.py::stratify
#     modal run modal_replicate.py::retune_famo_report

STRATIFY_CONFIGS = ZERO_FILL_CONFIGS


@app.local_entrypoint()
def stratify(seeds: str = "42,1,2,3", configs: str = ""):
    names = configs.split(",") if configs else list(STRATIFY_CONFIGS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_strat_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        balancer, lam, mode = STRATIFY_CONFIGS[config]
        name = f"{config}_strat_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer=balancer, mlm_lambda=lam, mlm_mode=mode,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            stratify_batches=True,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


# Label-aware masking: famo_fixed plus the masked copy sent through the whole
# model, scored against the real labels ("mlab", --masked-label-weight) or
# against the model's own full-sentence answers on every row ("cons",
# --consistency-weight). Stage 1 tries both weights on seeds 42 and 1; stage 2
# runs the best weight per option on val F1 with seeds 2 and 3. Runs go to
# RETUNE_FAMO_DIR as famo_fixed_<option><weight>_s<seed>, next to famo_fixed.
#
#     modal run modal_replicate.py::masked_lm_tests
#     modal run modal_replicate.py::label_aware_smoke
#     modal run --detach modal_replicate.py::label_aware --stage 1
#     modal run --detach modal_replicate.py::label_aware --stage 2
#     modal run modal_replicate.py::label_aware_report

LABEL_AWARE_OPTIONS = {"mlab": "masked_label_weight", "cons": "consistency_weight"}
LABEL_AWARE_WEIGHTS = [0.3, 1.0]
LABEL_AWARE_STAGE_SEEDS = {1: [42, 1], 2: [2, 3]}


def _label_aware_kwargs(option: str, weight: float) -> dict:
    balancer, lam, mode = MLM_BALANCE_CONFIGS["famo_fixed"]
    return dict(pooled_loss=True, verbose=2, save_predictions=True,
                loss_balancer=balancer, mlm_lambda=lam, mlm_mode=mode,
                early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
                data_dir="/data/paper", **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
                **{LABEL_AWARE_OPTIONS[option]: weight})


def _label_aware_name(option: str, weight: float) -> str:
    return f"famo_fixed_{option}{weight:g}"


def _spawn_all(calls: list) -> None:
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def label_aware_smoke(seed: int = 42):
    """One epoch of each option at weight 1, outside the results directory,
    to check time per epoch and memory."""
    _spawn_all([(f"smoke_{o}", replicate.spawn(
        seed=seed, tag=f"label_aware_smoke/{_label_aware_name(o, 1.0)}_s{seed}",
        **{**_label_aware_kwargs(o, 1.0), "epochs": 1})) for o in LABEL_AWARE_OPTIONS])


def _label_aware_val_f1(names: list) -> dict:
    """{run name: val F1 of the epoch the F1 p3 rule keeps}."""
    import json

    out = {}
    for name in names:
        d = json.loads(b"".join(out_vol.read_file(f"{RETUNE_FAMO_DIR}/{name}.json")))
        out[name] = _replay(d["epoch_scores"], "f1", RETUNE_PATIENCE)["val_f1"]
    return out


@app.local_entrypoint()
def label_aware(stage: int = 1, seeds: str = ""):
    """`seeds` overrides the stage's seeds, e.g. `--stage 1 --seeds 42,1,2,3`
    runs every weight on all four seeds at once instead of staging."""
    done = _done(RETUNE_FAMO_DIR)
    if stage == 1:
        configs = [(o, w) for o in LABEL_AWARE_OPTIONS for w in LABEL_AWARE_WEIGHTS]
    elif stage == 2:
        configs = []
        for o in LABEL_AWARE_OPTIONS:
            names = {w: [f"{_label_aware_name(o, w)}_s{s}" for s in LABEL_AWARE_STAGE_SEEDS[1]]
                     for w in LABEL_AWARE_WEIGHTS}
            missing = [n for ns in names.values() for n in ns if n not in done]
            if missing:
                raise SystemExit(f"Stage 1 is not finished: missing {missing}")
            f1 = _label_aware_val_f1([n for ns in names.values() for n in ns])
            mean = {w: sum(f1[n] for n in ns) / len(ns) for w, ns in names.items()}
            best = max(mean, key=mean.get)
            print(f"{o}: mean val F1 over seeds {LABEL_AWARE_STAGE_SEEDS[1]} "
                  + ", ".join(f"weight {w:g} {100 * v:.2f}" for w, v in mean.items())
                  + f" -> weight {best:g}", flush=True)
            configs.append((o, best))
    else:
        raise SystemExit("--stage must be 1 or 2")
    seed_list = [int(s) for s in seeds.split(",")] if seeds else LABEL_AWARE_STAGE_SEEDS[stage]
    jobs = [(o, w, s) for o, w in configs for s in seed_list]
    todo = [(o, w, s) for o, w, s in jobs if f"{_label_aware_name(o, w)}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for option, weight, seed in todo:
        name = f"{_label_aware_name(option, weight)}_s{seed}"
        calls.append((name, replicate.spawn(seed=seed, tag=f"retune_famo/{name}",
                                            **_label_aware_kwargs(option, weight))))
    _spawn_all(calls)


@app.local_entrypoint()
def label_aware_report(rule: str = "F1 p3", baseline: str = "famo_fixed",
                       prefixes: str = "famo_fixed_mlab,famo_fixed_cons"):
    """Label-aware runs against famo_fixed: mean scores, paired differences,
    test F1 per task beside the paper, Propaganda, and how spread out each
    head's val scores are (a collapse toward one value means a lazy head)."""
    import io
    import json
    import statistics

    import numpy as np
    from scipy import stats

    def load(path):
        return b"".join(out_vol.read_file(path))

    metric, patience = RETUNE_RULES[rule]
    prefixes = tuple(prefixes.split(","))
    runs, spread = {}, {}  # config: {seed: picked epoch record}, {seed: {output: sd}}
    kept = {}  # config: {seed: mean share of cells the consistency term used}
    for name in sorted(_done(RETUNE_FAMO_DIR)):
        config, seed = name.rsplit("_s", 1)
        if config != baseline and not config.startswith(prefixes):
            continue
        d = json.loads(load(f"{RETUNE_FAMO_DIR}/{name}.json"))
        runs.setdefault(config, {})[int(seed)] = _replay(d["epoch_scores"], metric, patience)
        if d.get("history", {}).get("consistency_kept"):
            kept.setdefault(config, {})[int(seed)] = statistics.mean(d["history"]["consistency_kept"])
        try:
            z = np.load(io.BytesIO(load(f"{RETUNE_FAMO_DIR}/{name}_predictions.npz")))
            spread.setdefault(config, {})[int(seed)] = {
                k: float(np.std(z[f"val/prob/{k}"])) for k in TABLE3_OUTPUTS}
        except Exception:
            pass
    if baseline not in runs:
        raise SystemExit(f"No {baseline} runs found")
    configs = [baseline] + sorted(c for c in runs if c != baseline)

    def ms(xs):
        return (f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1
                else f"{xs[0]:.1f}")

    def paired(c, get):
        seeds = sorted(set(runs[c]) & set(runs[baseline]))
        d = [100 * (get(runs[c][s]) - get(runs[baseline][s])) for s in seeds]
        if len(d) < 2:
            return "–"
        m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        h = stats.t.ppf(0.975, len(d) - 1) * se
        return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]"

    print(f"\nEpoch picked by '{rule}' on val; mean over the 10 Table-3 outputs, x100, "
          "mean ± sd over seeds; Δ is paired by seed with a 95% t interval")
    print(f"{'config':<22}{'n':>3}{'epoch':>7}{'val F1':>13}{'test F1':>13}{'test AUC':>13}"
          f"{'Paper (Table 3 F1)':>20}{'ΔF1 vs ' + baseline:>22}{'ΔAUC vs ' + baseline:>22}")
    for c in configs:
        recs = list(runs[c].values())
        base = c == baseline
        print(f"{c:<22}{len(recs):>3}{statistics.mean(r['epoch'] for r in recs):>7.1f}"
              f"{ms([100 * r['val_f1'] for r in recs]):>13}"
              f"{ms([100 * r['test_f1'] for r in recs]):>13}"
              f"{ms([100 * r['test_auc'] for r in recs]):>13}{PAPER_TABLE3_MEAN:>20.2f}"
              f"{'' if base else paired(c, lambda r: r['test_f1']):>22}"
              f"{'' if base else paired(c, lambda r: r['test_auc']):>22}")

    print("\nWeight picked per option on mean val F1 (test scores play no part)")
    for o in prefixes:
        val = {c: statistics.mean(r["val_f1"] for r in runs[c].values())
               for c in configs if c.startswith(o)}
        if val:
            best = max(val, key=val.get)
            print(f"  {o}: " + ", ".join(f"{c} {100 * v:.2f} (n={len(runs[c])})"
                                        for c, v in val.items()) + f" -> {best}")

    if kept:
        print("\nShare of train cells the consistency term used (mean over epochs and seeds)")
        for c in configs:
            if c in kept:
                print(f"  {c:<22} {100 * statistics.mean(kept[c].values()):.1f}%")

    teachers = [c for c in configs if all("teacher" in r for r in runs[c].values())]
    if teachers:
        print("\nEMA teacher scored at the student's picked epoch (a read-out; it does not "
              "pick the epoch). Δ = teacher - student, paired by seed")
        for c in teachers:
            recs = list(runs[c].values())
            d = [100 * (r["teacher"]["test_f1"] - r["test_f1"]) for r in recs]
            a = [100 * (r["teacher"]["test_auc"] - r["test_auc"]) for r in recs]
            print(f"  {c:<22} test F1 {ms([100 * r['teacher']['test_f1'] for r in recs])}"
                  f"  AUC {ms([100 * r['teacher']['test_auc'] for r in recs])}"
                  f"  ΔF1 {statistics.mean(d):+.1f}  ΔAUC {statistics.mean(a):+.1f}")

    print(f"\nPropaganda vs {baseline} (paired by seed)")
    for c in configs[1:]:
        print(f"  {c:<22} F1 {paired(c, lambda r: r['test']['propaganda_max_pool']['f1'])}"
              f"   AUC {paired(c, lambda r: r['test']['propaganda_max_pool']['auc'])}")

    print("\nTest weighted F1 per task, x100")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{c:>24}" for c in configs))
    for key, (label, paper_f1) in TABLE3_LABELS.items():
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(
            f"{ms([100 * r['test'][key]['f1'] for r in runs[c].values()]):>24}"
            for c in configs))

    if spread:
        print("\nSpread of val scores (sd over all val rows, mean over seeds); "
              f"a big drop vs {baseline} means a head collapsed toward one value")
        print(f"{'Task':<24}" + "".join(f"{c:>24}" for c in configs if c in spread))
        for key, (label, _) in TABLE3_LABELS.items():
            print(f"{label:<24}" + "".join(
                f"{statistics.mean(s[key] for s in spread[c].values()):>24.3f}"
                for c in configs if c in spread))


# Wider confidence filters for Option 2 (5% masking, weight 1, no masked-word
# loss): famo_consf1_m5 keeps the 10% most confident cells per side per head.
# q 0.2 keeps more; q 0.3 with a 0.2 floor keeps up to 30% per side but only
# scores <= 0.2 or >= 0.8, sides equal, so weak heads drop out rather than
# feeding near-coin-flip targets. Runs go to RETUNE_FAMO_DIR.
#
#     modal run --detach modal_replicate.py::filter_sweep
#     modal run modal_replicate.py::label_aware_report --baseline famo_consf1_m5 --prefixes famo_consq

FILTER_SWEEP_CONFIGS = {  # name: (confident share per side, floor)
    "famo_consq2_m5": (0.2, 0.0),
    "famo_consq3f2_m5": (0.3, 0.2),
}


@app.local_entrypoint()
def filter_sweep(seeds: str = "42,1,2,3", configs: str = ""):
    names = configs.split(",") if configs else list(FILTER_SWEEP_CONFIGS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        keep, floor = FILTER_SWEEP_CONFIGS[config]
        name = f"{config}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer="famo", mlm_lambda=0.0, mlm_mask_prob=0.05,
            consistency_weight=1.0, consistency_keep=keep, consistency_floor=floor,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    _spawn_all(calls)


# Teacher consistency without masking: the target comes from a teacher run
# with dropout off, the student answer from a pass with encoder dropout on or
# off. Same filter (q 0.1) and weight (1) as famo_consf1_m5. Arms:
#   famo_tc_drop: current weights as teacher, student dropout on (dropout views)
#   famo_tc_ema:  EMA teacher, student dropout off (better targets, no noise)
#   famo_tc_mt:   EMA teacher, student dropout on (Mean Teacher)
#
#     modal run modal_replicate.py::teacher_consistency --seeds 42 --epochs 1 --smoke
#     modal run --detach modal_replicate.py::teacher_consistency
#     modal run modal_replicate.py::label_aware_report --baseline famo --prefixes famo_tc,famo_consf1_m5

TEACHER_CONSISTENCY_CONFIGS = {  # name: (teacher, student dropout)
    "famo_tc_drop": ("current", True),
    "famo_tc_ema": ("ema", False),
    "famo_tc_mt": ("ema", True),
}


@app.local_entrypoint()
def teacher_consistency(seeds: str = "42,1,2,3", configs: str = "", epochs: int = 10,
                        smoke: bool = False):
    """`smoke` writes to smoke_<config>_s<seed>, so a short check never counts
    as a finished run or matches a report prefix."""
    names = configs.split(",") if configs else list(TEACHER_CONSISTENCY_CONFIGS)
    prefix = "smoke_" if smoke else ""
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [(c, s) for c, s in jobs if f"{prefix}{c}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        teacher, student_dropout = TEACHER_CONSISTENCY_CONFIGS[config]
        name = f"{prefix}{config}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer="famo", consistency_weight=1.0, consistency_keep=0.1,
            consistency_teacher=teacher, consistency_student_dropout=student_dropout,
            ema_decay=0.99,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR, "epochs": epochs},
        )))
    _spawn_all(calls)


# Mask check: retrain famo_fixed and, after training, score val on masked
# copies at several mask rates (BERT's rule, eval mode) against the clean
# answers: F1 change and the share of answers that flip, labelled vs
# unlabelled cells. Writes /outputs/paper_replication_mask_check/<name>.json.
#
#     modal run modal_replicate.py::mask_check
#     modal run modal_replicate.py::mask_check_report

MASK_CHECK_DIR = "paper_replication_mask_check"
MASK_CHECK_RATES = "0.05,0.1,0.15,0.3"


@app.local_entrypoint()
def mask_check(seed: int = 42, rates: str = MASK_CHECK_RATES):
    balancer, lam, mode = MLM_BALANCE_CONFIGS["famo_fixed"]
    name = f"famo_fixed_s{seed}"
    _spawn_all([(name, replicate.spawn(
        seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
        loss_balancer=balancer, mlm_lambda=lam, mlm_mode=mode,
        early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
        mask_sweep=rates, tag=f"mask_check/{name}", data_dir="/data/paper",
        **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
    ))])


@app.local_entrypoint()
def mask_check_report():
    import json
    import statistics

    for name in sorted(_done(MASK_CHECK_DIR)):
        d = json.loads(b"".join(out_vol.read_file(f"{MASK_CHECK_DIR}/{name}.json")))
        ms = d["mask_sweep"]
        rates = [f"{r:g}" for r in ms["rates"]]
        print(f"\n{name}: best epoch {d['best_epoch']}, val set, {ms['repeats']} maskings per "
              "rate; numbers are means over maskings")

        def mean(k, r, field):
            vals = [x[k][field] for x in ms["by_rate"][r] if x[k][field] is not None]
            return statistics.mean(vals) if vals else None

        for title, field, scale, sign in (("F1 change vs clean, x100", "f1", 100, True),
                                          ("Answers flipped, labelled cells, %", "flip_labelled", 100, False),
                                          ("Answers flipped, unlabelled cells, %", "flip_unlabelled", 100, False)):
            print(f"\n{title}")
            print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}{'clean val F1':>14}"
                  + "".join(f"{'rate ' + r:>12}" for r in rates))
            for key, (label, paper_f1) in TABLE3_LABELS.items():
                cells = []
                for r in rates:
                    v = mean(key, r, field)
                    if v is None:
                        cells.append(f"{'–':>12}")
                    elif sign:
                        cells.append(f"{scale * (v - ms['clean'][key]['f1']):>+12.1f}")
                    else:
                        cells.append(f"{scale * v:>12.1f}")
                print(f"{label:<24}{paper_f1:>20.2f}{100 * ms['clean'][key]['f1']:>14.1f}"
                      + "".join(cells))


# Option 2 alone (no masked-word loss) at a gentler 5% mask rate, with and
# without the per-head confidence filter on the consistency targets, against
# FAMO alone. Runs go to RETUNE_FAMO_DIR as <config>_s<seed>.
#
#     modal run --detach modal_replicate.py::consistency_filter
#     modal run modal_replicate.py::label_aware_report --baseline famo --prefixes famo_cons

CONSISTENCY_FILTER_CONFIGS = {  # name: (consistency weight, confident share per side)
    "famo_cons1_m5": (1.0, 0.0),    # control: every cell
    "famo_consf1_m5": (1.0, 0.1),   # confident 10% each side per head
    "famo_consf3_m5": (3.0, 0.1),
}


# Caruana (1997, Section 7.1): early stopping on the main task alone. Same
# setup as famo_consf1_m5 (same loss, all 10 outputs trained, equal task
# treatment by FAMO), but the epoch is picked, and training stopped, on one
# output's val F1 (patience RETUNE_PATIENCE on its F1 and AUC) instead of the
# mean over outputs. Baselines: famo_consf1_m5_s<seed> (mean stopping, same
# seed) and only_<task>_s<seed> (single-task).
#
#     modal run --detach modal_replicate.py::main_task_stop --task disagree_agree_bi --seeds 42
#     modal run modal_replicate.py::main_task_report --task disagree_agree_bi
#
# Writes /outputs/paper_replication_retune_famo/famo_consf1_m5_main_<task>_s<seed>.json

MAIN_TASK_BASE = "famo_consf1_m5"


# Approach 2: per-task stopping with self-distillation on the plain pooled
# loss (no FAMO), lr 1e-4, trained from scratch. Each output switches from its
# labels to its own best-epoch scores once its val F1 stalls `patience`
# epochs; the run ends when all have switched. Baseline: the saved re-tuned
# plain run RETUNE_DIR/lr0.0001_s<seed> (same settings); --baseline also
# trains a fresh one into SELF_DISTILL_DIR.
#
#     modal run --detach modal_replicate.py::self_distill_runs --seeds 42
#     modal run modal_replicate.py::self_distill_report --seeds 42
#
# Writes /outputs/paper_replication_self_distill/{sd_p<P>,plain}_s<seed>.json (+ .npz).

SELF_DISTILL_DIR = "paper_replication_self_distill"


@app.local_entrypoint()
def _sd_name(seed: int, patience: int, famo: bool, signal: str = "f1") -> str:
    sig = "" if signal == "f1" else f"_{signal}"
    return f"{MAIN_TASK_BASE + '_' if famo else ''}sd_p{patience}{sig}_s{seed}"


@app.local_entrypoint()
def self_distill_runs(seeds: str = "42", patience: int = 2, baseline: bool = False,
                      famo: bool = False, signal: str = "f1"):
    """--famo: on famo_consf1_m5 (FAMO + filtered 5% masked consistency);
    baseline = the saved RETUNE_FAMO_DIR/famo_consf1_m5_s<seed>."""
    done = _done(SELF_DISTILL_DIR)
    calls = []
    extra = {}
    if famo:
        weight, keep = CONSISTENCY_FILTER_CONFIGS[MAIN_TASK_BASE]
        extra = dict(loss_balancer="famo", mlm_lambda=0.0, mlm_mask_prob=0.05,
                     consistency_weight=weight, consistency_keep=keep)
    for seed in (int(s) for s in seeds.split(",")):
        arms = [(_sd_name(seed, patience, famo, signal), True)]
        if baseline and not famo:
            arms.append((f"plain_s{seed}", False))
        for name, sd in arms:
            if name in done:
                print(f"{name} already done, skipping", flush=True)
                continue
            calls.append((name, replicate.spawn(
                seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
                early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
                self_distill=sd, self_distill_patience=patience, self_distill_signal=signal,
                tag=f"self_distill/{name}", data_dir="/data/paper",
                **extra, **{**WINNER, "learning_rate": 1e-4},
            )))
    _spawn_all(calls)


@app.local_entrypoint()
def self_distill_report(seeds: str = "42", patience: int = 2, famo: bool = False,
                        signal: str = "f1"):
    """Per output, val and test F1: fresh plain run at its mean-F1 epoch, the
    same run replayed at each output's own val-best epoch (needs several
    models), the self-distilled run's one model, and the saved single-task
    runs. Every epoch is picked on val; test is only read at that epoch."""
    import json
    import statistics

    def load(directory, name):
        try:
            return json.loads(b"".join(out_vol.read_file(f"{directory}/{name}.json")))
        except Exception:
            return None

    base = "famo_consf1_m5" if famo else "plain"
    cols = [f"{base}, mean stop", f"{base}, per-task epochs", "self-distilled", "single-task"]
    score = {(c, split): {} for c in cols for split in ("val", "test")}

    def add(col, rec, k):
        for split in ("val", "test"):
            score[(col, split)].setdefault(k, []).append(rec[split][k]["f1"])

    for seed in (int(s) for s in seeds.split(",")):
        # The saved re-tuned plain run (same settings); a fresh one if launched.
        if famo:
            plain = load(RETUNE_FAMO_DIR, f"{MAIN_TASK_BASE}_s{seed}")
        else:
            plain = (load(SELF_DISTILL_DIR, f"plain_s{seed}")
                     or load(RETUNE_DIR, f"lr0.0001_s{seed}"))
        sd_name = _sd_name(seed, patience, famo, signal)
        sd = load(SELF_DISTILL_DIR, sd_name)
        if plain:
            recs = plain["epoch_scores"]
            print(f"{base}_s{seed}: {len(recs)} epochs, kept epoch {plain['best_epoch']}")
            for k in TABLE3_LABELS:
                add(f"{base}, mean stop", recs[plain["best_epoch"] - 1], k)
                e = max(range(len(recs)), key=lambda i: (recs[i]["val"][k]["f1"], -i))
                add(f"{base}, per-task epochs", recs[e], k)
        if sd:
            recs = sd["epoch_scores"]
            print(f"{sd_name}: {len(recs)} epochs, kept epoch {sd['best_epoch']}; "
                  "switches (output: best epoch -> switched after epoch):")
            switches = (sd.get("self_distill") or {}).get("switches")
            if switches is None:  # older JSONs: rebuild from each epoch's "distilled" list
                switches = []
                for k in TABLE3_LABELS:
                    n = next((i + 1 for i, r in enumerate(recs) if k in r.get("distilled", [])), None)
                    if n:
                        v = [r["val"][k][signal] for r in recs[:n]]
                        b = v.index(min(v) if signal == "loss" else max(v))
                        switches.append({"output": k, "best_epoch": 1 + b,
                                         "switched_after_epoch": n})
                switches.sort(key=lambda w: w["switched_after_epoch"])
            for sw in switches:
                print(f"    {TABLE3_LABELS[sw['output']][0]:<24} {sw['best_epoch']} -> "
                      f"{sw['switched_after_epoch']}")
            for k in TABLE3_LABELS:
                add("self-distilled", recs[sd["best_epoch"] - 1], k)
        for k in TABLE3_LABELS:
            single = load(SINGLE_RETUNE_DIR, f"only_{k}_s{seed}")
            if single:
                add("single-task", single["epoch_scores"][single["best_epoch"] - 1], k)

    have = [c for c in cols if score[(c, "test")]]

    def mean_of_10(c, split):
        d = score[(c, split)]
        n = min(len(v) for v in d.values())
        return statistics.mean(statistics.mean(d[k][i] for k in TABLE3_LABELS) for i in range(n))

    def pair(v, t):
        return f"{100 * v:.1f} / {100 * t:.1f}"

    def cell(c, k):
        return pair(statistics.mean(score[(c, "val")][k]), statistics.mean(score[(c, "test")][k]))

    print(f"\nWeighted F1 x100, val / test, mean over seeds {seeds}")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{c + ' val/test':>32}" for c in have))
    for k, (label, paper_f1) in TABLE3_LABELS.items():
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(f"{cell(c, k):>32}" for c in have))
    paper_mean = statistics.mean(p for _, p in TABLE3_LABELS.values())
    print(f"{'Mean of 10':<24}{paper_mean:>20.2f}" + "".join(
        f"{pair(mean_of_10(c, 'val'), mean_of_10(c, 'test')):>32}" for c in have))


def _main_task_name(task: str, seed: int) -> str:
    return f"{MAIN_TASK_BASE}_main_{task}_s{seed}"


@app.local_entrypoint()
def main_task_stop(task: str = "disagree_agree_bi", seeds: str = "42"):
    weight, keep = CONSISTENCY_FILTER_CONFIGS[MAIN_TASK_BASE]
    done = _done(RETUNE_FAMO_DIR)
    calls = []
    for seed in (int(s) for s in seeds.split(",")):
        name = _main_task_name(task, seed)
        if name in done:
            print(f"{name} already done, skipping", flush=True)
            continue
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer="famo", mlm_lambda=0.0, mlm_mask_prob=0.05,
            consistency_weight=weight, consistency_keep=keep,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            main_task=task,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    _spawn_all(calls)


@app.local_entrypoint()
def main_task_report(task: str = "disagree_agree_bi", seeds: str = "42"):
    """Test F1/AUC of the main-task-stopped run vs famo_consf1_m5 (mean
    stopping) and single-task, same seed. Also replays, inside the mean-stopped
    run, the epoch its own val F1 for ``task`` would have picked."""
    import json

    def load(directory, name):
        try:
            return json.loads(b"".join(out_vol.read_file(f"{directory}/{name}.json")))
        except Exception:
            return None

    def at_best(d):
        return d["epoch_scores"][d["best_epoch"] - 1]

    label, paper_f1 = TABLE3_LABELS[task]
    for seed in (int(s) for s in seeds.split(",")):
        main = load(RETUNE_FAMO_DIR, _main_task_name(task, seed))
        mean = load(RETUNE_FAMO_DIR, f"{MAIN_TASK_BASE}_s{seed}")
        single = load(SINGLE_RETUNE_DIR, f"only_{task}_s{seed}")
        print(f"\nSeed {seed}, main task {label}. Paper (Table 3 F1) {paper_f1:.2f}")
        rows = []
        if single:
            r = at_best(single)
            rows.append(("single-task", single["best_epoch"], r["val"][task], r["test"][task], None))
        if mean:
            r = at_best(mean)
            rows.append(("MTL, stop on mean F1", mean["best_epoch"], r["val"][task],
                         r["test"][task], r))
            recs = mean["epoch_scores"]
            # first best val F1 of the task within the epochs that run trained
            # (it stopped on the mean, so later epochs were never seen)
            e = 1 + max(range(len(recs)), key=lambda i: (recs[i]["val"][task]["f1"], -i))
            rows.append((f"  replay: its {task} val-best epoch", e, recs[e - 1]["val"][task],
                         recs[e - 1]["test"][task], recs[e - 1]))
        if main:
            r = at_best(main)
            rows.append(("MTL, stop on main task", main["best_epoch"], r["val"][task],
                         r["test"][task], r))
        print(f"{'Setup':<40}{'epoch':>6}{'val F1':>8}{'test F1':>9}{'test AUC':>10}"
              f"{'mean test F1 (10 outputs)':>27}")
        for name, ep, v, t, full in rows:
            mean_f1 = f"{100 * full['test_f1']:.1f}" if full else "–"
            print(f"{name:<40}{ep:>6}{100 * v['f1']:>8.1f}{100 * t['f1']:>9.1f}"
                  f"{100 * t['auc']:>10.1f}{mean_f1:>27}")
        if main:
            r = at_best(main)
            print(f"\nMain-task run, val / test F1 per output at its kept epoch {main['best_epoch']}"
                  f" (epochs trained {len(main['epoch_scores'])}):")
            def vt(rec, key):
                return f"{100 * rec['val'][key]['f1']:.1f} / {100 * rec['test'][key]['f1']:.1f}"

            print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}{'main-stop val/test':>22}"
                  f"{'mean-stop val/test':>22}")
            for key, (lab, pf1) in TABLE3_LABELS.items():
                ms = vt(at_best(mean), key) if mean else "–"
                print(f"{lab:<24}{pf1:>20.2f}{vt(r, key):>22}{ms:>22}")


@app.local_entrypoint()
def consistency_filter(seeds: str = "42,1,2,3", configs: str = ""):
    names = configs.split(",") if configs else list(CONSISTENCY_FILTER_CONFIGS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        weight, keep = CONSISTENCY_FILTER_CONFIGS[config]
        name = f"{config}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer="famo", mlm_lambda=0.0, mlm_mask_prob=0.05,
            consistency_weight=weight, consistency_keep=keep,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    _spawn_all(calls)


# Semi-supervised terms on foreign cells (a head with no label, on rows from a
# named source): a target rate for the batch-mean score (KL) and an entropy
# (confidence) penalty, the latter off for the first 2 epochs. Built on the
# plain pooled loss at the re-tuned settings, so the baseline is the plain run
# in RETUNE_DIR (lr0.0001_s<seed>), same seeds. Target rates: SSL_DEFAULT_RATES
# in the script, mirrored in SSL_RATES below.
#
#     modal run modal_replicate.py::masked_lm_tests
#     modal run --detach modal_replicate.py::semi_supervised --configs ssl_re1 --seeds 42
#     modal run modal_replicate.py::semi_supervised_check --name ssl_re1_s42
#     modal run --detach modal_replicate.py::semi_supervised
#     modal run modal_replicate.py::semi_supervised_report
#
# Writes /outputs/paper_replication_ssl/<config>_s<seed>.json (+ .npz).

SSL_DIR = "paper_replication_ssl"
SSL_BASELINE = "lr0.0001"  # plain pooled loss, dropout 0, wide, lr 1e-4, F1 p3
SSL_LR = 1e-4
# The sanity run (ssl_re1_s42) showed the IAC heads' targets on Propaganda
# rows clash with the rule-made IAC labels there, so the "nonews" configs keep
# only the propaganda pairs and the IAC heads on IBM rows (rate term only).
SSL_NONEWS_RATES = (
    "propaganda_max_pool@IAC=0.30,propaganda_max_pool@IBM=0.10,"
    "emotion_fact_bi@IBM=0.15,nasty_nice_bi@IBM=0.03,attacking_respectful_bi@IBM=0.05"
)
SSL_CONFIGS = {  # name: (rate weight, entropy weight[, rates; default all 8 pairs])
    "ssl_r0.1": (0.1, 0.0),
    "ssl_r1": (1.0, 0.0),
    "ssl_re0.1": (0.1, 0.1),
    "ssl_re1": (1.0, 1.0),
    "ssl_nonews_r0.1": (0.1, 0.0, SSL_NONEWS_RATES),
    "ssl_nonews_r1": (1.0, 0.0, SSL_NONEWS_RATES),
}
# (head, source): target rate, as SSL_DEFAULT_RATES in the script.
SSL_RATES = {
    ("propaganda_max_pool", "IAC"): 0.30, ("propaganda_max_pool", "IBM"): 0.10,
    ("emotion_fact_bi", "Propaganda"): 0.12, ("nasty_nice_bi", "Propaganda"): 0.05,
    ("attacking_respectful_bi", "Propaganda"): 0.07,
    ("emotion_fact_bi", "IBM"): 0.15, ("nasty_nice_bi", "IBM"): 0.03,
    ("attacking_respectful_bi", "IBM"): 0.05,
}


def _ssl_label(head: str, source: str) -> str:
    short = "propaganda" if head == "propaganda_max_pool" else head[:-3]
    return f"{short}_{source}"


@app.local_entrypoint()
def semi_supervised(seeds: str = "42,1,2", configs: str = ""):
    names = configs.split(",") if configs else list(SSL_CONFIGS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(SSL_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for config, seed in todo:
        rate_w, entropy_w, *rates = SSL_CONFIGS[config]
        name = f"{config}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            semi_supervised=True, ssl_rate_weight=rate_w, ssl_entropy_weight=entropy_w,
            ssl_rates=rates[0] if rates else "",
            tag=f"ssl/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": SSL_LR},
        )))
    _spawn_all(calls)


@app.local_entrypoint()
def semi_supervised_check(name: str = "ssl_re1_s42"):
    """Per-epoch logs of one run, for the sanity checks: the extra terms on
    train and val, the mean score on each pair's foreign val cells against
    its target, and val / test F1."""
    import json

    d = json.loads(b"".join(out_vol.read_file(f"{SSL_DIR}/{name}.json")))
    h, recs = d["history"], d["epoch_scores"]
    print(f"{name}: {len(recs)} epochs, weights kept from epoch {d['best_epoch']} (best val F1)")
    cols = [("ssl_rate_loss", "rate (train)"), ("val_ssl_rate_loss", "rate (val)"),
            ("ssl_entropy_loss", "entropy (train)"), ("val_ssl_entropy_loss", "entropy (val)"),
            ("ssl_entropy_weight", "entropy weight"), ("loss", "sup. loss"),
            ("val_loss", "val sup. loss")]
    print(f"{'epoch':>5}" + "".join(f"{label:>17}" for _, label in cols)
          + f"{'val F1':>9}{'test F1':>9}")
    for i, r in enumerate(recs):
        print(f"{r['epoch']:>5}" + "".join(
            f"{h[k][i]:>17.4f}" if k in h else f"{'–':>17}" for k, _ in cols)
            + f"{100 * r['val_f1']:>9.1f}{100 * r['test_f1']:>9.1f}")
    print("\nMean score on foreign val cells per epoch (target in brackets)")
    pairs = [(_ssl_label(hd, src), rate) for (hd, src), rate in SSL_RATES.items()]
    print(f"{'epoch':>5}" + "".join(f"{lab[:20]:>22}" for lab, _ in pairs))
    print(f"{'':>5}" + "".join(f"{'(' + format(rate, '.2f') + ')':>22}" for _, rate in pairs))
    for i, r in enumerate(recs):
        print(f"{r['epoch']:>5}" + "".join(
            f"{h[f'val_ssl_mean_{lab}'][i]:>22.3f}" if f"val_ssl_mean_{lab}" in h else f"{'–':>22}"
            for lab, _ in pairs))


@app.local_entrypoint()
def semi_supervised_report(rule: str = "F1 p3"):
    """Semi-supervised runs against the plain pooled-loss runs on the same
    seeds: mean scores and paired differences, test F1 per task beside the
    paper, IAC heads on IAC rows only (without the rule-labelled Propaganda
    rows), how often the pooled propaganda output catches each technique,
    and the mean score on foreign test cells against the target rates."""
    import io
    import json
    import statistics

    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.metrics import f1_score

    def load(vol, path):
        return b"".join(vol.read_file(path))

    metric, patience = RETUNE_RULES[rule]
    runs = {}  # config: {seed: {"rec", "thr", "npz", "saved"}}

    def add(config, seed, directory, name):
        d = json.loads(load(out_vol, f"{directory}/{name}.json"))
        z = np.load(io.BytesIO(load(out_vol, f"{directory}/{name}_predictions.npz")))
        runs.setdefault(config, {})[seed] = {
            "rec": _replay(d["epoch_scores"], metric, patience), "saved": d["best_epoch"],
            "thr": d["thresholds"], "npz": {k: z[k] for k in z.files}}

    for name in sorted(_done(SSL_DIR)):
        config, seed = name.rsplit("_s", 1)
        add(config, int(seed), SSL_DIR, name)
    if not runs:
        raise SystemExit(f"No runs in {SSL_DIR}")
    base = "plain (baseline)"
    for seed in sorted({s for r in runs.values() for s in r}):
        add(base, seed, RETUNE_DIR, f"{SSL_BASELINE}_s{seed}")
    configs = [base] + [c for c in SSL_CONFIGS if c in runs]
    src = {split: pd.read_csv(io.BytesIO(load(data_vol, f"paper/no_prop_overlap_{split}.csv")),
                              usecols=["dataset"], low_memory=False)["dataset"].to_numpy()
           for split in ("val", "test")}

    def ms(xs):
        return (f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1
                else f"{xs[0]:.1f}")

    def paired(c, get):
        seeds = sorted(set(runs[c]) & set(runs[base]))
        d = [get(runs[c][s]) - get(runs[base][s]) for s in seeds]
        if len(d) < 2:
            return "–" if not d else f"{d[0]:+.1f} (n=1)"
        m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        h = stats.t.ppf(0.975, len(d) - 1) * se
        return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]"

    print(f"\nEpoch picked by '{rule}' on val; mean over the 10 Table-3 outputs, x100, "
          "mean ± sd over seeds; Δ is paired by seed with a 95% t interval")
    print(f"{'config':<18}{'n':>3}{'epoch':>7}{'val F1':>13}{'test F1':>13}{'test AUC':>13}"
          f"{'Paper (Table 3 F1)':>20}{'ΔF1 vs plain':>22}{'ΔAUC vs plain':>22}")
    for c in configs:
        recs = [r["rec"] for r in runs[c].values()]
        print(f"{c:<18}{len(recs):>3}{statistics.mean(r['epoch'] for r in recs):>7.1f}"
              f"{ms([100 * r['val_f1'] for r in recs]):>13}"
              f"{ms([100 * r['test_f1'] for r in recs]):>13}"
              f"{ms([100 * r['test_auc'] for r in recs]):>13}{PAPER_TABLE3_MEAN:>20.2f}"
              f"{'' if c == base else paired(c, lambda r: 100 * r['rec']['test_f1']):>22}"
              f"{'' if c == base else paired(c, lambda r: 100 * r['rec']['test_auc']):>22}")

    print("\nTest weighted F1 per task, x100 (mean ± sd over seeds)")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{c:>20}" for c in configs))
    for key, (label, paper_f1) in TABLE3_LABELS.items():
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(
            f"{ms([100 * r['rec']['test'][key]['f1'] for r in runs[c].values()]):>20}"
            for c in configs))
    print("\nPer-task ΔF1 vs plain, paired by seed")
    for key, (label, _) in TABLE3_LABELS.items():
        print(f"{label:<24}" + "".join(
            f"{paired(c, lambda r: 100 * r['rec']['test'][key]['f1']):>24}" for c in configs[1:]))

    # The rest uses the saved predictions, i.e. the weights kept at the end of
    # the run (best val F1 over every epoch run), with the saved thresholds.
    differ = {c: [s for s, r in runs[c].items() if r["saved"] != r["rec"]["epoch"]]
              for c in configs}
    differ = {c: s for c, s in differ.items() if s}
    print("\nFrom here on: saved predictions (epoch with the best val F1 of the whole run)"
          + (f"; differs from the '{rule}' epoch for seeds {differ}" if differ else ""))

    def wf1(y, p, t):
        return 100 * f1_score(y, (p >= t).astype(int), average="weighted", zero_division=0)

    print("\nIAC heads on test rows from IAC only (no rule-labelled Propaganda rows), "
          "weighted F1 x100 at the saved threshold")
    print(f"{'Task':<24}{'Paper (Table 3 F1)':>20}" + "".join(f"{c:>20}" for c in configs)
          + "".join(f"{'Δ ' + c:>24}" for c in configs[1:]))
    iac = src["test"] == "IAC"

    def iac_f1(r, key):
        y, p = r["npz"][f"test/label/{key}"].reshape(-1), r["npz"][f"test/prob/{key}"].reshape(-1)
        m = iac & (y != -1)
        return wf1(y[m], p[m], r["thr"][key])

    for key, (label, paper_f1) in list(TABLE3_LABELS.items())[1:9]:
        print(f"{label:<24}{paper_f1:>20.2f}" + "".join(
            f"{ms([iac_f1(r, key) for r in runs[c].values()]):>20}" for c in configs)
            + "".join(f"{paired(c, lambda r: iac_f1(r, key)):>24}" for c in configs[1:]))

    print("\nShare of test rows with each technique that the pooled propaganda output "
          "flags (recall x100 at the saved threshold), rarest first")
    techniques = sorted(k.split("/", 2)[2] for k in runs[base][next(iter(runs[base]))]["npz"]
                        if k.startswith("test/label/") and k.split("/", 2)[2] not in TABLE3_LABELS)

    def caught(r, t):
        y = r["npz"][f"test/label/{t}"].reshape(-1) == 1
        p = r["npz"]["test/prob/propaganda_max_pool"].reshape(-1)
        return 100 * float(np.mean(p[y] >= r["thr"]["propaganda_max_pool"])) if y.any() else np.nan

    any_run = runs[base][next(iter(runs[base]))]["npz"]
    counts = {t: int((any_run[f"test/label/{t}"].reshape(-1) == 1).sum()) for t in techniques}
    print(f"{'Technique':<44}{'n':>5}" + "".join(f"{c:>20}" for c in configs))
    for t in sorted(techniques, key=counts.get):
        print(f"{t[:43]:<44}{counts[t]:>5}" + "".join(
            f"{ms([caught(r, t) for r in runs[c].values()]):>20}" for c in configs))

    print("\nMean score on foreign test cells (head unlabelled, row from that source), "
          "mean over seeds")
    print(f"{'pair':<30}{'target':>8}" + "".join(f"{c:>20}" for c in configs))
    for (head, source), rate in SSL_RATES.items():
        def mean_score(r):
            y, p = r["npz"][f"test/label/{head}"].reshape(-1), r["npz"][f"test/prob/{head}"].reshape(-1)
            m = (src["test"] == source) & (y == -1)
            return float(p[m].mean())
        print(f"{_ssl_label(head, source):<30}{rate:>8.2f}" + "".join(
            f"{statistics.mean(mean_score(r) for r in runs[c].values()):>20.3f}"
            for c in configs))



# Propaganda pooling (Nov 2026): the max over the 18 technique heads trains
# only the top head in each row, and in practice one random head wins on
# 96-100% of rows. Three ways to reach every head, each on top of the current
# best run (famo_consf1_m5: FAMO, consistency weight 1 on the confident 10%
# per side, 5% masking, no masked-word loss). Runs go to RETUNE_FAMO_DIR.
#
#     modal run --detach modal_replicate.py::propaganda_pooling
#     modal run modal_replicate.py::label_aware_report --baseline famo_consf1_m5 --prefixes famo_consf1_p

PROPAGANDA_POOLING_CONFIGS = {  # name: (pool strategy, per-technique loss weight)
    "famo_consf1_pnor": ("noisy_or", 0.0),   # 1 - prod(1 - p_i)
    "famo_consf1_plse": ("lse", 0.0),        # sigmoid(logsumexp(logit p_i)), t = 1
    "famo_consf1_ptech": ("max", 0.1),       # max pool + 0.1 x per-technique loss
}


@app.local_entrypoint()
def propaganda_pooling(seeds: str = "42,1,2,3", configs: str = ""):
    names = configs.split(",") if configs else list(PROPAGANDA_POOLING_CONFIGS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [(c, s) for c, s in jobs if f"{c}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    weight, keep = CONSISTENCY_FILTER_CONFIGS["famo_consf1_m5"]
    calls = []
    for config, seed in todo:
        pool, tech = PROPAGANDA_POOLING_CONFIGS[config]
        name = f"{config}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer="famo", mlm_lambda=0.0, mlm_mask_prob=0.05,
            consistency_weight=weight, consistency_keep=keep,
            propaganda_pool=pool, technique_loss_weight=tech,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    _spawn_all(calls)


# Rule-made IAC labels (Nov 2026): on Propaganda (news) rows the 8 IAC
# discourse labels were filled from the technique tags by a fixed lookup
# (e.g. Name_Calling -> disagree, attacking, emotion, nasty = 1), 45,111 train
# cells, ~12% of the discourse training labels. The paper's Table 1 counts
# include them. This reruns the best run (famo_consf1_m5) with those labels
# masked in train only; val/test keep them, so the epoch choice and the
# standard scores use the same rows as the baseline. Runs go to RETUNE_FAMO_DIR.
#
#     modal run modal_replicate.py::masked_lm_tests
#     modal run --detach modal_replicate.py::no_rule_labels
#     modal run modal_replicate.py::label_aware_report --baseline famo_consf1_m5 --prefixes famo_consf1_norule

NO_RULE_CONFIG = "famo_consf1_norule"


@app.local_entrypoint()
def no_rule_labels(seeds: str = "42,1,2,3"):
    jobs = [int(s) for s in seeds.split(",")]
    done = _done(RETUNE_FAMO_DIR)
    todo = [s for s in jobs if f"{NO_RULE_CONFIG}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    weight, keep = CONSISTENCY_FILTER_CONFIGS["famo_consf1_m5"]
    calls = []
    for seed in todo:
        name = f"{NO_RULE_CONFIG}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
            loss_balancer="famo", mlm_lambda=0.0, mlm_mask_prob=0.05,
            consistency_weight=weight, consistency_keep=keep,
            mask_news_discourse_train=True,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"retune_famo/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    _spawn_all(calls)


# Single-task baselines at the re-tuned settings (lr 1e-4, val-F1 stopping with
# patience 3, dropout 0, wide heads), the same as the multi-task runs in
# RETUNE_DIR / RETUNE_FAMO_DIR. As in `transfer`, each run trains one Table-3
# output and drops train rows with no label for it. The old single-task runs
# (TRANSFER_DIR, lr 3e-4, val-loss stopping) are not comparable with these.
#
#     modal run --detach modal_replicate.py::single_task_retune
#     modal run modal_replicate.py::single_task_retune_report
#
# Writes /outputs/paper_replication_single_retune/only_<output>_s<seed>.json (+ .npz).

SINGLE_RETUNE_DIR = "paper_replication_single_retune"
# Paper Table 3 (weighted F1, small BERT): single-task, multi-task.
PAPER_TABLE3_BOTH = {
    "propaganda_max_pool": 63.07, "disagree_agree_bi": 71.15, "emotion_fact_bi": 68.11,
    "attacking_respectful_bi": 67.46, "nasty_nice_bi": 66.90, "personal_audience_bi": 63.25,
    "defeater_undercutter_bi": 45.97, "negotiate_attack_bi": 64.76,
    "questioning_asserting_bi": 59.61, "WA_bi": 80.93,
}
# Multi-task runs at the same settings: label: (directory, run-name prefix).
SINGLE_RETUNE_MULTI = {
    "plain MTL": (RETUNE_DIR, "lr0.0001"),
    "famo": (RETUNE_FAMO_DIR, "famo"),
    "famo_fixed": (RETUNE_FAMO_DIR, "famo_fixed"),
    "famo_consf1_m5": (RETUNE_FAMO_DIR, "famo_consf1_m5"),
}


@app.local_entrypoint()
def single_task_retune(seeds: str = "42,1,2,3", tasks: str = ""):
    outputs = tasks.split(",") if tasks else TABLE3_OUTPUTS
    jobs = [(t, int(s)) for t in outputs for s in seeds.split(",")]
    done = _done(SINGLE_RETUNE_DIR)
    todo = [(t, s) for t, s in jobs if f"only_{t}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = []
    for task, seed in todo:
        name = f"only_{task}_s{seed}"
        calls.append((name, replicate.spawn(
            seed=seed, pooled_loss=True, verbose=2, save_predictions=True, tasks=task,
            drop_unlabeled_train=True,
            early_stop_metric="f1", early_stop_patience=RETUNE_PATIENCE,
            tag=f"single_retune/{name}", data_dir="/data/paper",
            **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
        )))
    _spawn_all(calls)


@app.local_entrypoint()
def single_task_retune_report(rule: str = "F1 p3"):
    """Single-task test F1 per output beside the paper's single- and multi-task
    columns, and each multi-task setup minus single-task, paired by seed."""
    import json
    import statistics

    from scipy import stats

    metric, patience = RETUNE_RULES[rule]

    def load(directory, name):
        d = json.loads(b"".join(out_vol.read_file(f"{directory}/{name}.json")))
        return _replay(d["epoch_scores"], metric, patience)

    single = {}  # output: {seed: record}
    for name in sorted(_done(SINGLE_RETUNE_DIR)):
        task, seed = name[len("only_"):].rsplit("_s", 1)
        single.setdefault(task, {})[int(seed)] = load(SINGLE_RETUNE_DIR, name)
    if not single:
        raise SystemExit(f"No runs in {SINGLE_RETUNE_DIR}")
    seeds = sorted({s for r in single.values() for s in r})
    multi = {}  # label: {seed: record}
    for label, (directory, prefix) in SINGLE_RETUNE_MULTI.items():
        have = _done(directory)
        multi[label] = {s: load(directory, f"{prefix}_s{s}") for s in seeds
                        if f"{prefix}_s{s}" in have}

    def f1(rec, task):
        return 100 * rec["test"][task]["f1"]

    def ms(xs):
        return (f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}" if len(xs) > 1
                else f"{xs[0]:.1f}" if xs else "–")

    def ci(d):
        if len(d) < 2:
            return f"{d[0]:+.1f} (n=1)" if d else "–"
        m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
        h = stats.t.ppf(0.975, len(d) - 1) * se
        return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]"

    tasks = [t for t in TABLE3_OUTPUTS if t in single]
    print(f"\nTest weighted F1 x100, epoch picked by '{rule}' on val, mean ± sd over seeds "
          f"{seeds}. Single-task runs: {sum(len(v) for v in single.values())}")
    print(f"{'Task':<24}{'Paper single':>13}{'Paper multi (Table 3 F1)':>26}{'single':>14}"
          + "".join(f"{label:>16}" for label in multi))
    for t in tasks:
        print(f"{TABLE3_LABELS[t][0]:<24}{PAPER_TABLE3_BOTH[t]:>13.2f}{TABLE3_LABELS[t][1]:>26.2f}"
              f"{ms([f1(r, t) for r in single[t].values()]):>14}"
              + "".join(f"{ms([f1(r, t) for r in m.values()]):>16}" for m in multi.values()))

    # Mean over outputs per seed, only seeds where every output has a run.
    full = [s for s in seeds if all(s in single[t] for t in tasks)]
    single_mean = {s: statistics.mean(f1(single[t][s], t) for t in tasks) for s in full}
    paper_single = statistics.mean(PAPER_TABLE3_BOTH[t] for t in tasks)
    paper_multi = statistics.mean(TABLE3_LABELS[t][1] for t in tasks)
    print(f"{'Mean over outputs':<24}{paper_single:>13.2f}{paper_multi:>26.2f}"
          f"{ms(list(single_mean.values())):>14}"
          + "".join(f"{ms([statistics.mean(f1(m[s], t) for t in tasks) for s in full if s in m]):>16}"
                    for m in multi.values()))

    print("\nMulti-task minus single-task, test F1 x100, paired by seed with a 95% t interval "
          "(negative = multi-task is worse)")
    print(f"{'Task':<24}" + "".join(f"{label:>24}" for label in multi))
    for t in tasks:
        print(f"{TABLE3_LABELS[t][0]:<24}" + "".join(
            f"{ci([f1(m[s], t) - f1(single[t][s], t) for s in sorted(single[t]) if s in m]):>24}"
            for m in multi.values()))
    print(f"{'Mean over outputs':<24}" + "".join(
        f"{ci([statistics.mean(f1(m[s], t) for t in tasks) - single_mean[s] for s in full if s in m]):>24}"
        for m in multi.values()))
    # Delta_m: mean relative change over outputs (Maninis et al., 2019), per seed.
    print(f"{'Δm (% vs single)':<24}" + "".join(
        f"{ci([100 * statistics.mean((f1(m[s], t) - f1(single[t][s], t)) / f1(single[t][s], t) for t in tasks) for s in full if s in m]):>24}"
        for m in multi.values()))
    print(f"\nPaper: mean single {paper_single:.2f}, multi {paper_multi:.2f}, "
          f"Δ {paper_multi - paper_single:+.2f}, Δm "
          f"{100 * statistics.mean((TABLE3_LABELS[t][1] - PAPER_TABLE3_BOTH[t]) / PAPER_TABLE3_BOTH[t] for t in tasks):+.2f}%")
    print("Epochs picked (single-task, mean over seeds): " + ", ".join(
        f"{TABLE3_LABELS[t][0]} {statistics.mean(r['epoch'] for r in single[t].values()):.1f}"
        for t in tasks))


# --------------------------------------------------------------------------
# No rule labels anywhere, and hide-one-label runs (Nov 2026)
# --------------------------------------------------------------------------
# From here on the 45k rule-made IAC labels on Propaganda rows are removed
# from train, val and test (`no_rule_labels`), so IAC labels are scored on IAC
# rows only. Settings are the re-tuned ones (WINNER, lr 1e-4, val-F1 stopping
# with patience 3), as in `retune_famo`.
#
# `norule_baselines`: full multi-task runs, with probes on every output (the
#   reference the hide-one-label probes are compared against).
# `hide_one`: one run per output, with that output hidden on every train row;
#   its head is scored but left out of epoch selection, and a probe reads it
#   from the frozen model. The other outputs' scores give leave-one-task-out
#   transfer against the full run of the same method and seed.
#
#     MULTITASK_GPU=H100,A100-80GB,L40S,L4 modal run --detach modal_replicate.py::norule_baselines --seeds 42,1,2
#     MULTITASK_GPU=L4 modal run --detach modal_replicate.py::hide_one --tasks nasty_nice_bi --seeds 42
#
# Writes /outputs/paper_replication_hide_one/<method>_{full,hide_<output>}_s<seed>.json (+ .npz).

HIDE_ONE_DIR = "paper_replication_hide_one"
HIDE_ONE_METHODS = {  # name: replicate() kwargs on top of the shared ones
    "plain": {},
    "famo": {"loss_balancer": "famo"},
    "famo_consf1_m5": {"loss_balancer": "famo", "mlm_lambda": 0.0, "mlm_mask_prob": 0.05,
                       "consistency_weight": 1.0, "consistency_keep": 0.1},
}


def _hide_one_kwargs(method: str, seed: int, name: str) -> dict:
    if method not in HIDE_ONE_METHODS:
        raise SystemExit(f"method must be one of {list(HIDE_ONE_METHODS)}")
    return dict(seed=seed, pooled_loss=True, verbose=2, save_predictions=True,
                no_rule_labels=True, early_stop_metric="f1",
                early_stop_patience=RETUNE_PATIENCE, tag=f"hide_one/{name}",
                data_dir="/data/paper", **{**WINNER, "learning_rate": RETUNE_FAMO_LR},
                **HIDE_ONE_METHODS[method])


@app.local_entrypoint()
def norule_unweighted(seeds: str = "42", method: str = "plain"):
    """Main data, no rule labels, all labels seen, with an unweighted loss
    (--unweighted-loss: no task weights, no class weights)."""
    jobs = [int(s) for s in seeds.split(",")]
    done = _done(HIDE_ONE_DIR)
    todo = [s for s in jobs if f"{method}_unweighted_full_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    _spawn_all([(f"{method}_unweighted_full_s{s}", replicate.spawn(
        **_hide_one_kwargs(method, s, f"{method}_unweighted_full_s{s}"),
        probe_tasks="all", unweighted_loss=True)) for s in todo])


# Label-budget runs: one output keeps 5 / 10 / 25 / 100% of its train labels
# (seeded, nested), val and test untouched. Multi-task (famo_consf1_m5, all
# other labels kept) against single-task on the same kept rows. A single-task
# run on a share p repeats its rows 1/p times, so every single-task run takes
# the same number of training steps and is scored after the same steps.
# The 100% multi-task run is `famo_consf1_m5_full_s<seed>` in HIDE_ONE_DIR.
#
#     MULTITASK_GPU=L4 modal run --detach modal_replicate.py::label_budget --task attacking_respectful_bi --seeds 42
#
# Writes /outputs/paper_replication_label_budget/{mtl,single}_<task>_p<pct>_s<seed>.json.

LABEL_BUDGET_DIR = "paper_replication_label_budget"


def _label_budget_name(arm: str, task: str, share: float, seed: int) -> str:
    return f"{arm}_{task}_p{round(100 * share)}_s{seed}"


@app.local_entrypoint()
def label_budget(task: str = "attacking_respectful_bi", shares: str = "0.05,0.1,0.25,1",
                 seeds: str = "42", arms: str = "mtl,single"):
    if task == "propaganda_max_pool":
        raise SystemExit("label_budget takes a single label column, not the propaganda pool")
    done = _done(LABEL_BUDGET_DIR)
    full_done = _done(HIDE_ONE_DIR)
    calls = []
    for seed in (int(s) for s in seeds.split(",")):
        for share in (float(x) for x in shares.split(",")):
            for arm in arms.split(","):
                name = _label_budget_name(arm, task, share, seed)
                if name in done or (arm == "mtl" and share == 1
                                    and f"famo_consf1_m5_full_s{seed}" in full_done):
                    continue
                kw = {**_hide_one_kwargs("famo_consf1_m5" if arm == "mtl" else "plain",
                                         seed, name),
                      "tag": f"label_budget/{name}", "label_share": f"{task}={share}"}
                if arm == "single":
                    kw.update(tasks=task, drop_unlabeled_train=True,
                              repeat_train=max(1, round(1 / share)))
                calls.append((name, replicate.spawn(**kw)))
    print(f"launching {len(calls)}: {[n for n, _ in calls]}", flush=True)
    _spawn_all(calls)


@app.local_entrypoint()
def untrained_probe(seeds: str = "42"):
    """Control for the hide-one-label probes: the same probes on the model
    before any training (pretrained small BERT). No training run."""
    _spawn_all([(f"untrained_probe_s{s}", replicate.spawn(
        **_hide_one_kwargs("plain", int(s), f"untrained_probe_s{s}"),
        probe_tasks="all", probe_untrained=True)) for s in seeds.split(",")])


@app.local_entrypoint()
def norule_baselines(methods: str = "plain,famo_consf1_m5", seeds: str = "42,1,2"):
    jobs = [(m, int(s)) for m in methods.split(",") for s in seeds.split(",")]
    done = _done(HIDE_ONE_DIR)
    todo = [(m, s) for m, s in jobs if f"{m}_full_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    _spawn_all([(f"{m}_full_s{s}", replicate.spawn(
        **_hide_one_kwargs(m, s, f"{m}_full_s{s}"), probe_tasks="all")) for m, s in todo])


@app.local_entrypoint()
def hide_one(tasks: str = "", seeds: str = "42,1,2", method: str = "plain"):
    outputs = tasks.split(",") if tasks else TABLE3_OUTPUTS
    jobs = [(t, int(s)) for t in outputs for s in seeds.split(",")]
    done = _done(HIDE_ONE_DIR)
    todo = [(t, s) for t, s in jobs if f"{method}_hide_{t}_s{s}" not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    _spawn_all([(f"{method}_hide_{t}_s{s}", replicate.spawn(
        **_hide_one_kwargs(method, s, f"{method}_hide_{t}_s{s}"),
        hide_task=t, probe_tasks="hidden")) for t, s in todo])
