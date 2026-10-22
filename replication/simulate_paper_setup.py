"""
Paper-shaped simulation for the `multitask` library — no CSVs, no downloads.

Builds a synthetic dataset with the same *structure* as the paper replication
(`run_paper_replication` in replication/test_no_prop_overlap.py):

  * 27 binary tasks: 18 propaganda techniques, WA_bi (argument quality),
    8 IAC discourse dimensions
  * hierarchy [[3], [18, 1, 8]], shared [64, 32], branches [[32, 16], [16, 8]]
  * max-pool over the 18 propaganda heads
  * three "source corpora": each row has labels only for its own corpus's
    tasks, everything else is masked (-1) — exactly like the merged CSV
  * per-task class imbalance taken from the paper's Table 1 and from the class
    weights printed in cluster_runs/paper_replication.md
  * 80/10/10 split; thresholds tuned on VAL, metrics reported on held-out TEST

Inputs are 128-d "precomputed embeddings" (same width as small BERT) generated
from a low-dimensional latent signal, so tasks share structure the multi-task
model can exploit. Numbers produced here say nothing about the paper's results;
they show the pipeline (loss, weighting, masking, pooling, thresholds,
metrics) runs end to end and behaves sensibly.

    python simulate_paper_setup.py --repo /path/to/multitask --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

PROP = [
    "Loaded_Language", "Slogans", "Black-and-White_Fallacy", "Flag-Waving",
    "Name_Calling", "Whataboutism", "Causal_Oversimplification",
    "Exaggeration_Minimisation", "Doubt", "Appeal_to_Authority", "Repetition",
    "Appeal_to_fear-prejudice", "Red_Herring", "Thought-terminating_Cliches",
    "Reductio_ad_hitlerum", "Bandwagon",
    "Obfuscation_Intentional_Vagueness_Confusion", "Straw_Men",
]
# class_1 weights printed in paper_replication.md -> prevalence p = 1 / (2 w1)
PROP_W1 = [1.44375, 26.6105, 28.4045, 11.2606, 2.56, 46.8148, 15.5569, 6.33584,
           5.88591, 23.1927, 5.26667, 12.2421, 91.9273, 41.7851, 55.5604,
           297.412, 280.889, 421.333]
DISC = ["disagree_agree_bi", "attacking_respectful_bi", "emotion_fact_bi",
        "nasty_nice_bi", "personal_audience_bi", "defeater_undercutter_bi",
        "negotiate_attack_bi", "questioning_asserting_bi"]
# paper Table 1 balance (share of class 1)
DISC_P = [0.79, 0.34, 0.59, 0.27, 0.75, 0.62, 0.56, 0.34]
# last four IAC dims are annotated on roughly a third of IAC posts (Table 1 N)
DISC_COVERAGE = [1.0, 1.0, 1.0, 1.0, 0.35, 0.35, 0.35, 0.35]
WA_P = 0.94

TASKS = PROP + ["WA_bi"] + DISC
GROUP = {**{t: "propaganda_binary" for t in PROP}, "WA_bi": "wa_bi",
         **{t: "discourse" for t in DISC}}
PREV = dict(zip(PROP, [1 / (2 * w) for w in PROP_W1]))
PREV.update({"WA_bi": WA_P, **dict(zip(DISC, DISC_P))})


def simulate(n: int, rng: np.random.Generator, latent_dim=12, emb_dim=128,
             signal=1.0, W=None, group_vecs=None, task_vecs=None, cut=None):
    """Return (X, labels dict) for n rows. Shared params are created on first call."""
    if W is None:
        W = rng.standard_normal((latent_dim, emb_dim)) / np.sqrt(latent_dim)
        group_vecs = {g: rng.standard_normal(latent_dim)
                      for g in ("propaganda_binary", "wa_bi", "discourse")}
        task_vecs = {t: 0.8 * group_vecs[GROUP[t]] + 0.6 * rng.standard_normal(latent_dim)
                     for t in TASKS}

    # corpus mix roughly proportional to paper Table 1 sizes (PTC / IBM / IAC)
    source = rng.choice(["ptc", "ibm", "iac"], size=n, p=[0.26, 0.41, 0.33])
    z = rng.standard_normal((n, latent_dim))
    X = (z @ W + 0.5 * rng.standard_normal((n, emb_dim))).astype(np.float32)

    scores = {t: signal * z @ task_vecs[t] + rng.standard_normal(n) for t in TASKS}
    if cut is None:  # fix thresholds on the first (train) draw so prevalence carries over
        cut = {t: np.quantile(scores[t], 1 - PREV[t]) for t in TASKS}

    labels = {}
    for t in TASKS:
        y = (scores[t] > cut[t]).astype(np.float32)
        if t in PROP:
            valid = source == "ptc"
        elif t == "WA_bi":
            valid = source == "ibm"
        else:
            cov = DISC_COVERAGE[DISC.index(t)]
            valid = (source == "iac") & (rng.random(n) < cov)
        labels[t] = np.where(valid, y, -1.0).astype(np.float32)
    return X, labels, (W, group_vecs, task_vecs, cut)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to a multitask checkout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="sim_results.json")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="disable early stopping (see min_delta note in README)")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import tensorflow as tf
    from multitask import MultiTaskModel, Trainer
    from multitask.config import (EncoderConfig, EncoderInputType, ModelConfig,
                                  TrainingConfig)
    from sklearn.metrics import f1_score

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n_tr = args.n_train
    n_va = n_te = n_tr // 8
    Xtr, ytr, params = simulate(n_tr, rng)
    Xva, yva, _ = simulate(n_va, rng, W=params[0], group_vecs=params[1],
                           task_vecs=params[2], cut=params[3])
    Xte, yte, _ = simulate(n_te, rng, W=params[0], group_vecs=params[1],
                           task_vecs=params[2], cut=params[3])

    def ds(X, y, shuffle=False):
        d = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            d = d.shuffle(len(X), seed=args.seed)
        return d.batch(256).prefetch(tf.data.AUTOTUNE)

    config = ModelConfig(
        task_structure=[[3], [18, 1, 8]],
        shared_layer_sizes=[64, 32],
        branch_layer_sizes=[[32, 16], [16, 8]],
        task_names=TASKS,
        task_types=["binary"] * len(TASKS),
        num_classes_per_task=[2] * len(TASKS),
        output_pools={"propaganda_max_pool": {
            "tasks": PROP, "strategy": "max", "task_type": "binary", "num_classes": 2}},
        encoder_config=EncoderConfig(encoder_identifier=None,
                                     encoder_input_type=EncoderInputType.PRECOMPUTED,
                                     embedding_dim=128),
    )
    # paper hyperparameters (PAPER_TRAINING_CONFIG in the replication script)
    tcfg = TrainingConfig(optimizer_type="adamw", learning_rate=3e-4,
                          weight_decay_rate=0.01, dropout_rate=0.4,
                          batch_size=256, warmup_percent=5, epochs=args.epochs,
                          early_stopping=not args.no_early_stop)
    model = MultiTaskModel(config, embedding_dim=128, dropout_rate=tcfg.dropout_rate)
    _ = model(tf.zeros((1, 128)), training=False)
    trainer = Trainer(model, tcfg)

    t0 = time.time()
    history, thresholds = trainer.fit(
        ds(Xtr, ytr, shuffle=True), ds(Xva, yva), num_tasks=len(TASKS),
        task_type_mapping=GROUP, optimize_thresholds=True, verbose=2)
    train_secs = time.time() - t0

    # thresholds were tuned on VAL; score on held-out TEST
    res = trainer.evaluate(ds(Xte, yte), thresholds=thresholds)
    probs = res["probabilities_dict"]

    rows = []
    for t in TASKS:
        y = yte[t]
        m = y != -1
        p = np.asarray(probs[t]).reshape(len(y), -1)[:, 0][m]
        th = thresholds[t] if not isinstance(thresholds[t], list) else thresholds[t][0]
        yhat = (p > th).astype(int)
        tm = res["task_metrics"][t]
        rows.append({
            "task": t, "n_test": int(m.sum()), "pos_rate": float(y[m].mean()),
            "weighted_f1": tm.get("f1"), "macro_f1": float(f1_score(y[m], yhat, average="macro", zero_division=0)),
            "positive_f1": float(f1_score(y[m], yhat, average="binary", zero_division=0)),
            "accuracy": tm.get("accuracy"), "auc": tm.get("auc"),
        })

    out = {
        "seed": args.seed, "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "epochs_run": len(history.history["loss"]),
        "train_loss": [round(v, 4) for v in history.history["loss"]],
        "val_loss": [round(v, 4) for v in history.history["val_loss"]],
        "train_seconds": round(train_secs, 1),
        "overall_metrics_test": res["overall_metrics"],
        "pool_metrics_test": res.get("pool_metrics", {}),
        "per_task_test": rows,
        "mean_macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
        "mean_positive_f1": float(np.mean([r["positive_f1"] for r in rows])),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: out[k] for k in ("epochs_run", "train_seconds",
          "overall_metrics_test", "mean_macro_f1", "mean_positive_f1")}, indent=2, default=float))


if __name__ == "__main__":
    main()
