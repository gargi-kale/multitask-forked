"""Train the multi-task model on the Measuring Hate Speech data, alone or
together with the IAC forum posts, with a random share of the hate speech
training labels optionally hidden.

Kept apart from the main experiments: this file only *imports* the model,
loss, data loader and epoch scorer from ``replication/test_no_prop_overlap.py``
so the setup matches the main runs exactly; it changes nothing there.

Setup (same as the main re-tuned runs): small BERT, max length 512, "wide"
heads, lr 1e-4, weight decay 0.01, 5% warmup, dropout 0, batch 256, up to 10
epochs, keep the epoch with the best val F1 (mean over all heads), stop after
3 epochs without improvement, thresholds tuned on val (Youden's J), weighted F1.

``--sources``:
  hatespeech       10 hate speech heads, one task-type group.
  iac              the 8 IAC heads, on the IAC rows of no_prop_overlap only
                   (no Propaganda or IBM rows, so no rule-made labels).
  iac+hatespeech   both groups in one model: shared layers, then one branch
                   per group, as the main model does for its three groups.
                   IAC rows carry only IAC labels, hate speech rows only hate
                   speech labels; the rest are missing (masked in the loss).

``--branches`` splits the hate speech heads into two branches: the
``--extreme-items`` (default violence, genocide) and the rest. Both keep the
one "hatespeech" task type, so every head's loss weight is unchanged; only
the layers change (with IAC: three branches).

``--hide-task t`` hides one label (a hate speech item or an IAC label) on
every train row; its head is still scored but left out of epoch selection.
``--probe-tasks`` ("all", "hidden" or a comma list) fits a linear probe per
label on the frozen model after training (``replication/hidden_label_probe.py``).

Masking: ``--mask-share p`` hides, for each hate speech item separately, a
random p share of the hate speech training rows' labels (exactly round(p * n)
cells per item), drawn with ``--seed``, so runs of the same seed hide the same
cells. Val and test are never masked. IAC labels are never hidden.

After training (best epoch restored) the hate speech train rows are scored:
  * hidden cells   - labels the model never saw (masked runs);
  * visible cells  - labels it trained on;
  * for an unmasked run, the cells that each standard share (0.25/0.5/0.75)
    would have hidden with this seed, as the "had the label" reference.

Usage (inside the Modal container, cwd = repo root):
    python hatespeech/run_hatespeech.py --data-dir /data/hatespeech \
        --mask-share 0.5 --seed 42 --out /outputs/hatespeech_mask/mask0.5_s42
    python hatespeech/run_hatespeech.py --sources iac+hatespeech \
        --data-dir /data/hatespeech --iac-dir /data/paper --seed 42 --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "replication"))

ITEMS = ["sentiment", "respect", "insult", "humiliate", "status",
         "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"]
SOURCES = ("hatespeech", "iac", "iac+hatespeech")
METHODS = ("plain", "famo", "famo_consf1_m5")
# famo_consf1_m5 as launched by modal_replicate.py::consistency_filter:
# loss_balancer famo, mlm_lambda 0, mlm_mask_prob 0.05, consistency_weight 1,
# consistency_keep 0.1 (rampup 1 epoch, floor 0, masked_label_weight 0: the defaults).
CONSF1_M5 = dict(mlm_lambda=0.0, mask_prob=0.05, consistency_weight=1.0, consistency_keep=0.1,
                 consistency_rampup_epochs=1.0, consistency_floor=0.0, masked_label_weight=0.0)
STANDARD_SHARES = (0.25, 0.5, 0.75, 0.9, 0.95)
DEFAULT_CUT04 = ">2"
DEFAULT_CUT02 = ">=1"
SPLITS = ("train", "val", "test")


def parse_cut(spec: str):
    for op in (">=", ">"):
        if spec.startswith(op):
            return op, float(spec[len(op):])
    raise ValueError(f"cut-off must look like '>2' or '>=1', got {spec!r}")


def binarise(df: pd.DataFrame, cut04: str, cut02: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ITEMS:
        op, v = parse_cut(cut02 if c == "hatespeech" else cut04)
        out[f"{c}_bi"] = ((df[c] >= v) if op == ">=" else (df[c] > v)).astype(np.float32)
    return out


def hidden_cells(n_rows: int, share: float, seed: int) -> np.ndarray:
    """[n_rows, 10] bool, True = hidden. Exactly round(share * n_rows) cells
    per item, drawn independently per item."""
    rng = np.random.default_rng(seed)
    hide = np.zeros((n_rows, len(ITEMS)), dtype=bool)
    k = int(round(share * n_rows))
    for j in range(len(ITEMS)):
        hide[rng.choice(n_rows, size=k, replace=False), j] = True
    return hide


def cell_scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Weighted F1 at the val threshold and AUC, as the library scores them."""
    from sklearn.metrics import f1_score, roc_auc_score
    out = {"n": int(y.size), "yes_share": float(y.mean()) if y.size else None}
    if y.size == 0 or len(np.unique(y)) < 2:
        return {**out, "f1": None, "auc": None}
    out["f1"] = float(f1_score(y, (p > threshold).astype(int), average="weighted", zero_division=0))
    out["auc"] = float(roc_auc_score(y, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="hatespeech", choices=SOURCES)
    ap.add_argument("--method", default="plain", choices=METHODS,
                    help="plain: fixed task weights. famo: FAMO task weighting. famo_consf1_m5: "
                         "FAMO + consistency on a 5%%-word-masked copy, 10%% most confident "
                         "cells per side per head, weight 1 (no masked-word loss) - the 69.9 setup")
    ap.add_argument("--data-dir", required=True, help="folder with hatespeech_{train,val,test}.csv")
    ap.add_argument("--iac-dir", default=None, help="folder with no_prop_overlap_{train,val,test}.csv")
    ap.add_argument("--out", required=True, help="output path prefix (writes .json and .npz)")
    ap.add_argument("--mask-share", type=float, default=0.0)
    ap.add_argument("--items", default="",
                    help="comma-separated hate speech items to train (default all 10); "
                         "one item = a single-task model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cut04", default=DEFAULT_CUT04, help="yes/no cut-off for the 0-4 items")
    ap.add_argument("--cut02", default=DEFAULT_CUT02, help="yes/no cut-off for hatespeech (0-2)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--verbose", type=int, default=2)
    ap.add_argument("--branches", action="store_true",
                    help="two hate speech branches: --extreme-items and the rest")
    ap.add_argument("--extreme-items", default="violence,genocide")
    ap.add_argument("--hide-task", default="",
                    help="hide this label (e.g. violence_bi, nasty_nice_bi) on every train row")
    ap.add_argument("--label-share", default="",
                    help="label=share: keep that share of the label's train labels (seeded, "
                         "nested across shares); for an IAC label the kept rows are the ones "
                         "test_no_prop_overlap.py --label-share --no-rule-labels keeps")
    ap.add_argument("--probe-tasks", default="",
                    help="'all', 'hidden' or a comma list: linear probes after training")
    args = ap.parse_args()
    use_hs = "hatespeech" in args.sources.split("+")
    use_iac = "iac" in args.sources.split("+")
    if not 0.0 <= args.mask_share < 1.0:
        raise SystemExit("--mask-share must be in [0, 1)")
    items = args.items.split(",") if args.items else list(ITEMS)
    if [c for c in items if c not in ITEMS]:
        raise SystemExit(f"--items: unknown {[c for c in items if c not in ITEMS]}; choose from {ITEMS}")
    if args.items and (not use_hs or args.mask_share):
        raise SystemExit("--items needs hatespeech in --sources and no --mask-share")
    if args.mask_share and not use_hs:
        raise SystemExit("--mask-share hides hate speech labels: it needs hatespeech in --sources")
    if use_iac and not args.iac_dir:
        raise SystemExit("--sources with iac needs --iac-dir")
    extreme = [c for c in args.extreme_items.split(",") if c] if args.branches else []
    if [c for c in extreme if c not in items] or (args.branches and len(extreme) in (0, len(items))):
        raise SystemExit(f"--extreme-items must be a proper subset of the trained items {items}")
    if args.branches and not use_hs:
        raise SystemExit("--branches needs hatespeech in --sources")
    if args.hide_task and args.mask_share:
        raise SystemExit("--hide-task and --mask-share are separate experiments")
    if args.probe_tasks == "hidden" and not args.hide_task:
        raise SystemExit("--probe-tasks hidden needs --hide-task")
    label_share = {}
    for item in filter(None, args.label_share.split(",")):
        col, share = item.split("=")
        if not 0 < float(share) <= 1:
            raise SystemExit(f"--label-share {item!r}: share must be in (0, 1]")
        if float(share) < 1:
            label_share[col] = float(share)
    if label_share and (args.mask_share or args.hide_task):
        raise SystemExit("--label-share is a separate experiment from --mask-share / --hide-task")

    import tensorflow as tf
    from transformers import AutoTokenizer
    from multitask.config import ModelConfig
    from test_no_prop_overlap import (
        DISCOURSE_BINARY_COLUMNS, ENCODER_CONFIGS, FAMO, PAPER_ARCHITECTURES,
        PAPER_TRAINING_CONFIG, BalancedLossTrainer, BalancedMultiTaskModel, EpochScorer,
        LabelAwareMaskedModel, LabelAwareMaskedTrainer, MultiTaskModel,
        PooledOutputLossTrainer, attach_mlm_balancing,
        compute_full_train_binary_class_weights, load_splits, run_experiment,
    )

    # The main trainers count FAMO's tasks as heads - pooled heads + 1, i.e.
    # they assume a pooled propaganda output. With no pool that is one too
    # many (11 weights for 10 losses), so these two redo fit() with the plain
    # head count; everything else is the main code unchanged.
    class NoPoolFamoTrainer(BalancedLossTrainer):
        def fit(self, train_dataset, *a, **kw):
            self.model(next(iter(train_dataset))[0], training=False)  # create all variables
            self.model._balancer = FAMO(len(self.model.task_names))
            print(f"Loss balancer: famo over {len(self.model.task_names)} tasks", flush=True)
            return PooledOutputLossTrainer.fit(self, train_dataset, *a, **kw)

    class NoPoolLabelAwareTrainer(LabelAwareMaskedTrainer):
        def fit(self, train_dataset, *a, **kw):
            steps = int(train_dataset.cardinality())
            self.prepare(train_dataset)
            attach_mlm_balancing(self.model, "famo", "fixed", len(self.model.task_names))
            self.model.attach_label_aware(
                self.masked_label_weight, self.consistency_weight,
                round(self.consistency_rampup_epochs * steps), self.consistency_keep,
                {}, self.consistency_floor)
            return PooledOutputLossTrainer.fit(self, train_dataset, *a, **kw)

    tf.keras.utils.set_random_seed(args.seed)
    hs_other = [f"{c}_bi" for c in items if c not in extreme] if use_hs else []
    hs_extreme = [f"{c}_bi" for c in items if c in extreme]
    hs_tasks = hs_other + hs_extreme  # branch order; hide matrices follow it
    iac_tasks = list(DISCOURSE_BINARY_COLUMNS) if use_iac else []
    # Both hate speech branches keep the "hatespeech" task type: same loss weights.
    groups = [(g, ty) for g, ty in ((hs_other, "hatespeech"), (hs_extreme, "hatespeech"),
                                    (iac_tasks, "discourse")) if g]
    tasks = [t for g, _ in groups for t in g]
    if args.hide_task and args.hide_task not in tasks:
        raise SystemExit(f"--hide-task {args.hide_task!r} is not a trained label: {tasks}")
    hs_cols = [ITEMS.index(t[:-3]) for t in hs_tasks]  # ITEMS column of each hs_tasks entry
    print(f"Sources: {args.sources}; outputs: {tasks}"
          f"{f'; branches {[g for g, _ in groups]}' if args.branches else ''}"
          f"{f'; hidden label {args.hide_task}' if args.hide_task else ''}", flush=True)

    # ---- hate speech labels and masking -----------------------------------
    hs_frames, hs_labels = {}, {}
    hide = np.zeros((0, len(ITEMS)), dtype=bool)
    if use_hs:
        hs_frames = {s: pd.read_csv(f"{args.data_dir}/hatespeech_{s}.csv") for s in SPLITS}
        hs_labels = {s: binarise(df, args.cut04, args.cut02) for s, df in hs_frames.items()}
        n_hs = len(hs_frames["train"])
        hide = hidden_cells(n_hs, args.mask_share, args.seed)[:, hs_cols] if args.mask_share else \
            np.zeros((n_hs, len(hs_tasks)), dtype=bool)
        print(f"Hate speech: cut-offs 0-4 items '{args.cut04}', hatespeech '{args.cut02}'; "
              f"train {n_hs:,} rows, hiding {int(hide.sum()):,} of {hide.size:,} label cells "
              f"({100 * args.mask_share:g}% per item)", flush=True)
        for j, t in enumerate(hs_tasks):
            y = hs_labels["train"][t].to_numpy()
            print(f"  {t:18s} yes {100 * y.mean():5.1f}%   visible {int((~hide[:, j]).sum()):,}  "
                  f"hidden {int(hide[:, j].sum()):,}"
                  f" (yes {100 * y[hide[:, j]].mean() if hide[:, j].any() else 0:5.1f}%)")

    # ---- the combined CSVs load_splits reads ------------------------------
    tmp = tempfile.mkdtemp(prefix="hs_")
    train_true_all = None  # train labels before any hiding, for the probes
    for s in SPLITS:
        parts = []
        if use_hs:
            lab = hs_labels[s][hs_tasks].copy()
            if s == "train":
                lab = lab.mask(pd.DataFrame(hide, index=lab.index, columns=hs_tasks))  # hidden -> NaN -> -1
            parts.append(pd.concat([hs_frames[s][["text"]], lab], axis=1).assign(dataset="hatespeech"))
        if use_iac:
            iac = pd.read_csv(f"{args.iac_dir}/no_prop_overlap_{s}.csv")
            if s == "train":
                iac_full_train = iac
            iac = iac[iac["dataset"] == "IAC"]
            parts.append(iac[["text", *iac_tasks]].assign(dataset="IAC"))
        df = pd.concat(parts, ignore_index=True)  # the other group's columns -> NaN -> -1
        if s == "train":
            if use_hs:  # undo the cell masking for the probes' copy
                df_true = df.copy()
                df_true.loc[:len(hs_frames["train"]) - 1, hs_tasks] = hs_labels["train"][hs_tasks].to_numpy()
            else:
                df_true = df.copy()
            train_true_all = df_true
            if args.hide_task:
                print(f"train: hid all {int(df[args.hide_task].notna().sum()):,} labels of "
                      f"{args.hide_task}", flush=True)
                df[args.hide_task] = np.nan
            for col, share in label_share.items():
                if col not in tasks:
                    raise SystemExit(f"--label-share {col!r} is not a trained label: {tasks}")
                if col in iac_tasks:
                    # Same rule as load_splits(train_label_share=...) on the main data with
                    # the rule labels removed: positions in the full train CSV.
                    full = iac_full_train
                    lab = pd.to_numeric(full[col], errors="coerce").notna().to_numpy()
                    rows = np.flatnonzero(lab & (full["dataset"] != "Propaganda").to_numpy())
                    order = np.random.default_rng(args.seed or 0).permutation(rows)
                    n_keep = max(1, int(round(share * rows.size)))
                    iac_pos = np.flatnonzero((full["dataset"] == "IAC").to_numpy())
                    offset = len(df) - iac_pos.size  # the IAC rows are the last part
                    cut = offset + np.flatnonzero(np.isin(iac_pos, order[n_keep:]))
                else:
                    rows = np.flatnonzero(df[col].notna().to_numpy())
                    order = np.random.default_rng(args.seed or 0).permutation(rows)
                    n_keep = max(1, int(round(share * rows.size)))
                    cut = order[n_keep:]
                df.loc[cut, col] = np.nan
                left = df[col].dropna()
                print(f"train: kept {len(left):,} of {rows.size:,} labels of {col} "
                      f"({100 * share:g}%; {int((left == 1).sum()):,} yes)", flush=True)
        df.to_csv(f"{tmp}/hs_{s}.csv", index=False)
        print(f"{s}: {len(df):,} rows " + ", ".join(
            f"{k} {v:,}" for k, v in df["dataset"].value_counts().items()), flush=True)

    encoder_config = ENCODER_CONFIGS["small-bert"]
    train_ds, val_ds, test_ds = load_splits(
        label_columns=tasks, encoder_config=encoder_config, batch_size=args.batch_size,
        data_dir=tmp, prefix="hs_", max_length=args.max_length, seed=args.seed)
    # Class weights from the visible train labels only (same rule as the main runs).
    class_weights = compute_full_train_binary_class_weights(
        data_dir=tmp, task_names=tasks, prefix="hs_")

    # ---- model and training: the main re-tuned setup ----------------------
    shared, branches = PAPER_ARCHITECTURES["wide"]
    config = ModelConfig(
        task_structure=[[len(groups)], [len(g) for g, _ in groups]],
        shared_layer_sizes=shared, branch_layer_sizes=branches,
        task_names=tasks, task_types=["binary"] * len(tasks),
        num_classes_per_task=[2] * len(tasks), output_pools={},
        encoder_config=encoder_config)
    training_config = replace(
        PAPER_TRAINING_CONFIG, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout_rate=0.0, weight_decay_rate=0.01,
        warmup_percent=5, early_stopping=False)
    scorer = EpochScorer(val_ds, test_ds, monitor="f1", patience=args.patience,
                         outputs=[t for t in tasks if t != args.hide_task])

    kw = dict(pooled_tasks=[], pooled_class_weights={0: 1.0, 1: 1.0})
    if args.method == "plain":
        model_cls = MultiTaskModel
        make_trainer = lambda model, cfg: PooledOutputLossTrainer(model, cfg, **kw)
    elif args.method == "famo":
        model_cls = BalancedMultiTaskModel
        make_trainer = lambda model, cfg: NoPoolFamoTrainer(model, cfg, balancer="famo", **kw)
    else:
        model_cls = LabelAwareMaskedModel
        make_trainer = lambda model, cfg: NoPoolLabelAwareTrainer(
            model, cfg, balancer="famo", mlm_mode="fixed", **CONSF1_M5, **kw)
    print(f"Method: {args.method}", flush=True)

    def trainer_factory(model, cfg):
        scorer.trainer = make_trainer(model, cfg)
        return scorer.trainer

    results = run_experiment(
        config, train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        task_type_mapping={t: ty for g, ty in groups for t in g},
        class_weights=class_weights, optimize_thresholds=True, verbose=args.verbose,
        show_structure=False, training_config=training_config, test_ds=test_ds,
        trainer_factory=trainer_factory, model_cls=model_cls, callbacks=[scorer])
    trainer = scorer.trainer
    model = trainer.model
    thresholds = {t: float(trainer.optimal_thresholds[t]) for t in tasks}

    # ---- score the hate speech train rows (best epoch restored) -----------
    train_scores = {}
    train_prob = train_true = None
    if use_hs:
        n_hs = len(hs_frames["train"])
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)  # as load_splits
        enc = tok([str(x) for x in hs_frames["train"]["text"]], padding="max_length",
                  truncation=True, max_length=args.max_length, return_tensors="np")
        train_inputs = tf.data.Dataset.from_tensor_slices(
            {"input_ids": enc["input_ids"].astype(np.int32),
             "attention_mask": enc["attention_mask"].astype(np.int32)}).batch(args.batch_size)
        train_pred = model.predict(train_inputs, verbose=0)
        train_prob = np.stack([np.asarray(train_pred[t]).reshape(-1) for t in hs_tasks], axis=1)
        train_true = hs_labels["train"][hs_tasks].to_numpy()

        def score_cells(cells: np.ndarray) -> dict:
            return {t: cell_scores(train_true[cells[:, j], j], train_prob[cells[:, j], j], thresholds[t])
                    for j, t in enumerate(hs_tasks)}

        train_scores["visible"] = score_cells(~hide)
        if args.mask_share:
            train_scores["hidden"] = score_cells(hide)
        elif len(hs_tasks) == len(ITEMS):  # reference: the same cells each masked run of this seed hides
            train_scores["reference"] = {f"{p:g}": score_cells(hidden_cells(n_hs, p, args.seed)[:, hs_cols])
                                         for p in STANDARD_SHARES}

    probes = None
    if args.probe_tasks:
        from hidden_label_probe import run_probes

        def ds_labels(ds):
            ys = [y for _, y in ds]
            return {k: np.concatenate([np.asarray(y[k]).reshape(-1) for y in ys]) for k in ys[0]}

        names = (list(tasks) if args.probe_tasks == "all" else
                 [args.hide_task] if args.probe_tasks == "hidden" else args.probe_tasks.split(","))
        probes = run_probes(
            model, names, texts=train_true_all["text"].fillna("").astype(str).tolist(),
            train_labels={t: train_true_all[t].fillna(-1).to_numpy(dtype=np.float32) for t in names},
            val_ds=val_ds, val_labels=ds_labels(val_ds), test_ds=test_ds,
            test_labels=ds_labels(test_ds), max_length=args.max_length,
            batch_size=args.batch_size, seed=args.seed)

    def split_scores(res):
        m = res.get("task_metrics", {})
        return {t: {"f1": float(m[t]["f1"]), "auc": float(m[t]["auc"])} for t in tasks}

    out = {
        "sources": args.sources, "method": args.method, "items": items, "tasks": tasks,
        "mask_share": args.mask_share, "seed": args.seed,
        "branches": [g for g, _ in groups] if args.branches else None,
        "hide_task": args.hide_task or None, "probes": probes,
        "label_share": label_share or None,
        "cut04": args.cut04, "cut02": args.cut02,
        "settings": {"learning_rate": args.learning_rate, "batch_size": args.batch_size,
                     "max_length": args.max_length, "epochs": args.epochs,
                     "patience": args.patience, "arch": "wide", "dropout": 0.0},
        "n_train": len(hs_frames["train"]) if use_hs else 0, "n_hidden": int(hide.sum()),
        "class_weights": {t: {str(k): float(v) for k, v in w.items()}
                          for t, w in zip(tasks, class_weights)},
        "thresholds": thresholds,
        "best_epoch": scorer.best_epoch,
        "epoch_scores": scorer.records,
        "val": split_scores(results),
        "test": split_scores(results["test"]),
        "train_cells": train_scores,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.json", "w") as f:
        json.dump(out, f, indent=1)
    if use_hs:
        np.savez_compressed(f"{args.out}_predictions.npz", train_prob=train_prob.astype(np.float32),
                            train_true=train_true.astype(np.int8), hidden=hide)

    def mean(d, k, names):
        vals = [d[t][k] for t in names if d.get(t, {}).get(k) is not None]
        return 100 * float(np.mean(vals)) if vals else float("nan")

    print(f"\nBest epoch {scorer.best_epoch}.", flush=True)
    for name, g in (("hate speech", hs_tasks), ("IAC", iac_tasks)):
        if g:
            print(f"Test {name} heads: F1 {mean(out['test'], 'f1', g):.2f}  "
                  f"AUC {mean(out['test'], 'auc', g):.2f}", flush=True)
    if args.mask_share:
        print(f"Hidden train cells F1 {mean(train_scores['hidden'], 'f1', hs_tasks):.2f}  "
              f"AUC {mean(train_scores['hidden'], 'auc', hs_tasks):.2f}", flush=True)
    print(f"Wrote {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
