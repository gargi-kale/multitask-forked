"""Linear probes on a trained multi-task model, for hide-one-label runs.

When a label is hidden from every train row, its own head gets no training
signal and scores at chance by construction. The question a hide-one-label
run asks is instead: did learning the *other* labels give the model a
representation from which the hidden label can be read? A probe answers it:
freeze the trained model, take a representation of each text, and fit a
logistic regression for the label on train rows (the labels the network
never trained on), with the yes/no cut-off tuned on val (Youden's J, as the
library does) and weighted F1 / AUC on test. Running the same probe on a
model that did see the label gives the reference.

Three representations are probed, from the bottom of the model up:
  encoder  the BERT pooled output (what every head starts from);
  shared   the output of the shared layers;
  group    the output of the label's own task-type branch (trained by the
           other labels of its group; untrained if the group has no other label).

Used by ``multitask train --probes``, ``test_no_prop_overlap.py --probe-tasks`` and
``hatespeech/run_hatespeech.py --probe-tasks``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

PROBE_MAX_TRAIN = 20_000  # labelled train rows per probe, sampled with the run seed
REPRESENTATIONS = ("encoder", "shared", "group")


def group_of(model, task: str) -> int:
    """Index of the first-level branch (task-type group) holding ``task``;
    a pool name maps to its first constituent's group."""
    names = list(model.task_names)
    if task not in names:
        pool = model.config.output_pools.get(task)
        if not pool:
            raise KeyError(f"{task!r} is neither a task nor a pool of this model")
        task = pool["tasks"][0]
    idx, start = names.index(task), 0
    for g, size in enumerate(model.config.task_structure[1]):
        if idx < start + size:
            return g
        start += size
    raise ValueError(f"{task!r} is outside task_structure {model.config.task_structure}")


def representations(model, inputs) -> Dict[str, np.ndarray]:
    """{'encoder', 'shared', 'group<g>' for every group: [n, d]} for a
    batched dataset of model inputs (dicts, or (inputs, labels) pairs)."""
    import tensorflow as tf

    out: Dict[str, List[np.ndarray]] = {}
    n_groups = model.config.task_structure[0][0]
    for batch in inputs:
        x = batch[0] if isinstance(batch, tuple) else batch
        h = model._encode_input(x, training=False)
        reps = {"encoder": h}
        for layer in model.shared_layers:
            h = layer(h, training=False)
        reps["shared"] = h
        # First branch level: every group branch starts from the shared output
        # (as MultiTaskModel.forward_from_encoding does at level 0).
        for g in range(n_groups):
            z = h
            for layer in model.branch_layers[0][g]:
                z = layer(z, training=False)
            reps[f"group{g}"] = z
        for k, v in reps.items():
            out.setdefault(k, []).append(tf.convert_to_tensor(v).numpy().astype(np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    t = thr[int(np.argmax(tpr - fpr))]
    return float(t) if np.isfinite(t) else float(np.clip(np.nanmax(p), 0.0, 1.0))


def fit_probe(x_tr, y_tr, x_va, y_va, x_te, y_te, seed: int) -> dict:
    """Standardised logistic regression (balanced class weights), cut-off on
    val, weighted F1 and AUC on val and test, as the library scores heads."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {"n_train": int(y_tr.size), "train_yes": float(y_tr.mean()) if y_tr.size else None}
    if min(len(np.unique(y_tr)), len(np.unique(y_va)), len(np.unique(y_te))) < 2:
        return {**out, "error": "a split has one class"}
    clf = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=seed))
    clf.fit(x_tr, y_tr)
    p_va, p_te = clf.predict_proba(x_va)[:, 1], clf.predict_proba(x_te)[:, 1]
    t = youden_threshold(y_va, p_va)
    for split, y, p in (("val", y_va, p_va), ("test", y_te, p_te)):
        out[split] = {"f1": float(f1_score(y, (p > t).astype(int), average="weighted",
                                           zero_division=0)),
                      "auc": float(roc_auc_score(y, p)), "n": int(y.size)}
    out["threshold"] = t
    return out


def run_probes(model, names: Sequence[str], *, texts: Sequence[str],
               train_labels: Dict[str, np.ndarray], val_ds, val_labels: Dict[str, np.ndarray],
               test_ds, test_labels: Dict[str, np.ndarray], max_length: int, batch_size: int,
               seed: int, tokenizer_name: str = "distilbert-base-uncased") -> dict:
    """{name: {rep: probe scores}} for each output in ``names``.

    ``texts`` / ``train_labels`` are the full train split in CSV order, with
    -1 for missing labels; up to PROBE_MAX_TRAIN labelled rows per output are
    sampled (seeded) and encoded once. ``val_ds`` / ``test_ds`` are the
    unshuffled eval datasets and ``*_labels`` their labels in the same order.
    """
    import tensorflow as tf
    from transformers import AutoTokenizer

    rng = np.random.default_rng(seed)
    picks = {}
    for name in names:
        rows = np.flatnonzero(train_labels[name] != -1)
        picks[name] = np.sort(rng.choice(rows, size=min(PROBE_MAX_TRAIN, rows.size), replace=False))
    union = np.unique(np.concatenate(list(picks.values()))) if picks else np.zeros(0, int)
    print(f"Probes: {list(names)}; encoding {union.size:,} train rows", flush=True)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)  # as load_splits
    enc = tok([texts[i] for i in union], padding="max_length", truncation=True,
              max_length=max_length, return_tensors="np")
    train_inputs = tf.data.Dataset.from_tensor_slices(
        {"input_ids": enc["input_ids"].astype(np.int32),
         "attention_mask": enc["attention_mask"].astype(np.int32)}).batch(batch_size)
    reps = {"train": representations(model, train_inputs),
            "val": representations(model, val_ds), "test": representations(model, test_ds)}
    pos = {r: i for i, r in enumerate(union)}

    results = {}
    for name in names:
        g = group_of(model, name)
        tr_idx = np.array([pos[r] for r in picks[name]], dtype=int)
        y_tr = train_labels[name][picks[name]].astype(int)
        va, te = val_labels[name] != -1, test_labels[name] != -1
        y_va, y_te = val_labels[name][va].astype(int), test_labels[name][te].astype(int)
        results[name] = {"group": g}
        for rep in REPRESENTATIONS:
            key = f"group{g}" if rep == "group" else rep
            results[name][rep] = fit_probe(
                reps["train"][key][tr_idx], y_tr, reps["val"][key][va], y_va,
                reps["test"][key][te], y_te, seed)
        s = results[name]
        print(f"  probe {name:28s} test F1 / AUC  " + "  ".join(
            f"{rep} {100 * s[rep]['test']['f1']:.1f}/{100 * s[rep]['test']['auc']:.1f}"
            if "test" in s[rep] else f"{rep} n/a" for rep in REPRESENTATIONS), flush=True)
    return results
