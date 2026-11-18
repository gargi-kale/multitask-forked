"""Per-epoch validation scoring, early stopping and best-epoch restore.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from multitask.train.columns import PAPER_TABLE3_MULTITASK_F1
from multitask.train.mlm import mask_tokens

# ---------------------------------------------------------------------------
# Per-epoch scoring and early stopping on validation F1 / AUC
# ---------------------------------------------------------------------------

EARLY_STOP_METRICS = ("loss", "f1", "auc")


class EpochScorer(tf.keras.callbacks.Callback):
    """After every epoch, scores the ten Table-3 outputs on val (thresholds
    tuned on val, as after training) and on test (with those val thresholds),
    and logs the val means as ``val_f1`` / ``val_auc``.

    Stops once val loss, val F1 and val AUC have all gone ``patience`` epochs
    without improving, then restores the weights of the best epoch by
    ``monitor``. Every stopping rule with patience <= ``patience`` on any of
    the three can therefore be replayed from ``records``, test is never used
    to pick an epoch. ``trainer`` is set once the trainer exists. ``outputs``
    defaults to the Table-3 outputs the model has.

    ``main_output`` (Caruana 1997, Section 7.1: early stopping on the main
    task alone): the epoch is picked on that one output's val F1 / AUC
    instead of the mean, and training stops once that output's val F1 and
    AUC have both stalled ``patience`` epochs. Val loss plays no part then,
    since it sums every task. All outputs are still scored every epoch.

    ``evals_per_epoch`` = k > 1 scores k times per epoch, at equal step
    intervals (the last one at the epoch's end), so a best point between
    epochs can be kept. A record's ``epoch`` is then a fraction (0.25, 0.5,
    ...), ``patience`` counts scorings instead of epochs, and training can
    stop in the middle of an epoch. ``val_loss`` is then worked out here at
    every scoring, as the mean over the val batches of the model's loss.
    (With one scoring per epoch it stays the number Keras logs, which for
    this model is the loss of the last val batch only: see
    test_finetune_recipe.py.)
    """

    def __init__(self, val_ds, test_ds, monitor: str = "f1", patience: int = 3,
                 outputs: Optional[Sequence[str]] = None,
                 main_output: Optional[str] = None, stops: bool = True,
                 evals_per_epoch: int = 1):
        super().__init__()
        if evals_per_epoch < 1:
            raise ValueError("evals_per_epoch must be at least 1")
        self.evals_per_epoch = int(evals_per_epoch)
        if monitor not in EARLY_STOP_METRICS:
            raise ValueError(f"monitor must be one of {EARLY_STOP_METRICS}, got {monitor!r}")
        if main_output and monitor == "loss":
            raise ValueError("main_output needs monitor 'f1' or 'auc' (val loss sums every task)")
        self.val_ds, self.test_ds = val_ds, test_ds
        self.monitor, self.patience = monitor, patience
        self.outputs = list(outputs) if outputs is not None else None
        self.main_output = main_output or None
        self.stops = stops  # False: another callback ends training (self-distillation)
        self.trainer = None
        self.records: List[Dict[str, Any]] = []
        self.best_epoch = None   # the kept record's ``epoch``
        self.best_index = None   # its 1-based position in ``records``

    def _scores(self, ds, thresholds):
        res = self.trainer.evaluate(ds, thresholds=thresholds)
        merged = {**res.get("task_metrics", {}), **res.get("pool_metrics", {})}
        outputs = self.outputs or [k for k in PAPER_TABLE3_MULTITASK_F1 if k in merged]
        return {k: {"f1": float(merged[k]["f1"]), "auc": float(merged[k]["auc"])}
                for k in outputs}

    def on_train_begin(self, logs=None):
        if self.trainer is None:
            raise RuntimeError("EpochScorer.trainer was never set")
        self.records, self.best_epoch, self._best_weights = [], None, None
        self.best_index, self._epoch, self._stopped = None, 0, False
        names = list(self.model.task_names)
        labels = {k: [] for k in names}
        self._val_batch_sizes = []
        for _, y in self.val_ds:
            for k in names:
                labels[k].append(y[k].numpy())
            self._val_batch_sizes.append(len(labels[names[0]][-1]))
        self._val_labels = {k: np.concatenate(v, axis=0) for k, v in labels.items()}
        self._val_labels.update(self.trainer._derive_pooled_labels(self._val_labels))

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        """The in-between scorings of ``evals_per_epoch``; the epoch's last
        one is ``on_epoch_end``."""
        if self.evals_per_epoch == 1 or self._stopped:
            return
        steps = self.params.get("steps")
        if not steps:
            raise RuntimeError("evals_per_epoch needs a known number of steps per epoch")
        k = self.evals_per_epoch
        if batch + 1 in {round(j * steps / k) for j in range(1, k)} and batch + 1 < steps:
            self._score(self._epoch + (batch + 1) / steps, None, None)

    def on_epoch_end(self, epoch, logs=None):
        if self._stopped:  # stopped at an in-between scoring of this epoch
            return
        logs = logs if logs is not None else {}
        self._score(epoch + 1,
                    float(logs["val_loss"]) if self.evals_per_epoch == 1 else None, logs)

    def _val_loss(self, pred) -> float:
        """Mean over the val batches of the model's loss, from predictions
        already made."""
        names = list(self.model.task_names)
        losses, start = [], 0
        for size in self._val_batch_sizes:
            rows = slice(start, start + size)
            y = {k: tf.constant(self._val_labels[k][rows], tf.float32) for k in names}
            p = {k: tf.constant(np.asarray(pred[k])[rows], tf.float32) for k in names}
            losses.append(float(self.model._mtl_loss_fn(y, p)))
            start += size
        return float(np.mean(losses))

    def _score(self, epoch, val_loss, logs):
        """Score val and test now and record it as ``epoch`` (a whole number
        at an epoch's end, a fraction in between)."""
        pred = self.model.predict(self.val_ds, verbose=0)
        thresholds, _ = self.trainer._find_optimal_thresholds(self._val_labels, pred)
        val, test = self._scores(self.val_ds, thresholds), self._scores(self.test_ds, thresholds)
        for k, v in self._val_task_losses(pred).items():
            if k in val:
                val[k]["loss"] = v
        if val_loss is None:
            val_loss = self._val_loss(pred)
        rec = {"epoch": epoch, "val_loss": float(val_loss), "val": val, "test": test}
        for m in ("f1", "auc"):
            rec[f"val_{m}"] = float(np.mean([s[m] for s in val.values()]))
            rec[f"test_{m}"] = float(np.mean([s[m] for s in test.values()]))
            if logs is not None:
                logs[f"val_{m}"] = rec[f"val_{m}"]
        if self.main_output:
            if self.main_output not in val:
                raise ValueError(f"main_output {self.main_output!r} is not scored: {sorted(val)}")
            for m in ("f1", "auc"):
                rec[f"val_main_{m}"] = val[self.main_output][m]
                rec[f"test_main_{m}"] = test[self.main_output][m]
        if getattr(self.model, "teacher", None) is not None:
            rec["teacher"] = self._teacher_scores()
        self.records.append(rec)
        index = len(self.records)  # = the epoch number unless evals_per_epoch > 1
        print(f"Epoch {epoch:g}: val_loss {rec['val_loss']:.4f}  val F1 {100 * rec['val_f1']:.2f}  "
              f"val AUC {100 * rec['val_auc']:.2f}  (test F1 {100 * rec['test_f1']:.2f})",
              flush=True)
        if self.main_output:
            print(f"Epoch {epoch:g}: main task {self.main_output} val F1 "
                  f"{100 * rec['val_main_f1']:.2f}  val AUC {100 * rec['val_main_auc']:.2f}",
                  flush=True)
        if "teacher" in rec:
            print(f"Epoch {epoch:g}: EMA teacher val F1 {100 * rec['teacher']['val_f1']:.2f}  "
                  f"val AUC {100 * rec['teacher']['val_auc']:.2f}", flush=True)

        if self.best_index is None or self._better(rec, self.records[self.best_index - 1]):
            self.best_index, self.best_epoch = index, epoch
            self._best_weights = self.model.get_weights()
        stop_metrics = ("f1", "auc") if self.main_output else EARLY_STOP_METRICS
        stalled = all(index - self._best_by(m) >= self.patience for m in stop_metrics)
        if stalled and self.stops:
            what = (f"main task {self.main_output} val F1 and AUC both" if self.main_output
                    else "val loss, F1 and AUC all")
            print(f"Epoch {epoch:g}: {what} stalled for {self.patience} "
                  f"{'epochs' if self.evals_per_epoch == 1 else 'scorings'}, stopping",
                  flush=True)
            self.model.stop_training = True
            self._stopped = True

    def _teacher_scores(self):
        """The EMA teacher scored like the student (its own val thresholds),
        by loading its weights into the student for the scoring. A read-out
        only: it plays no part in picking the epoch."""
        student = [v.numpy() for v in self.model.trainable_variables]
        for v, t in zip(self.model.trainable_variables, self.model.teacher.trainable_variables):
            v.assign(t)
        try:
            pred = self.model.predict(self.val_ds, verbose=0)
            thresholds, _ = self.trainer._find_optimal_thresholds(self._val_labels, pred)
            val, test = self._scores(self.val_ds, thresholds), self._scores(self.test_ds, thresholds)
        finally:
            for v, w in zip(self.model.trainable_variables, student):
                v.assign(w)
        rec = {"val": val, "test": test}
        for m in ("f1", "auc"):
            rec[f"val_{m}"] = float(np.mean([s[m] for s in val.values()]))
            rec[f"test_{m}"] = float(np.mean([s[m] for s in test.values()]))
        return rec

    def _val_task_losses(self, pred) -> Dict[str, float]:
        """Each output's own class-weighted mean val loss (the per-task loss
        the model trains on, without nu_k: ``_mtl_task_losses_fn``), over the
        whole val split; outputs with no labelled val row are left out."""
        fn = getattr(self.model, "_mtl_task_losses_fn", None)
        if fn is None:
            return {}
        names = list(self.model.task_names)
        y = {k: tf.constant(self._val_labels[k], tf.float32) for k in names}
        p = {k: tf.constant(np.asarray(pred[k]), tf.float32) for k in names}
        losses, has = fn(y, p)
        return {n: float(v) for n, v, h in
                zip(self.model._mtl_task_loss_names, losses.numpy(), has.numpy()) if h > 0}

    def _key(self, metric):
        if metric == "loss":
            return "val_loss"
        return f"val_main_{metric}" if self.main_output else f"val_{metric}"

    def _better(self, a, b):
        key = self._key(self.monitor)
        if self.monitor == "loss":
            return a[key] < b[key]
        return a[key] > b[key]

    def _best_by(self, metric):
        """Position in ``records`` (1-based; the epoch number unless
        ``evals_per_epoch`` > 1) of the first best value of ``metric`` so far."""
        key = self._key(metric)
        vals = [r[key] for r in self.records]
        return 1 + int(np.argmin(vals) if metric == "loss" else np.argmax(vals))

    def on_train_end(self, logs=None):
        if self._best_weights is not None:
            what = f"{self.main_output} " if self.main_output else ""
            print(f"Restoring weights from epoch {self.best_epoch:g} "
                  f"(best val {what}{self.monitor})", flush=True)
            self.model.set_weights(self._best_weights)


def mask_sweep(scorer: "EpochScorer", rates: Sequence[float], repeats: int = 3,
               seed: int = 0) -> Dict[str, Any]:
    """How much masking the input changes the trained model's val answers.

    Uses the model and trainer of a finished run (best epoch restored) with
    thresholds tuned on the clean val set. For each mask rate and ``repeats``
    random maskings (BERT's rule, as in training; eval mode, so no dropout)
    it scores the masked val set with those thresholds and compares each
    output's answers with the clean ones: F1 / AUC on labelled cells, the
    share of cells whose yes/no answer flips, and the mean absolute score
    change, the last two split into labelled and unlabelled cells."""
    model, trainer = scorer.model, scorer.trainer
    # One fixed copy of val, so clean and masked answers line up row by row.
    xs, ys = [], []
    for x, y in scorer.val_ds:
        xs.append({k: v.numpy() for k, v in x.items()})
        ys.append({k: v.numpy() for k, v in y.items()})
    x_all = {k: np.concatenate([b[k] for b in xs]) for k in xs[0]}
    y_all = {k: np.concatenate([b[k] for b in ys]) for k in ys[0]}
    batch_size = len(xs[0]["input_ids"])
    vocab_size = model.encoder.config.vocab_size

    def dataset(ids):
        return tf.data.Dataset.from_tensor_slices(
            ({**x_all, "input_ids": ids}, y_all)).batch(batch_size)

    clean_ds = dataset(x_all["input_ids"])
    labels = {**y_all, **trainer._derive_pooled_labels(y_all)}
    clean_pred = model.predict(clean_ds, verbose=0)
    thresholds, _ = trainer._find_optimal_thresholds(labels, clean_pred)
    clean = scorer._scores(clean_ds, thresholds)
    outputs = list(clean)

    result = {"rates": list(rates), "repeats": repeats, "clean": clean,
              "thresholds": {k: float(thresholds[k]) for k in outputs}, "by_rate": {}}
    for r in rates:
        runs = []
        for i in range(repeats):
            tf.random.set_seed(seed + 1000 * i + int(round(1e4 * r)))
            masked, _ = mask_tokens(x_all["input_ids"], x_all["attention_mask"], vocab_size, r)
            ds = dataset(masked.numpy())
            scores = scorer._scores(ds, thresholds)
            pred = model.predict(ds, verbose=0)
            rec = {}
            for k in outputs:
                c = np.asarray(clean_pred[k]).ravel()
                m = np.asarray(pred[k]).ravel()
                lab = np.asarray(labels[k]).ravel() != trainer.mask_value
                flip = (c >= thresholds[k]) != (m >= thresholds[k])
                shift = np.abs(m - c)
                rec[k] = {"f1": scores[k]["f1"], "auc": scores[k]["auc"],
                          "flip_labelled": float(flip[lab].mean()),
                          "flip_unlabelled": float(flip[~lab].mean()) if (~lab).any() else None,
                          "shift_labelled": float(shift[lab].mean()),
                          "shift_unlabelled": float(shift[~lab].mean()) if (~lab).any() else None}
            runs.append(rec)
        result["by_rate"][f"{r:g}"] = runs

    print("\nMask sweep on val (mean over repeats): ΔF1 x100 vs clean, flips % labelled / unlabelled")
    print(f"{'output':<26}" + "".join(f"{'rate ' + format(r, 'g'):>24}" for r in rates))
    for k in outputs:
        cells = []
        for r in rates:
            runs = result["by_rate"][f"{r:g}"]
            df1 = 100 * (np.mean([x[k]["f1"] for x in runs]) - clean[k]["f1"])
            fl = 100 * np.mean([x[k]["flip_labelled"] for x in runs])
            fu = [x[k]["flip_unlabelled"] for x in runs if x[k]["flip_unlabelled"] is not None]
            cells.append(f"{df1:+.1f}  {fl:.1f}/{100 * np.mean(fu):.1f}" if fu else f"{df1:+.1f}  {fl:.1f}/–")
        print(f"{k:<26}" + "".join(f"{c:>24}" for c in cells))
    return result
