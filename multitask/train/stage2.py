"""Shared train/eval harness and frozen stage 2.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from multitask.config import ModelConfig, TrainingConfig
from multitask import MultiTaskModel, Trainer


# ---------------------------------------------------------------------------
# Shared train/eval harness
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Frozen stage 2: re-train each task's own block on the frozen shared part
# ---------------------------------------------------------------------------
# After stage 1 (any trainer here, with EpochScorer's best epoch restored) the
# encoder, the shared layers and the task-type ("family") blocks are frozen.
# Their outputs are computed once per text and cached, and only each task's
# own block (model.branch_layers[1][i]) and output layer keep training, on the
# real labels with stage 1's per-task loss and class weights. No balancer, no
# consistency term, no masking. With the shared part frozen the tasks share
# no trained weight, so each output can keep the epoch that is best for its
# own val F1; stage 1's block counts as epoch 0. The result is one model.

def table3_scores(trainer, labels, pred, thresholds, outputs):
    """Weighted F1 at the given cut-offs and AUC for each output, from arrays,
    computed as ``Trainer.evaluate`` does (labelled rows only; a pool's label
    is "any constituent positive")."""
    from sklearn.metrics import f1_score, roc_auc_score

    mask_value = trainer.mask_value
    scores = {}
    for k in outputs:
        y, p = np.asarray(labels[k]), np.asarray(pred[k])
        p = p[:, 0] if p.ndim > 1 else p
        valid = y != mask_value
        valid = valid[:, 0] if valid.ndim > 1 else valid
        y = y[valid]
        y = y[:, 0] if y.ndim > 1 else y
        if trainer.model.config.is_pool(k):
            y = (y > 0).astype(int)
        hard = (p > thresholds[k]).astype(int)[valid]
        scores[k] = {
            "f1": float(f1_score(y, hard, average="weighted", zero_division=0)),
            "auc": float(roc_auc_score(y, p[valid])),
        }
    return scores


class Stage2:
    """The frozen stage 2 for one trained model. ``outputs`` are the scored
    outputs (a task, or a pool standing for its constituent heads)."""

    def __init__(self, trainer, outputs: Sequence[str], batch_size: int = 256,
                 weight_decay: float = 0.01):
        model = trainer.model
        if len(model.branch_layers) != 2:
            raise ValueError("stage 2 needs a family level and a task level of blocks")
        if getattr(model, "_mtl_task_losses_fn", None) is None:
            raise ValueError("stage 2 needs the pooled-loss trainers (--pooled-propaganda-loss)")
        self.trainer, self.model = trainer, model
        self.names = list(model.task_names)
        self.outputs = list(outputs)
        self.batch_size, self.weight_decay = batch_size, weight_decay
        # family block that feeds each task block
        self.parent = [f for f, n in enumerate(model.config.task_structure[1]) for _ in range(n)]
        self.task_vars: Dict[str, List[tf.Variable]] = {}
        for k in self.outputs:
            tasks = model.config.get_pool_config(k)["tasks"] if model.config.is_pool(k) else [k]
            variables = []
            for t in tasks:
                for layer in model.branch_layers[1][self.names.index(t)]:
                    variables += layer.variables
                variables += model.output_layers[t].variables
            self.task_vars[k] = variables
        self.frozen_layers = ([model.encoder] if model.uses_encoder else []) + \
            list(model.shared_layers) + [l for branch in model.branch_layers[0] for l in branch]
        self.frozen_vars = [v for layer in self.frozen_layers for v in layer.variables]
        # reduce_retracing: batches padded to their own longest text come in many
        # widths; without it the encoder would be traced once per width.
        self._features_fn = tf.function(self._features, reduce_retracing=True)
        self._full = None  # (val_ds, val labels, test_ds, test labels): see use_full_model
        self._heads_fn = tf.function(lambda feats: self.heads(feats, training=False))

    def freeze(self):
        for layer in self.frozen_layers:
            layer.trainable = False

    def frozen_weights(self) -> List[np.ndarray]:
        return [v.numpy() for v in self.frozen_vars]

    def _features(self, x):
        h = self.model._encode_input(x, training=False)
        for layer in self.model.shared_layers:
            h = layer(h, training=False)
        feats = []
        for branch in self.model.branch_layers[0]:
            z = h
            for layer in branch:
                z = layer(z, training=False)
            feats.append(z)
        return feats

    def heads(self, feats, training=False):
        """Every task block and output layer on the cached family outputs,
        plus the pooled outputs: what the full model returns."""
        out = {}
        for i, name in enumerate(self.names):
            z = feats[self.parent[i]]
            for layer in self.model.branch_layers[1][i]:
                z = layer(z, training=training)
            out[name] = self.model.output_layers[name](z, training=training)
        out.update(self.model._compute_pooled_outputs(out))
        return out

    def cache(self, ds):
        """One pass over ``ds``: (family outputs, one array per family; labels,
        pooled labels included). Rows come in the order ``ds`` yields them."""
        feats = [[] for _ in self.model.branch_layers[0]]
        labels = {k: [] for k in self.names}
        for x, y in ds:
            for acc, f in zip(feats, self._features_fn(x)):
                acc.append(f.numpy())
            for k in self.names:
                labels[k].append(y[k].numpy())
        labels = {k: np.concatenate(v, axis=0) for k, v in labels.items()}
        labels.update(self.trainer._derive_pooled_labels(labels))
        return [np.concatenate(f, axis=0) for f in feats], labels

    def predict(self, feats, batch: int = 8192):
        parts: Dict[str, list] = {}
        for s in range(0, len(feats[0]), batch):
            out = self._heads_fn([tf.constant(f[s:s + batch]) for f in feats])
            for k, v in out.items():
                parts.setdefault(k, []).append(v.numpy())
        return {k: np.concatenate(v, axis=0) for k, v in parts.items()}

    def _labels(self, ds):
        labels = {k: [] for k in self.names}
        for _, y in ds:
            for k in self.names:
                labels[k].append(y[k].numpy())
        labels = {k: np.concatenate(v, axis=0) for k, v in labels.items()}
        labels.update(self.trainer._derive_pooled_labels(labels))
        return labels

    def use_full_model(self, val_ds, test_ds):
        """Score val and test through the whole network (``model.predict``,
        as EpochScorer does) instead of through cached features. On a GPU the
        two paths differ by ~1e-3 in a score, which is enough to move a J
        cut-off and the F1 at it; this way epoch 0 is exactly stage 1 and the
        assembled model scores exactly as each output's chosen epoch did.
        The cache is then only used to train."""
        self._full = (val_ds, self._labels(val_ds), test_ds, self._labels(test_ds))

    def score(self, val=None, test=None):
        """(val scores, test scores, cut-offs): the J cut-offs are tuned on val
        and used on both, as EpochScorer does. ``val`` / ``test`` are caches,
        ignored after ``use_full_model``."""
        if self._full is not None:
            val_ds, val_labels, test_ds, test_labels = self._full
            val_pred = self.model.predict(val_ds, verbose=0)
            test_pred = self.model.predict(test_ds, verbose=0)
        else:
            (val_feats, val_labels), (test_feats, test_labels) = val, test
            val_pred, test_pred = self.predict(val_feats), self.predict(test_feats)
        thresholds, _ = self.trainer._find_optimal_thresholds(val_labels, val_pred)
        return (table3_scores(self.trainer, val_labels, val_pred, thresholds, self.outputs),
                table3_scores(self.trainer, test_labels, test_pred, thresholds, self.outputs),
                thresholds)

    def get_blocks(self):
        return {k: [v.numpy() for v in vs] for k, vs in self.task_vars.items()}

    def set_blocks(self, blocks):
        for k, weights in blocks.items():
            for v, w in zip(self.task_vars[k], weights):
                v.assign(w)

    def fit(self, train, val=None, test=None, *, lr: float, epochs: int, seed: Optional[int] = None,
            only: Optional[Sequence[str]] = None):
        """Trains the task blocks (those of ``only``, default every output)
        from their current weights, keeps each output's own val-F1-best epoch
        (epoch 0 = the weights it started with; a later epoch must be strictly
        better) and leaves the model holding that assembly."""
        import time

        feats, labels = train
        n = len(feats[0])
        train_outputs = list(only) if only is not None else self.outputs
        variables = [v for k in train_outputs for v in self.task_vars[k]]
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=self.weight_decay)
        optimizer.build(variables)
        loss_fn = self.model._mtl_task_losses_fn
        F = [tf.constant(f) for f in feats]
        Y = {k: tf.constant(labels[k]) for k in self.names}

        @tf.function
        def step(idx):
            with tf.GradientTape() as tape:
                pred = self.heads([tf.gather(f, idx) for f in F], training=True)
                losses, has = loss_fn({k: tf.gather(Y[k], idx) for k in self.names},
                                      {k: pred[k] for k in self.names})
                loss = tf.reduce_sum(losses * has)
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            return loss

        def record(epoch, v, t, seconds):
            rec = {"epoch": epoch, "val": v, "test": t, "seconds": seconds}
            for m in ("f1", "auc"):
                rec[f"val_{m}"] = float(np.mean([s[m] for s in v.values()]))
                rec[f"test_{m}"] = float(np.mean([s[m] for s in t.values()]))
            return rec

        v, t, _ = self.score(val, test)
        records = [record(0, v, t, 0.0)]
        best_f1 = {k: v[k]["f1"] for k in self.outputs}
        best_epoch = {k: 0 for k in self.outputs}
        best_blocks = self.get_blocks()
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            start = time.time()
            order = rng.permutation(n).astype(np.int32)
            for s in range(0, n, self.batch_size):
                step(tf.constant(order[s:s + self.batch_size]))
            seconds = time.time() - start
            v, t, _ = self.score(val, test)
            records.append(record(epoch, v, t, seconds))
            for k in train_outputs:
                if v[k]["f1"] > best_f1[k]:
                    best_f1[k], best_epoch[k] = v[k]["f1"], epoch
                    best_blocks[k] = [w.numpy() for w in self.task_vars[k]]
            print(f"Stage 2 lr {lr:g} epoch {epoch}: val F1 {100 * records[-1]['val_f1']:.2f}  "
                  f"val AUC {100 * records[-1]['val_auc']:.2f}  ({seconds:.1f}s train)", flush=True)
        self.set_blocks(best_blocks)
        return {"lr": lr, "epochs": records, "chosen_epoch": best_epoch}

    def full_model_scores(self, val_ds, test_ds):
        """The model as it stands, scored by the library through the whole
        network: J cut-offs re-tuned on val, test scored once with them."""
        val_labels = self._labels(val_ds)
        pred = self.model.predict(val_ds, verbose=0)
        thresholds, _ = self.trainer._find_optimal_thresholds(val_labels, pred)
        out = {"thresholds": {k: thresholds[k] for k in self.outputs}}
        raw = {}
        for split, ds in (("val", val_ds), ("test", test_ds)):
            res = self.trainer.evaluate(ds, thresholds=thresholds)
            merged = {**res.get("task_metrics", {}), **res.get("pool_metrics", {})}
            out[split] = {k: {"f1": float(merged[k]["f1"]), "auc": float(merged[k]["auc"])}
                          for k in self.outputs}
            for m in ("f1", "auc"):
                out[f"{split}_{m}"] = float(np.mean([s[m] for s in out[split].values()]))
            raw[split] = res
        return out, raw


def run_stage2(trainer, train_ds, val_ds, test_ds, *, lrs: Sequence[str], epochs: int,
               outputs: Sequence[str], batch_size: int = 256, weight_decay: float = 0.01,
               seed: Optional[int] = None, predictions_prefix: Optional[str] = None,
               weights_path: Optional[str] = None) -> Dict[str, Any]:
    """Stage 2 at each learning rate in ``lrs`` (strings, kept as the result
    keys), all from the same stage-1 model and the same cache: the task blocks
    are put back to stage 1 after each rate. Writes one predictions file per
    rate (``<predictions_prefix>_stage2_lr<lr>_predictions.npz``) and, if
    ``weights_path`` is given, the stage-1 weights (``model.variables`` order
    before freezing, with their names)."""
    import time

    model = trainer.model
    if weights_path:
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        np.savez_compressed(
            weights_path, names=np.array([v.name for v in model.variables]),
            **{f"w{i}": v.numpy() for i, v in enumerate(model.variables)})
        print(f"Wrote {weights_path} ({len(model.variables)} variables)")
    s2 = Stage2(trainer, outputs, batch_size=batch_size, weight_decay=weight_decay)
    s2.freeze()
    frozen_before = s2.frozen_weights()
    # every variable stage 2 may not train (frozen part, and the blocks of
    # outputs not in ``outputs``, e.g. a hidden one)
    trained_ids = {id(v) for vs in s2.task_vars.values() for v in vs}
    untrained = [v for v in model.variables if id(v) not in trained_ids]
    untrained_before = [v.numpy() for v in untrained]
    start = time.time()
    train = s2.cache(train_ds)
    cache_seconds = time.time() - start
    full_val = model.predict(val_ds, verbose=0)
    cached_val = s2.predict(s2.cache(val_ds)[0])
    s2.use_full_model(val_ds, test_ds)
    cache_gap = max(float(np.max(np.abs(np.asarray(full_val[k]) - cached_val[k])))
                    for k in cached_val)
    print(f"Stage 2 cache: {len(train[0][0]):,} train rows in {cache_seconds:.0f}s; largest gap "
          f"between cached and full-model val scores {cache_gap:.2e}", flush=True)
    stage1_blocks = s2.get_blocks()
    stage1, _ = s2.full_model_scores(val_ds, test_ds)
    out: Dict[str, Any] = {
        "outputs": list(outputs), "stage2_epochs": epochs, "batch_size": batch_size,
        "weight_decay": weight_decay, "cache_seconds": cache_seconds,
        "cache_rows": {"train": len(train[0][0])},
        "cache_max_abs_gap": cache_gap, "stage1": stage1, "stage1_weights": weights_path,
        "runs": {},
    }
    for lr in lrs:
        run = s2.fit(train, lr=float(lr), epochs=epochs, seed=seed)
        run["assembled"], raw = s2.full_model_scores(val_ds, test_ds)
        # each output of the assembled model against its chosen epoch's record
        run["assembly_max_f1_gap"] = max(
            abs(run["assembled"][split][k]["f1"]
                - run["epochs"][run["chosen_epoch"][k]][split][k]["f1"])
            for split in ("val", "test") for k in outputs)
        if predictions_prefix:
            arrays = {}
            for split, res in raw.items():
                for name, prob in (res.get("probabilities_dict") or {}).items():
                    arrays[f"{split}/prob/{name}"] = np.asarray(prob, dtype=np.float32)
                for name, label in (res.get("labels_dict") or {}).items():
                    arrays[f"{split}/label/{name}"] = np.asarray(label, dtype=np.float32)
            path = f"{predictions_prefix}_stage2_lr{lr}_predictions.npz"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez_compressed(path, **arrays)
            run["predictions"] = os.path.basename(path)
            print(f"Wrote {path} ({len(arrays)} arrays)")
        a = run["assembled"]
        print(f"Stage 2 lr {lr}: assembled val F1 {100 * a['val_f1']:.2f} test F1 "
              f"{100 * a['test_f1']:.2f}  (stage 1: {100 * stage1['val_f1']:.2f} / "
              f"{100 * stage1['test_f1']:.2f}); chosen epochs {run['chosen_epoch']}", flush=True)
        out["runs"][lr] = run
        s2.set_blocks(stage1_blocks)
    out["frozen_unchanged"] = all(
        np.array_equal(a, b) for a, b in zip(frozen_before, s2.frozen_weights()))
    out["untrained_unchanged"] = all(
        np.array_equal(a, v.numpy()) for a, v in zip(untrained_before, untrained))
    print(f"Stage 2: frozen weights unchanged: {out['frozen_unchanged']}; every variable "
          f"outside the trained blocks unchanged: {out['untrained_unchanged']}", flush=True)
    return out


def run_experiment(
    config: ModelConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 4e-5,
    optimize_thresholds: bool = True,
    verbose: int = 1,
    show_structure: bool = True,
    trainer: Optional[Trainer] = None,
    task_type_mapping: Optional[Dict[str, str]] = None,
    training_config: Optional[TrainingConfig] = None,
    class_weights: Optional[List[Dict[int, float]]] = None,
    task_weights: Optional[List[float]] = None,
    test_ds: Optional[tf.data.Dataset] = None,
    trainer_factory: Optional[Any] = None,
    model_cls: type = MultiTaskModel,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> Dict[str, Any]:
    # Resolve the TrainingConfig before building the model so we can pass
    # dropout_rate into MultiTaskModel — the model is constructed before the
    # Trainer, so the trainer can't push it in later.
    if trainer is None:
        resolved_training_config = training_config or TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    else:
        resolved_training_config = trainer.config

    model = model_cls(config, dropout_rate=resolved_training_config.dropout_rate)
    # Under Keras 3 a tf_keras encoder (HuggingFace TF) is not tracked, so it
    # would silently stay frozen. Fail instead; TF_USE_LEGACY_KERAS=1 fixes it.
    if model.uses_encoder and model.encoder.trainable:
        tracked = {id(v) for v in model.trainable_variables}
        missing = [v for v in model.encoder.trainable_variables if id(v) not in tracked]
        if missing:
            raise RuntimeError(
                f"{len(missing)} trainable encoder variables are not in "
                "model.trainable_variables, so the encoder would never train. "
                "Set TF_USE_LEGACY_KERAS=1 before importing TensorFlow."
            )
    if show_structure:
        print("\nModel structure:")
        print(model.summary_tree())
        print("Task paths:")
        for info in model.get_task_info():
            print(
                f"  {info['output_idx']:>2}  {info['name']:<35} "
                f"path={info['parent_branch_path']}"
            )

    if trainer is None:
        trainer = (trainer_factory or Trainer)(model, resolved_training_config)

    _, thresholds = trainer.fit(
        train_ds,
        val_ds,
        num_tasks=len(config.task_names),
        class_weights=class_weights,
        task_weights=task_weights,
        task_type_mapping=task_type_mapping,
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        optimize_thresholds=optimize_thresholds,
        verbose=verbose,
        callbacks=callbacks,
    )
    print("Thresholds:", thresholds)

    results = trainer.evaluate(val_ds, thresholds=thresholds)
    results["history"] = {k: [float(v) for v in vs] for k, vs in trainer.history.history.items()}
    print("Overall metrics:", results.get("overall_metrics", {}))
    print("Task-specific metrics:", results.get("task_metrics", {}))
    if test_ds is not None:
        # Thresholds stay the val-tuned ones; the test split is never fit on.
        print("\n--- Held-out test split ---")
        test_results = trainer.evaluate(test_ds, thresholds=thresholds)
        print("Test overall metrics:", test_results.get("overall_metrics", {}))
        print("Test task-specific metrics:", test_results.get("task_metrics", {}))
        print("Test pool metrics:", test_results.get("pool_metrics", {}))
        results["test"] = test_results
    return results
