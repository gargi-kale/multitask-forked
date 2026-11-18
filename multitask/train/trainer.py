"""The pooled-output loss trainer every method builds on.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import tensorflow as tf

from multitask import Trainer
from multitask.models.hierarchical import pool_probabilities

from multitask.train.columns import PROPAGANDA_POOLS
from multitask.train.optim import build_finetune_optimizer

def binary_kl(t, p, eps=1e-7):
    """Per-row KL(Bernoulli(t) || Bernoulli(p)) = BCE(t, p) - H(t). Zero at
    p = t, and its gradient in the logit is p - t, as for BCE on a label."""
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    tc = tf.clip_by_value(t, eps, 1.0 - eps)
    return (t * (tf.math.log(tc) - tf.math.log(p))
            + (1.0 - t) * (tf.math.log(1.0 - tc) - tf.math.log(1.0 - p)))


class PooledOutputLossTrainer(Trainer):
    """Trainer whose loss sees a pooled output instead of its constituent heads.

    The paper (Sec. 4.1) max-pools the 18 propaganda heads into one binary
    output and describes the network output as 10-dimensional, so the loss
    acts on that pooled output. The stock library loss instead trains each
    constituent head on its own label. Here the constituents are dropped from
    the loss and replaced by one term on max(constituent predictions) against
    max(constituent labels); that is the same tensor as the model's
    ``propaganda_max_pool`` output. The pooled term keeps its task-type's
    ν_k and counts as a single task (|T_k| = 1). Every other task is
    weighted exactly as in ``Trainer._build_loss_function``. Binary tasks only.

    ``pool_strategy`` ('max', 'noisy_or' or 'lse', see
    ``multitask.models.hierarchical.pool_probabilities``) pools the
    constituent predictions; it must match the model's pool config, since the
    model's pooled output is what gets scored. The pooled label is always the
    max of the constituent labels ("any technique present").

    ``technique_loss_weight`` > 0 adds that weight x the mean over the
    constituent heads of each head's class-weighted BCE on its own label, to
    the pooled term (so the balancers still see one propaganda task). Each
    head's term is a weighted mean (divided by the sum of its weights over the
    labelled rows, as PyTorch's weighted cross-entropy does): the rarest
    techniques have class weights in the hundreds, and dividing by the row
    count would let one positive dominate a batch. ``technique_class_weights``
    gives the per-head weights, in ``pooled_tasks`` order.

    ``task_loss_options`` maps a non-pooled task to ``(logit_shift, focal_gamma)``.
    A non-zero shift trains on ``z + logit_shift`` (logit adjustment; predictions
    stay on the raw ``z``), and a non-zero gamma multiplies the term by
    ``(1 - p_correct) ** gamma`` (focal loss).
    """

    def __init__(self, *args, pooled_tasks: Sequence[str],
                 pooled_class_weights: Dict[int, float],
                 task_loss_options: Optional[Dict[str, Tuple[float, float]]] = None,
                 pool_strategy: str = "max", technique_loss_weight: float = 0.0,
                 technique_class_weights: Optional[Sequence[Dict[int, float]]] = None,
                 finetune: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Keyword arguments of build_finetune_optimizer; None = the library's optimizer.
        self.finetune = dict(finetune) if finetune else None
        self.pooled_tasks = list(pooled_tasks)
        self.pooled_class_weights = pooled_class_weights
        self.task_loss_options = task_loss_options or {}
        if pool_strategy not in PROPAGANDA_POOLS:
            raise ValueError(f"pool_strategy must be one of {PROPAGANDA_POOLS}")
        if technique_loss_weight < 0:
            raise ValueError("technique_loss_weight must be >= 0")
        if technique_loss_weight and (technique_class_weights is None
                                      or len(technique_class_weights) != len(self.pooled_tasks)):
            raise ValueError("technique_loss_weight needs one class-weight dict per pooled task")
        self.pool_strategy = pool_strategy
        self.technique_loss_weight = float(technique_loss_weight)
        self.technique_class_weights = (
            [(float(w.get(0, 1.0)), float(w.get(1, 1.0))) for w in technique_class_weights]
            if technique_class_weights is not None else None)

    def _build_optimizer(self, steps_per_epoch):
        if not self.finetune:
            return super()._build_optimizer(steps_per_epoch)
        if self.config.optimizer_type.lower() != "adamw":
            raise ValueError("The fine-tuning optimizer options need optimizer_type 'adamw'")
        return build_finetune_optimizer(
            self.model, self._build_lr_schedule(steps_per_epoch),
            self.config.weight_decay_rate, **self.finetune)

    def _build_loss_function(self, num_tasks, class_weights, task_weights,
                             task_type_mapping=None, mask_value=None):
        if mask_value is None:
            mask_value = self.mask_value
        names = list(self.model.task_names)
        types = self._resolve_task_type_mapping(task_type_mapping)
        pooled = set(self.pooled_tasks)

        # |T_k| without the pooled constituents; ν_k is per task-type, so any
        # constituent carries the pooled term's ν_k.
        type_sizes: Dict[str, int] = {}
        for n in names:
            if n not in pooled:
                type_sizes[types[n]] = type_sizes.get(types[n], 0) + 1
        terms = [
            (n, float(class_weights[i].get(0, 1.0)), float(class_weights[i].get(1, 1.0)),
             float(task_weights[i]) / type_sizes[types[n]],
             *self.task_loss_options.get(n, (0.0, 0.0)))
            for i, n in enumerate(names) if n not in pooled
        ]
        # No pooled tasks (a --tasks subset without propaganda): plain terms only.
        has_pool = bool(self.pooled_tasks)
        # The model output that reports the pool: the config's output pool over
        # these tasks (propaganda_max_pool on the paper's data).
        pool_output = next((name for name, p in (getattr(self.model.config, "output_pools", None) or {}).items()
                            if list(p.get("tasks", [])) == list(self.pooled_tasks)),
                           "propaganda_max_pool")
        if has_pool:
            pool_factor = float(task_weights[names.index(self.pooled_tasks[0])])
            pw0 = float(self.pooled_class_weights[0])
            pw1 = float(self.pooled_class_weights[1])
            print(f"Pooled loss over {len(self.pooled_tasks)} heads ({self.pool_strategy}): "
                  f"class weights {{0: {pw0:.4f}, 1: {pw1:.4f}}}, nu_k={pool_factor:.6f}, |T_k|=1")
            if self.technique_loss_weight:
                print(f"  + {self.technique_loss_weight:g} x mean per-technique weighted BCE "
                      f"over {len(self.pooled_tasks)} heads")
        for n, w0, w1, f, shift, gamma in terms:
            extra = (f", logit shift={shift:+.4f}" if shift else "") + \
                    (f", focal gamma={gamma:g}" if gamma else "")
            print(f"  {n}: class weights {{0: {w0:.4f}, 1: {w1:.4f}}}, "
                  f"nu_k/|T_k|={f:.6f}{extra}")

        def term(y_true, y_pred, w0, w1, factor, shift=0.0, gamma=0.0, soft=None):
            """``soft`` = (target [B], on scalar 0/1): when on, each labelled
            row's BCE on its label is replaced by the binary KL from the soft
            target (self-distillation); rows, class weights and scale stay."""
            y = tf.reshape(tf.cast(y_true, tf.float32), [-1])
            p = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
            mask = tf.cast(tf.not_equal(y, mask_value), tf.float32)
            y = tf.where(mask > 0, y, tf.zeros_like(y))
            if shift or gamma:
                # The heads end in a sigmoid, so recover the raw score z from p.
                # Clipping at 1e-7 matches what Keras BCE does to p anyway.
                p = tf.clip_by_value(p, 1e-7, 1.0 - 1e-7)
                z = tf.math.log(p) - tf.math.log1p(-p) + shift
                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)
                if gamma:
                    p_adj = tf.sigmoid(z)
                    p_correct = tf.where(tf.equal(y, 1.0), p_adj, 1.0 - p_adj)
                    bce = bce * tf.pow(1.0 - p_correct, gamma)
            else:
                bce = tf.reshape(tf.keras.losses.binary_crossentropy(
                    tf.reshape(y, [-1, 1]), tf.reshape(p, [-1, 1])), [-1])
            if soft is not None:
                if shift or gamma:
                    raise ValueError("self-distillation does not support logit shift / focal loss")
                t = tf.reshape(tf.cast(soft[0], tf.float32), [-1])
                on = tf.cast(soft[1], tf.float32)
                bce = on * binary_kl(t, p) + (1.0 - on) * bce
            w = tf.where(tf.equal(y, 0.0), w0, w1)
            return tf.reduce_sum(bce * w * mask * factor), tf.reduce_sum(mask)

        def stack(d):
            return tf.stack(
                [tf.reshape(tf.cast(d[t], tf.float32), [-1]) for t in self.pooled_tasks], axis=1)

        def pool_labels(d):
            return tf.reduce_max(stack(d), axis=1)

        def pool(d):
            return pool_probabilities(stack(d), self.pool_strategy, axis=1)

        tech_w = self.technique_class_weights
        tech_lambda = self.technique_loss_weight

        def technique_loss(y_true_dict, y_pred_dict):
            """Mean over the constituent heads of each head's weighted-mean
            BCE on its own label; a head with no labelled row adds 0 and is
            left out of the mean."""
            total, heads = 0.0, 0.0
            for t, (w0, w1) in zip(self.pooled_tasks, tech_w):
                y = tf.reshape(tf.cast(y_true_dict[t], tf.float32), [-1])
                p = tf.reshape(tf.cast(y_pred_dict[t], tf.float32), [-1])
                mask = tf.cast(tf.not_equal(y, mask_value), tf.float32)
                y = tf.where(mask > 0, y, tf.zeros_like(y))
                bce = tf.reshape(tf.keras.losses.binary_crossentropy(
                    tf.reshape(y, [-1, 1]), tf.reshape(p, [-1, 1])), [-1])
                w = tf.where(tf.equal(y, 0.0), w0, w1) * mask
                wsum = tf.reduce_sum(w)
                total += tf.math.divide_no_nan(tf.reduce_sum(bce * w), wsum)
                heads += tf.cast(wsum > 0, tf.float32)
            return tf.math.divide_no_nan(total, heads)

        def pooled_term(y_true_dict, y_pred_dict, factor, soft=None):
            """(sum, labelled rows) of the pooled propaganda term, with the
            per-technique loss folded in as a mean scaled to the same rows."""
            s, c = term(pool_labels(y_true_dict), pool(y_pred_dict), pw0, pw1, factor,
                        soft=soft)
            if tech_lambda:
                s += tech_lambda * factor * technique_loss(y_true_dict, y_pred_dict) * c
            return s, c

        def pooled_loss(y_true_dict, y_pred_dict, soft=None):
            """``soft`` (self-distillation) maps an output name (a task, or
            the pool's output name) to (target [B], on)."""
            soft = soft or {}
            total, n_valid = 0.0, 0.0
            for n, w0, w1, f, shift, gamma in terms:
                s, c = term(y_true_dict[n], y_pred_dict[n], w0, w1, f, shift, gamma,
                            soft=soft.get(n))
                total += s
                n_valid += c
            if has_pool:
                # mask_value (-1) is below both classes, so the max is masked only
                # when every constituent label is masked.
                s, c = pooled_term(y_true_dict, y_pred_dict, pool_factor,
                                   soft=soft.get(pool_output))
                total += s
                n_valid += c
            return total / (n_valid + 1e-7)

        def task_losses(y_true_dict, y_pred_dict):
            """Per-task class-weighted mean loss, without nu_k or 1/|T_k|, for
            the loss balancers and the gradient probe. Returns (losses [T],
            has_labels [T]); the pooled propaganda output, if any, is task 0."""
            parts = [pooled_term(y_true_dict, y_pred_dict, 1.0)] if has_pool else []
            parts += [term(y_true_dict[n], y_pred_dict[n], w0, w1, 1.0, shift, gamma)
                      for n, w0, w1, _, shift, gamma in terms]
            sums = tf.stack([s for s, _ in parts])
            counts = tf.stack([c for _, c in parts])
            return sums / tf.maximum(counts, 1.0), tf.cast(counts > 0, tf.float32)

        self.model._mtl_task_losses_fn = task_losses
        self.model._mtl_pooled_tasks = list(self.pooled_tasks)
        self.model._mtl_task_loss_names = (
            ([pool_output] if has_pool else []) + [t[0] for t in terms])
        return pooled_loss
