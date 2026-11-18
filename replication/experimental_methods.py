"""Methods that were tried and not adopted: semi-supervised rate
targets, per-task self-distillation, the gradient probe.

Kept for reproducing the research write-ups; not part of the toolkit.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from multitask import MultiTaskModel
from multitask.models.hierarchical import pool_probabilities

from multitask.train.balancers import shared_variables
from multitask.train.consistency import LabelAwareMaskedModel, LabelAwareMaskedTrainer
from multitask.train.data import ROW_ID_KEY
from multitask.train.scoring import EpochScorer
from multitask.train.trainer import PooledOutputLossTrainer

# ---------------------------------------------------------------------------
# Semi-supervised terms on foreign rows: a target rate and a confidence penalty
# ---------------------------------------------------------------------------

SSL_POOL = "propaganda_max_pool"
SSL_SOURCES = ("IAC", "IBM", "Propaganda")
# head@source=rate. Rough rates from reading 40 random train rows per source
# (Claude, 2026-11-03; not a proper annotation). For the IAC heads on
# Propaganda rows the rate is for rows the technique rule did not label,
# since rule-labelled rows are not foreign.
SSL_DEFAULT_RATES = (
    "propaganda_max_pool@IAC=0.30,propaganda_max_pool@IBM=0.10,"
    "emotion_fact_bi@Propaganda=0.12,nasty_nice_bi@Propaganda=0.05,"
    "attacking_respectful_bi@Propaganda=0.07,"
    "emotion_fact_bi@IBM=0.15,nasty_nice_bi@IBM=0.03,attacking_respectful_bi@IBM=0.05"
)
SSL_TRACKERS = ("ssl_rate_loss", "ssl_entropy_loss", "ssl_entropy_weight")


def parse_ssl_rates(spec: str) -> List[Tuple[str, str, float]]:
    """'head@source=rate,...' -> [(head, source, rate)]."""
    pairs = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        try:
            cell, rate = item.split("=")
            head, source = cell.split("@")
            rate = float(rate)
        except ValueError:
            raise ValueError(f"Bad rate {item!r}: expected head@source=rate") from None
        if not 0.0 < rate < 1.0:
            raise ValueError(f"Rate for {cell} must be strictly between 0 and 1, got {rate:g}")
        pairs.append((head, source, rate))
    if len({(h, s) for h, s, _ in pairs}) != len(pairs):
        raise ValueError("Each head@source pair may appear only once")
    return pairs


def ssl_pair_label(head: str, source: str) -> str:
    short = "propaganda" if head == SSL_POOL else re.sub(r"_bi$", "", head)
    return f"{short}_{source}"


def semi_supervised_terms(y, y_pred, pairs, pooled_tasks, home_tasks, rest_source,
                          mask_value=-1.0, eps=1e-6):
    """The two extra terms over foreign cells. A cell of head h is foreign for
    the pair (h, source, rate) when its row comes from ``source`` and h has no
    label there. A row's source is read from the labels: a label for any task
    in ``home_tasks[s]`` puts it in s, none of them puts it in ``rest_source``.
    The pooled propaganda output (``SSL_POOL``) is the max of the
    ``pooled_tasks`` heads, with a label wherever any technique has one.

    - Rate: the binary KL divergence from the target rate to the mean score
      over the pair's foreign cells in the batch. It constrains only the
      batch average, never a single row.
    - Entropy: the mean binary entropy of the scores on those cells, which
      is lowest when every score is near 0 or 1.

    Both are averaged over the pairs that have foreign cells in the batch, so
    they are 0 when there are none. Returns (rate, entropy, mean score per
    pair [P], foreign cells per pair [P]); a pair with no cells has mean 0."""
    def flat(d, k):
        return tf.reshape(tf.cast(d[k], tf.float32), [-1])

    def head_of(d, head):
        if head == SSL_POOL:
            return tf.reduce_max(tf.stack([flat(d, t) for t in pooled_tasks], axis=1), axis=1)
        return flat(d, head)

    home = {s: tf.reduce_any(tf.stack([tf.not_equal(flat(y, t), mask_value) for t in ts],
                                      axis=1), axis=1)
            for s, ts in home_tasks.items()}
    sources = dict(home)
    sources[rest_source] = tf.logical_not(tf.reduce_any(tf.stack(list(home.values()), axis=1),
                                                        axis=1))
    rates, ents, means, counts, active = [], [], [], [], []
    for head, source, rate in pairs:
        m = tf.cast(sources[source] & tf.equal(head_of(y, head), mask_value), tf.float32)
        p = tf.clip_by_value(head_of(y_pred, head), eps, 1.0 - eps)
        n = tf.reduce_sum(m)
        mean = tf.math.divide_no_nan(tf.reduce_sum(p * m), n)
        q = tf.clip_by_value(mean, eps, 1.0 - eps)
        kl = rate * tf.math.log(rate / q) + (1.0 - rate) * tf.math.log((1.0 - rate) / (1.0 - q))
        entropy = -(p * tf.math.log(p) + (1.0 - p) * tf.math.log(1.0 - p))
        on = tf.cast(n > 0, tf.float32)
        rates.append(kl * on)
        ents.append(tf.math.divide_no_nan(tf.reduce_sum(entropy * m), n))
        means.append(mean)
        counts.append(n)
        active.append(on)
    n_active = tf.maximum(tf.add_n(active), 1.0)
    return (tf.add_n(rates) / n_active, tf.add_n(ents) / n_active,
            tf.stack(means), tf.stack(counts))


class SemiSupervisedModel(MultiTaskModel):
    """MultiTaskModel trained on the supervised loss plus, on foreign cells
    (``semi_supervised_terms``), ``rate_weight`` x the rate term and
    ``entropy_weight`` x the entropy term. The entropy term is off until
    ``entropy_start_step`` optimizer steps have run. The reported and
    validation loss is the supervised loss alone, so early stopping compares
    with the plain pooled-loss runs. Logs both terms and, per pair, the mean
    score on its foreign cells (``ssl_mean_<head>_<source>``), on train and val.
    Call ``attach_semi_supervised`` before training."""

    def attach_semi_supervised(self, pairs, pooled_tasks, home_tasks, rest_source,
                               rate_weight, entropy_weight=0.0, entropy_start_step=0):
        names = set(self.task_names)
        sources = set(home_tasks) | {rest_source}
        missing = [t for ts in home_tasks.values() for t in ts if t not in names]
        if missing:
            raise ValueError(f"Source-defining tasks not in the model: {missing}")
        for head, source, _ in pairs:
            if head == SSL_POOL:
                if not pooled_tasks:
                    raise ValueError(f"{SSL_POOL} needs the pooled propaganda heads")
            elif head not in names:
                raise ValueError(f"Unknown head {head!r}")
            if source not in sources:
                raise ValueError(f"Unknown source {source!r}; choose from {sorted(sources)}")
        if rate_weight < 0 or entropy_weight < 0 or not (rate_weight or entropy_weight):
            raise ValueError("Weights must be >= 0 and at least one > 0")
        self.ssl_pairs = [(h, s, float(r)) for h, s, r in pairs]
        self.ssl_pooled_tasks = list(pooled_tasks)
        self.ssl_home_tasks = {k: list(v) for k, v in home_tasks.items()}
        self.ssl_rest_source = rest_source
        self.ssl_rate_weight, self.ssl_entropy_weight = float(rate_weight), float(entropy_weight)
        self.ssl_entropy_start_step = int(entropy_start_step)
        self.ssl_labels = [ssl_pair_label(h, s) for h, s, _ in self.ssl_pairs]
        # Attributes, so Keras resets them every epoch.
        for k in SSL_TRACKERS:
            setattr(self, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        self.ssl_mean_trackers = [tf.keras.metrics.Mean(name=f"ssl_mean_{lab}")
                                  for lab in self.ssl_labels]
        print(f"Semi-supervised: rate weight {self.ssl_rate_weight:g}, entropy weight "
              f"{self.ssl_entropy_weight:g} from step {self.ssl_entropy_start_step}; sources "
              f"{ {**self.ssl_home_tasks, rest_source: 'neither'} }")
        for (h, s, r), lab in zip(self.ssl_pairs, self.ssl_labels):
            print(f"  {lab}: target rate {r:g}")

    def ssl_terms(self, y, y_pred):
        return semi_supervised_terms(y, y_pred, self.ssl_pairs, self.ssl_pooled_tasks,
                                     self.ssl_home_tasks, self.ssl_rest_source,
                                     self.config.mask_value)

    def ssl_entropy_weight_now(self):
        on = tf.cast(self.optimizer.iterations, tf.float32) >= float(self.ssl_entropy_start_step)
        return tf.where(on, self.ssl_entropy_weight, 0.0)

    def _log_ssl(self, rate, entropy, means, counts, entropy_weight=None):
        out = {}
        pairs = [(self.ssl_rate_loss_tracker, rate), (self.ssl_entropy_loss_tracker, entropy)]
        if entropy_weight is not None:
            pairs.append((self.ssl_entropy_weight_tracker, entropy_weight))
        for tracker, v in pairs:
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        for i, tracker in enumerate(self.ssl_mean_trackers):
            tracker.update_state(means[i], sample_weight=counts[i])
            out[tracker.name] = tracker.result()
        return out

    def train_step(self, data):
        x, y = data[0], data[1]
        names = self.task_names
        entropy_weight = self.ssl_entropy_weight_now()
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            y_pred = {k: pred[k] for k in names}
            loss = self._mtl_loss_fn(y, y_pred)
            rate, entropy, means, counts = self.ssl_terms(y, y_pred)
            total = loss + self.ssl_rate_weight * rate + entropy_weight * entropy
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        return {'loss': loss, **self._log_ssl(rate, entropy, means, counts, entropy_weight)}

    def test_step(self, data):
        x, y = data[0], data[1]
        pred = self(x, training=False)
        self._log_ssl(*self.ssl_terms(y, {k: pred[k] for k in self.task_names}))
        return MultiTaskModel.test_step(self, data)


class SemiSupervisedTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a SemiSupervisedModel: attaches the pairs
    and weights, with the entropy start given in epochs of the train dataset."""

    def __init__(self, *args, ssl_pairs, ssl_home_tasks, ssl_rest_source: str = "IAC",
                 rate_weight: float, entropy_weight: float = 0.0,
                 entropy_start_epochs: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, SemiSupervisedModel):
            raise TypeError("SemiSupervisedTrainer needs a SemiSupervisedModel")
        self.ssl_pairs, self.ssl_home_tasks = ssl_pairs, ssl_home_tasks
        self.ssl_rest_source = ssl_rest_source
        self.rate_weight, self.entropy_weight = rate_weight, entropy_weight
        self.entropy_start_epochs = entropy_start_epochs

    def fit(self, train_dataset, *args, **kwargs):
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("The entropy start needs a train dataset of known length")
        self.model.attach_semi_supervised(
            self.ssl_pairs, self.pooled_tasks, self.ssl_home_tasks, self.ssl_rest_source,
            self.rate_weight, self.entropy_weight, round(self.entropy_start_epochs * steps))
        return super().fit(train_dataset, *args, **kwargs)


# ---------------------------------------------------------------------------
# Per-task stopping with self-distillation (plain pooled loss)
# ---------------------------------------------------------------------------
# Each Table-3 output trains on its labels until its val F1 has not improved
# for ``patience`` epochs. Then the weights of its best epoch score every train
# row once (eval mode), and from there on that output's loss, on the same
# labelled rows with the same class weights and scale, is the binary KL from
# those stored scores instead of BCE on the label. The other outputs go on as
# before. Training ends once every output has switched (or at --epochs).
# After Caruana (1997, 7.1) per-task stopping and Lu et al. (NAACL 2022),
# distilling converged tasks from their own best checkpoint.


class _DistillState:
    """Plain holder (not tracked by Keras), so the stored scores are not part
    of get_weights / set_weights and survive weight restores."""


class SelfDistillMixin:
    """The score table and switches shared by the self-distilled models:
    ``_sd.scores`` [rows, outputs] (stored best-epoch scores), ``_sd.on``
    [outputs] (1 once an output has switched). Call ``attach_self_distill``
    before training; ``SelfDistillCallback`` fills both."""

    def attach_self_distill(self, outputs: Sequence[str], n_rows: int):
        st = _DistillState()
        st.outputs = list(outputs)
        st.scores = tf.Variable(tf.fill([n_rows, len(st.outputs)], 0.5),
                                trainable=False, name="self_distill_scores")
        st.on = tf.Variable(tf.zeros([len(st.outputs)]), trainable=False,
                            name="self_distill_on")
        self._sd = st
        print(f"Self-distillation: {len(st.outputs)} outputs, score table for {n_rows:,} rows")

    def soft_targets(self, y):
        """{output: (stored score per batch row, on)} for the pooled loss."""
        st = self._sd
        rows = tf.reshape(tf.cast(y[ROW_ID_KEY], tf.int32), [-1])
        target = tf.gather(st.scores, rows)
        return {k: (target[:, j], st.on[j]) for j, k in enumerate(st.outputs)}

    def output_on(self, name):
        """Switch flag of one output (0 for outputs not in the table)."""
        st = self._sd
        return st.on[st.outputs.index(name)] if name in st.outputs else tf.constant(0.0)


class SelfDistillModel(SelfDistillMixin, MultiTaskModel):
    """MultiTaskModel on the plain pooled loss whose outputs can be switched,
    one by one, from their labels to stored scores (``SelfDistillCallback``).
    The logged train loss is the loss being optimised; val loss stays the
    plain label loss. Call ``attach_self_distill`` before training."""

    def train_step(self, data):
        x, y = data[0], data[1]
        st = self._sd
        names = self.task_names
        soft = self.soft_targets(y)
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            y_pred = {k: pred[k] for k in names}
            loss = self._mtl_loss_fn(y, y_pred, soft)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        return {'loss': loss, 'distilled_outputs': tf.reduce_sum(st.on)}


class SelfDistillFamoModel(SelfDistillMixin, LabelAwareMaskedModel):
    """famo_consf1_m5 (FAMO + filtered masked consistency) with per-task
    self-distillation. FAMO works on log losses, and a just-switched output's
    KL starts near 0, so switched outputs leave FAMO (their has_labels is set
    to 0, so they get no FAMO weight and no FAMO update) and their KL from the
    stored scores is added outside FAMO with weight ``distill_weight``, like
    the masked-word and consistency terms: the plain pooled loss restricted
    to the switched outputs (mean KL per labelled cell, same class weights).
    The consistency term is unchanged and still covers all 10 outputs."""

    distill_weight = 1.0

    def distill_loss(self, y, y_pred):
        mask = self.config.mask_value
        pooled = set(getattr(self, "_mtl_pooled_tasks", []))
        y_d = {}
        for k in self.task_names:
            on = self.output_on("propaganda_max_pool" if k in pooled else k)
            y_d[k] = tf.where(on > 0, tf.cast(y[k], tf.float32),
                              tf.fill(tf.shape(y[k]), tf.cast(mask, tf.float32)))
        on_all = {k: (t, tf.constant(1.0)) for k, (t, _) in self.soft_targets(y).items()}
        return self._mtl_loss_fn(y_d, y_pred, on_all)

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, names = self._balancer, self.task_names
        variables = self.trainable_variables
        masking = self.mask_batch(x)
        cons_weight = self.consistency_weight * self.consistency_ramp()
        in_famo = 1.0 - tf.stack([self.output_on(n) for n in self._mtl_task_loss_names])
        with tf.GradientTape() as tape:
            clean = self(x, training=True)
            y_pred = {k: clean[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            has_labels = has_labels * in_famo
            mlm_loss, mlm_acc, label_loss, cons_loss, kept = self.label_aware_losses(
                x, y, clean, True, masking)
            distill = self.distill_loss(y, y_pred)
            objective = (bal.objective(losses, has_labels) + self.mlm_lambda * mlm_loss
                         + self.masked_label_weight * label_loss + cons_weight * cons_loss
                         + self.distill_weight * distill)
        grads = tape.gradient(objective, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        new_pred = self(x, training=True)
        new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
        bal.update(losses, new_losses, has_labels)

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        out = {'loss': loss, 'distill_loss': distill,
               'distilled_outputs': tf.reduce_sum(self._sd.on)}
        for tracker, v in ((self.mlm_loss_tracker, mlm_loss), (self.mlm_acc_tracker, mlm_acc),
                           (self.masked_label_loss_tracker, label_loss),
                           (self.consistency_loss_tracker, cons_loss),
                           (self.consistency_weight_tracker, cons_weight),
                           (self.consistency_kept_tracker, kept)):
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        return out


class SelfDistillFamoTrainer(LabelAwareMaskedTrainer):
    """LabelAwareMaskedTrainer for a SelfDistillFamoModel: also attaches the
    score table (batches x batch size >= rows)."""

    def __init__(self, *args, distill_outputs: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, SelfDistillFamoModel):
            raise TypeError("SelfDistillFamoTrainer needs a SelfDistillFamoModel")
        if self.technique_loss_weight:
            raise ValueError("self-distillation does not support --technique-loss-weight")
        self.distill_outputs = list(distill_outputs)

    def fit(self, train_dataset, *args, **kwargs):
        _, y = next(iter(train_dataset))
        if ROW_ID_KEY not in y:
            raise ValueError("self-distillation needs load_splits(train_row_ids=True)")
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("self-distillation needs a train dataset of known length")
        self.model.attach_self_distill(self.distill_outputs,
                                       steps * int(tf.shape(y[ROW_ID_KEY])[0]))
        return super().fit(train_dataset, *args, **kwargs)


class SelfDistillTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a SelfDistillModel: sizes the score table
    from the train dataset (batches x batch size >= rows) and attaches it."""

    def __init__(self, *args, distill_outputs: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, SelfDistillModel):
            raise TypeError("SelfDistillTrainer needs a SelfDistillModel")
        if self.technique_loss_weight:
            raise ValueError("self-distillation does not support --technique-loss-weight")
        self.distill_outputs = list(distill_outputs)

    def fit(self, train_dataset, *args, **kwargs):
        x, y = next(iter(train_dataset))
        if ROW_ID_KEY not in y:
            raise ValueError("self-distillation needs load_splits(train_row_ids=True)")
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("self-distillation needs a train dataset of known length")
        self.model(x, training=False)  # create all variables
        self.model.attach_self_distill(self.distill_outputs,
                                       steps * int(tf.shape(y[ROW_ID_KEY])[0]))
        return super().fit(train_dataset, *args, **kwargs)


SELF_DISTILL_SIGNALS = ("f1", "loss")  # per-output val signal for switching


class SelfDistillCallback(tf.keras.callbacks.Callback):
    """Switches each output to self-distillation once its val ``signal`` (F1,
    higher is better, or its own val loss, lower is better; from
    ``scorer.records``, so it must come after the scorer in the callback list)
    has gone ``patience`` epochs without improving, using the scores of its
    first best epoch. Keeps the trainable weights of only the epochs that can
    still be needed. Stops training once every output has switched.
    ``switches`` lists {output, best_epoch, switched_after_epoch}."""

    def __init__(self, scorer: "EpochScorer", train_ds, patience: int = 2,
                 pooled_tasks: Sequence[str] = (), pool_strategy: str = "max",
                 signal: str = "f1"):
        super().__init__()
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if signal not in SELF_DISTILL_SIGNALS:
            raise ValueError(f"signal must be one of {SELF_DISTILL_SIGNALS}")
        self.signal = signal
        self.scorer, self.train_ds, self.patience = scorer, train_ds, patience
        self.pooled_tasks, self.pool_strategy = list(pooled_tasks), pool_strategy

    def on_train_begin(self, logs=None):
        self.snapshots: Dict[int, List[np.ndarray]] = {}
        self.switches: List[Dict[str, Any]] = []
        self._done = set()
        self._predict = tf.function(lambda x: self.model(x, training=False))

    def _weights(self):
        return [v.numpy() for v in self.model.trainable_variables]

    def _assign(self, weights):
        for v, w in zip(self.model.trainable_variables, weights):
            v.assign(w)

    def _score_train(self, outputs: Sequence[str]) -> np.ndarray:
        """Scores [rows, len(outputs)] of the current weights on every train row."""
        st = self.model._sd
        out = np.full((int(st.scores.shape[0]), len(outputs)), 0.5, dtype=np.float32)
        for x, y in self.train_ds:
            pred = self._predict(x)
            rows = np.asarray(y[ROW_ID_KEY]).reshape(-1)
            for j, k in enumerate(outputs):
                if k == "propaganda_max_pool":
                    stacked = tf.stack([tf.reshape(pred[t], [-1]) for t in self.pooled_tasks], axis=1)
                    p = pool_probabilities(stacked, self.pool_strategy, axis=1)
                else:
                    p = pred[k]
                out[rows, j] = np.asarray(p, dtype=np.float32).reshape(-1)
        return out

    def on_epoch_end(self, epoch, logs=None):
        st = self.model._sd
        n = epoch + 1
        recs = self.scorer.records
        if len(recs) != n:
            raise RuntimeError("SelfDistillCallback must come after the EpochScorer")
        self.snapshots[n] = self._weights()
        pick = np.argmin if self.signal == "loss" else np.argmax
        best = {k: 1 + int(pick([r["val"][k][self.signal] for r in recs]))
                for k in st.outputs if k not in self._done}
        ready = {k: b for k, b in best.items() if n - b >= self.patience}
        by_epoch: Dict[int, List[str]] = {}
        for k, b in ready.items():
            by_epoch.setdefault(b, []).append(k)
        if by_epoch:
            current = self.snapshots[n]
            scores, on = st.scores.numpy(), st.on.numpy()
            for b, ks in sorted(by_epoch.items()):
                self._assign(self.snapshots[b])
                new = self._score_train(ks)
                for j, k in enumerate(ks):
                    col = st.outputs.index(k)
                    scores[:, col], on[col] = new[:, j], 1.0
                    self._done.add(k)
                    self.switches.append({"output": k, "best_epoch": b, "switched_after_epoch": n})
                f1s = ", ".join(f"{100 * recs[b - 1]['val'][k]['f1']:.2f}" for k in ks)
                sig = ", ".join(f"{recs[b - 1]['val'][k][self.signal]:.4f}" for k in ks)
                print(f"Epoch {n}: switched {ks} to self-distillation from epoch {b} "
                      f"(best val {self.signal} {sig}; val F1 {f1s})", flush=True)
            self._assign(current)
            st.scores.assign(scores)
            st.on.assign(on)
        recs[-1]["distilled"] = sorted(self._done)
        needed = {n} | {b for k, b in best.items() if k not in self._done}
        self.snapshots = {e: w for e, w in self.snapshots.items() if e in needed}
        if len(self._done) == len(st.outputs):
            print(f"Epoch {n}: every output has switched, stopping", flush=True)
            self.model.stop_training = True


# ---------------------------------------------------------------------------
# Gradient probe (negative-transfer metrics)
# ---------------------------------------------------------------------------

class GradientProbe(tf.keras.callbacks.Callback):
    """Every ``every`` training steps (and before the first), takes each task's
    gradient on the shared parameters over a fixed probe set of train batches
    and records the pairwise cosines, norms and how often two tasks' per-batch
    gradients point against each other.

    Task losses are the per-task class-weighted means of
    ``_mtl_task_losses_fn`` (pooled propaganda first), without nu_k / |T_k|,
    so cosines and conflict rates are what the weighting can't change; norms
    are of the unweighted losses. The probe set is the first train batches
    (up to ``max_batches``) that give every task ``min_labels`` labels; each
    task's probe gradient is the mean of its per-batch gradients.

    With ``lookahead_lr`` > 0 it also records task affinity (Fifty et al.,
    NeurIPS 2021): ``affinity[i][j] = 1 - L_j(after a step on task i) /
    L_j(now)``, the step taken on the shared parameters only and then undone.
    Positive means a step for task i lowers task j's loss. Training uses
    AdamW, whose step length is about lr x sqrt(#params) whatever the gradient
    size, so the lookahead step is task i's unit gradient times that length;
    ``affinity_small`` repeats it at a tenth of the length, where the result
    is close to first order (step x |g_j| x cos(i, j) / L_j)."""

    def __init__(self, train_ds, every: int, min_labels: int = 128, max_batches: int = 16,
                 lookahead_lr: float = 0.0):
        super().__init__()
        self.every, self.step, self.lookahead_lr = every, 0, lookahead_lr
        self.train_ds, self.min_labels, self.max_batches = train_ds, min_labels, max_batches
        self.records: List[Dict[str, Any]] = []

    def on_train_begin(self, logs=None):
        m = self.model
        self.task_names = list(m._mtl_task_loss_names)
        names = m.task_names
        self.batches, counts = [], None
        for x, y in self.train_ds:
            n = self._label_counts(y)
            counts = n if counts is None else counts + n
            self.batches.append((x, y))
            if counts.min() >= self.min_labels or len(self.batches) >= self.max_batches:
                break
        if not m.built:  # fit builds the model after on_train_begin
            m(self.batches[0][0], training=False)
        self.shared = shared_variables(m)
        print(f"Gradient probe: {len(self.batches)} batches, labels per task "
              f"{dict(zip(self.task_names, counts.tolist()))}, "
              f"{sum(int(np.prod(v.shape)) for v in self.shared):,} shared parameters")

        @tf.function
        def grads(x, y):
            with tf.GradientTape(persistent=True) as tape:
                y_pred = m(x, training=False)
                losses, has = m._mtl_task_losses_fn(y, {k: y_pred[k] for k in names})
                # Split per task inside the tape: ops outside it get no gradient.
                per_task = tf.unstack(losses)
            rows = []
            for loss in per_task:
                g = tape.gradient(loss, self.shared,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
                rows.append(tf.concat([tf.reshape(tf.convert_to_tensor(v), [-1]) for v in g], 0))
            del tape
            return tf.stack(rows), losses, has

        @tf.function
        def task_losses(x, y):
            y_pred = m(x, training=False)
            return m._mtl_task_losses_fn(y, {k: y_pred[k] for k in names})

        self._grads, self._task_losses = grads, task_losses
        self.epoch = 0
        self._probe()

    def _label_counts(self, y) -> np.ndarray:
        """Labelled rows per task, in ``task_names`` order (pool = any technique)."""
        def valid(k):
            return tf.reshape(tf.not_equal(tf.cast(y[k], tf.float32), -1.0), [-1])
        pooled = self.model._mtl_pooled_tasks
        out = []
        for t in self.task_names:
            if t == "propaganda_max_pool":
                v = tf.reduce_any(tf.stack([valid(k) for k in pooled], 1), 1)
            else:
                v = valid(t)
            out.append(int(tf.reduce_sum(tf.cast(v, tf.int32))))
        return np.array(out)

    def _probe(self):
        total, n_batches = 0.0, 0.0
        conflicts, pairs, loss_sum = 0.0, 0.0, 0.0
        for x, y in self.batches:
            g, losses, has = self._grads(x, y)
            has = tf.cast(has, tf.float32)
            total += g * has[:, None]
            n_batches += has
            loss_sum += losses * has
            unit = tf.math.l2_normalize(g, axis=1)
            both = has[:, None] * has[None, :]
            conflicts += tf.cast(tf.matmul(unit, unit, transpose_b=True) < 0, tf.float32) * both
            pairs += both
        mean = total / tf.maximum(n_batches, 1.0)[:, None]
        unit = tf.math.l2_normalize(mean, axis=1)
        self.records.append({
            "step": self.step, "epoch": self.epoch,
            "cosine": tf.matmul(unit, unit, transpose_b=True).numpy().tolist(),
            "norm": tf.norm(mean, axis=1).numpy().tolist(),
            "conflict_rate": (conflicts / tf.maximum(pairs, 1.0)).numpy().tolist(),
        })
        if self.lookahead_lr > 0:
            before = loss_sum / tf.maximum(n_batches, 1.0)
            self.records[-1]["affinity"] = self._affinity(unit, before, 1.0)
            self.records[-1]["affinity_small"] = self._affinity(unit, before, 0.1)
            self.records[-1]["loss"] = before.numpy().tolist()

    def _affinity(self, unit, before, scale: float) -> List[List[float]]:
        sizes = [int(np.prod(v.shape)) for v in self.shared]
        length = scale * self.lookahead_lr * float(np.sqrt(sum(sizes)))
        saved = [tf.identity(v) for v in self.shared]
        rows = []
        for i in range(len(self.task_names)):
            for v, d in zip(self.shared, tf.split(unit[i] * length, sizes)):
                v.assign_sub(tf.reshape(d, v.shape))
            after, n = 0.0, 0.0
            for x, y in self.batches:
                losses, has = self._task_losses(x, y)
                has = tf.cast(has, tf.float32)
                after += losses * has
                n += has
            rows.append((1.0 - (after / tf.maximum(n, 1.0)) / (before + 1e-12)).numpy().tolist())
            for v, orig in zip(self.shared, saved):
                v.assign(orig)
        return rows

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.every == 0:
            self._probe()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
