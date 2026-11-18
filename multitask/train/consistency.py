"""Consistency training (famo_consf1_m5 and the teacher variants).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import tensorflow as tf

from multitask import MultiTaskModel

from multitask.train.balancers import BalancedLossTrainer, BalancedMultiTaskModel, FAMO
from multitask.train.mlm import BalancedMaskedLMModel, BalancedMaskedLMTrainer, attach_mlm_balancing
from multitask.train.trainer import PooledOutputLossTrainer

# ---------------------------------------------------------------------------
# Label-aware masking: the masked copy goes through the whole model
# ---------------------------------------------------------------------------

LABEL_AWARE_TRACKERS = ("masked_label_loss", "consistency_loss", "consistency_weight",
                        "consistency_kept")


def confident_cells(t, eligible, share: float, floor: float = 0.0):
    """Per head, the eligible cells whose full-sentence score is among the
    lowest or highest ``share`` of eligible scores in the batch (at least one
    each side). Ranking within the head keeps both sides equal in size, which
    a single global cut-off does not on class-weighted, imbalanced heads.

    With ``floor`` > 0 a kept cell must also score <= ``floor`` (low side) or
    >= 1 - ``floor`` (high side). Both sides shrink to the smaller count, so
    they stay equal, and a head with no confident cells on one side keeps
    none at all in that batch."""
    n = tf.reduce_sum(tf.cast(eligible, tf.int32))
    k = tf.maximum(1, tf.cast(tf.floor(share * tf.cast(n, tf.float32)), tf.int32))
    k = tf.minimum(k, tf.maximum(n, 1))
    if floor:
        n_low = tf.reduce_sum(tf.cast(eligible & (t <= floor), tf.int32))
        n_high = tf.reduce_sum(tf.cast(eligible & (t >= 1.0 - floor), tf.int32))
        k = tf.minimum(k, tf.minimum(n_low, n_high))
    low = tf.sort(tf.where(eligible, t, 2.0))  # ineligible cells sort last
    high = tf.sort(tf.where(eligible, t, -2.0), direction="DESCENDING")
    i = tf.maximum(k - 1, 0)
    keep = (t <= low[i]) | (t >= high[i])
    return eligible & keep & (n > 0) & (k > 0)


def check_consistency_filter(keep: float, floor: float) -> None:
    if not 0.0 <= keep <= 0.5:
        raise ValueError("consistency_keep is a share per side, in [0, 0.5]")
    if not 0.0 <= floor < 0.5:
        raise ValueError("consistency_floor is a score distance from 0 or 1, in [0, 0.5)")
    if floor and not keep:
        raise ValueError("consistency_floor needs consistency_keep")


class ConsistencyMixin:
    """The consistency term shared by LabelAwareMaskedModel and
    TeacherConsistencyModel. Reads ``consistency_keep``, ``consistency_floor``,
    ``consistency_skip`` and ``consistency_rampup_steps`` from the model."""

    def consistency_loss(self, target, student, y=None):
        """(mean binary KL(target answer || student answer) over the cells
        used, share of cells used). The target answers are held fixed.
        Without ``consistency_keep`` every row of the 10 outputs is used; ``y``
        is needed only for ``consistency_skip``."""
        eps = 1e-7
        keep_share = getattr(self, "consistency_keep", 0.0)
        floor = getattr(self, "consistency_floor", 0.0)
        skip = getattr(self, "consistency_skip", {}) if y is not None else {}
        total, used, cells = 0.0, 0.0, 0.0
        for k in self._mtl_task_loss_names:
            t = tf.clip_by_value(tf.stop_gradient(
                tf.reshape(tf.cast(target[k], tf.float32), [-1])), eps, 1.0 - eps)
            p = tf.clip_by_value(tf.reshape(tf.cast(student[k], tf.float32), [-1]), eps, 1.0 - eps)
            kl = t * tf.math.log(t / p) + (1.0 - t) * tf.math.log((1.0 - t) / (1.0 - p))
            keep = tf.ones_like(t, dtype=tf.bool)
            if keep_share:
                for task in skip.get(k, []):
                    keep &= tf.equal(tf.reshape(tf.cast(y[task], tf.float32), [-1]),
                                     self.config.mask_value)
                keep = confident_cells(t, keep, keep_share, floor)
            w = tf.cast(keep, tf.float32)
            total += tf.reduce_sum(kl * w)
            used += tf.reduce_sum(w)
            cells += tf.cast(tf.size(t), tf.float32)
        return total / tf.maximum(used, 1.0), used / cells

    def consistency_ramp(self):
        """0 at the first step, rising linearly to 1 after the ramp."""
        if not self.consistency_rampup_steps:
            return tf.constant(1.0)
        step = tf.cast(self.optimizer.iterations, tf.float32)
        return tf.minimum(1.0, step / float(self.consistency_rampup_steps))


class LabelAwareMaskedModel(ConsistencyMixin, BalancedMaskedLMModel):
    """FAMO over the 10 task losses plus ``mlm_lambda`` x masked-word loss
    (famo_fixed), plus the masked copy of the batch sent up through the shared,
    branch and output layers. The masked copy reuses the masked-word encoder
    pass, so the extra cost is the small dense layers. Two extra terms, both
    outside FAMO like the masked-word loss:

    - ``masked_label_weight`` x the fixed-weight supervised loss on the masked
      copy against the real labels, so labelled cells only.
    - ``consistency_weight`` x the binary KL divergence from the full-sentence
      answers (no gradient) to the masked-copy answers, over every row and the
      10 outputs (pooled propaganda counted once), labelled or not. KL rather
      than BCE so it is 0 when the answers match; the gradient is the same.
      The weight rises linearly from 0 over ``consistency_rampup_steps``.
      With ``consistency_keep`` > 0 only confident targets count: per head,
      the cells whose full-sentence score is in the batch's lowest or highest
      ``consistency_keep`` share (``confident_cells``), skipping rows that
      have a label for any task in ``consistency_skip[head]`` (used to keep
      Propaganda rows, with their rule-made IAC labels, out of the IAC
      heads' targets). ``consistency_floor`` > 0 also requires a kept score
      to be <= floor or >= 1 - floor (``confident_cells``). Targets stay soft
      scores; nothing is sharpened.

    As in BalancedMultiTaskModel, the reported and validation loss is the
    fixed weighted supervised loss on full sentences, so early stopping is
    unchanged. Call ``attach_label_aware`` after ``attach_mlm_balancing``."""

    def attach_label_aware(self, masked_label_weight: float, consistency_weight: float,
                           consistency_rampup_steps: int = 0, consistency_keep: float = 0.0,
                           consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                           consistency_floor: float = 0.0):
        if not isinstance(self._balancer, FAMO) or self.mlm_mode != "fixed":
            raise ValueError("Label-aware masking is built on famo_fixed: FAMO, mlm_mode 'fixed'")
        check_consistency_filter(consistency_keep, consistency_floor)
        self.masked_label_weight = float(masked_label_weight)
        self.consistency_weight = float(consistency_weight)
        self.consistency_rampup_steps = int(consistency_rampup_steps)
        self.consistency_keep = float(consistency_keep)
        self.consistency_floor = float(consistency_floor)
        self.consistency_skip = {k: list(v) for k, v in (consistency_skip or {}).items()}
        for k in LABEL_AWARE_TRACKERS:  # attributes, so Keras resets them every epoch
            setattr(self, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        print(f"Label-aware masking: masked-label weight {self.masked_label_weight:g}, "
              f"consistency weight {self.consistency_weight:g} "
              f"(ramp over {self.consistency_rampup_steps} steps), mask prob "
              f"{self.mlm_mask_prob:g}, confident share per side {self.consistency_keep:g}, "
              f"floor {self.consistency_floor:g}, "
              f"{len(self.consistency_skip)} heads skip rows labelled for other tasks")

    def masked_pass(self, x, training, masking):
        """One encoder pass on the masked ids, giving (masked-word loss,
        masked-word accuracy, model outputs on the masked copy)."""
        masked, answers = masking
        enc = self.encoder({"input_ids": masked, "attention_mask": x["attention_mask"]},
                           training=training)
        mlm_loss, mlm_acc = self.mlm_loss_from_hidden(enc.last_hidden_state, answers)
        # Same choice as MultiTaskModel._encode_input for HuggingFace encoders.
        pooled = getattr(enc, "pooler_output", None)  # ELECTRA's output has no such field
        rep = pooled if pooled is not None else enc.last_hidden_state[:, 0, :]
        return mlm_loss, mlm_acc, self.forward_from_encoding(rep, training=training)

    def label_aware_losses(self, x, y, clean, training, masking):
        """(masked-word loss, accuracy, masked-label loss, consistency loss,
        share of cells the consistency loss used)."""
        mlm_loss, mlm_acc, masked_out = self.masked_pass(x, training, masking)
        label_loss = self._mtl_loss_fn(y, {k: masked_out[k] for k in self.task_names})
        cons_loss, kept = self.consistency_loss(clean, masked_out, y)
        return mlm_loss, mlm_acc, label_loss, cons_loss, kept

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, names = self._balancer, self.task_names
        variables = self.trainable_variables
        masking = self.mask_batch(x)
        cons_weight = self.consistency_weight * self.consistency_ramp()
        with tf.GradientTape() as tape:
            clean = self(x, training=True)
            y_pred = {k: clean[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            mlm_loss, mlm_acc, label_loss, cons_loss, kept = self.label_aware_losses(
                x, y, clean, True, masking)
            objective = (bal.objective(losses, has_labels) + self.mlm_lambda * mlm_loss
                         + self.masked_label_weight * label_loss + cons_weight * cons_loss)
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
        out = {'loss': loss}
        for tracker, v in ((self.mlm_loss_tracker, mlm_loss), (self.mlm_acc_tracker, mlm_acc),
                           (self.masked_label_loss_tracker, label_loss),
                           (self.consistency_loss_tracker, cons_loss),
                           (self.consistency_weight_tracker, cons_weight),
                           (self.consistency_kept_tracker, kept)):
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        return out

    def test_step(self, data):
        x, y = data[0], data[1]
        clean = self(x, training=False)
        mlm_loss, mlm_acc, label_loss, cons_loss, kept = self.label_aware_losses(
            x, y, clean, False, self.mask_batch(x))
        for tracker, v in ((self.mlm_loss_tracker, mlm_loss), (self.mlm_acc_tracker, mlm_acc),
                           (self.masked_label_loss_tracker, label_loss),
                           (self.consistency_loss_tracker, cons_loss),
                           (self.consistency_kept_tracker, kept)):
            tracker.update_state(v)
        return MultiTaskModel.test_step(self, data)


class LabelAwareMaskedTrainer(BalancedMaskedLMTrainer):
    """BalancedMaskedLMTrainer (FAMO, mlm_mode 'fixed') for a
    LabelAwareMaskedModel: also sets the two label-aware weights, with the
    consistency ramp sized in epochs of the train dataset."""

    def __init__(self, *args, masked_label_weight: float, consistency_weight: float,
                 consistency_rampup_epochs: float = 1.0, consistency_keep: float = 0.0,
                 consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                 consistency_floor: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, LabelAwareMaskedModel):
            raise TypeError("LabelAwareMaskedTrainer needs a LabelAwareMaskedModel")
        if self.balancer_name != "famo" or self.mlm_mode != "fixed":
            raise ValueError("Label-aware masking is built on famo_fixed: FAMO, mlm_mode 'fixed'")
        self.masked_label_weight = masked_label_weight
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.consistency_keep, self.consistency_skip = consistency_keep, consistency_skip
        self.consistency_floor = consistency_floor

    def fit(self, train_dataset, *args, **kwargs):
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("The consistency ramp needs a train dataset of known length")
        self.prepare(train_dataset)
        num_tasks = len(self.model.task_names) - len(self.pooled_tasks) + 1
        attach_mlm_balancing(self.model, self.balancer_name, self.mlm_mode, num_tasks)
        self.model.attach_label_aware(self.masked_label_weight, self.consistency_weight,
                                      round(self.consistency_rampup_epochs * steps),
                                      self.consistency_keep, self.consistency_skip,
                                      self.consistency_floor)
        return PooledOutputLossTrainer.fit(self, train_dataset, *args, **kwargs)


# ---------------------------------------------------------------------------
# Teacher consistency: dropout views and an EMA (mean) teacher, no masking
# ---------------------------------------------------------------------------

TEACHER_TRACKERS = ("consistency_loss", "consistency_weight", "consistency_kept",
                    "teacher_gap", "ema_decay")
CONSISTENCY_TEACHERS = ("current", "ema")


class _Untracked:
    """Holds the EMA teacher so Keras does not track it as part of the
    student: its weights must stay out of ``trainable_variables``, the
    optimizer and ``get_weights``."""

    def __init__(self, model):
        self.model = model


class TeacherConsistencyModel(ConsistencyMixin, BalancedMultiTaskModel):
    """FAMO over the 10 task losses plus ``consistency_weight`` x the binary
    KL divergence from a teacher's answers (held fixed, dropout off) to the
    student's answers, with the confidence filter of LabelAwareMaskedModel.
    No masking and no masked-word loss. Two switches:

    - teacher "current": the model itself with dropout off; "ema": a copy
      whose weights are an exponential moving average of the student's,
      decay min(``ema_decay``, 1 - 1/(step + 1)) as in Mean Teacher (Tarvainen
      & Valpola, 2017), also run with dropout off.
    - ``student_dropout``: True scores the supervised pass (encoder dropout
      on, so no extra pass); False runs a second student pass with dropout off.

    "current" with ``student_dropout`` False compares the model with itself,
    so the loss is always 0 and it is refused. ``teacher_gap`` logs the mean
    |teacher - student| answer over every cell of the 10 outputs, before the
    filter. As in BalancedMultiTaskModel, the reported and validation loss is
    the fixed weighted supervised loss, so early stopping is unchanged. Call
    ``attach_teacher`` after the balancer is set."""

    def attach_teacher(self, teacher: Optional[MultiTaskModel], student_dropout: bool,
                       consistency_weight: float, consistency_rampup_steps: int = 0,
                       consistency_keep: float = 0.0, consistency_floor: float = 0.0,
                       consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                       ema_decay: float = 0.99):
        if not isinstance(self._balancer, FAMO):
            raise ValueError("Teacher consistency is built on FAMO")
        if teacher is None and not student_dropout:
            raise ValueError("The current-weights teacher needs student dropout: "
                             "otherwise both answers are identical and the loss is 0")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        check_consistency_filter(consistency_keep, consistency_floor)
        if teacher is not None:
            pairs = list(zip(teacher.trainable_variables, self.trainable_variables))
            if len(teacher.trainable_variables) != len(self.trainable_variables) or any(
                    t.shape != s.shape for t, s in pairs):
                raise ValueError("The teacher's weights do not line up with the student's")
            for t, s in pairs:
                t.assign(s)
            self._teacher = _Untracked(teacher)
        self.student_dropout = bool(student_dropout)
        self.ema_decay = float(ema_decay)
        self.consistency_weight = float(consistency_weight)
        self.consistency_rampup_steps = int(consistency_rampup_steps)
        self.consistency_keep = float(consistency_keep)
        self.consistency_floor = float(consistency_floor)
        self.consistency_skip = {k: list(v) for k, v in (consistency_skip or {}).items()}
        for k in TEACHER_TRACKERS:  # attributes, so Keras resets them every epoch
            setattr(self, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        print(f"Teacher consistency: teacher {'ema' if teacher is not None else 'current'}"
              f"{f' (decay {self.ema_decay:g})' if teacher is not None else ''}, student "
              f"dropout {'on' if self.student_dropout else 'off'}, weight "
              f"{self.consistency_weight:g} (ramp over {self.consistency_rampup_steps} steps), "
              f"confident share per side {self.consistency_keep:g}, floor "
              f"{self.consistency_floor:g}, {len(self.consistency_skip)} heads skip rows "
              "labelled for other tasks")

    @property
    def teacher(self):
        box = getattr(self, "_teacher", None)
        return box.model if box is not None else None

    def teacher_outputs(self, x):
        net = self.teacher if self.teacher is not None else self
        out = net(x, training=False)
        return {k: tf.stop_gradient(out[k]) for k in self._mtl_task_loss_names}

    def teacher_gap(self, target, student):
        gaps = [tf.reduce_mean(tf.abs(tf.cast(target[k], tf.float32)
                                      - tf.cast(student[k], tf.float32)))
                for k in self._mtl_task_loss_names]
        return tf.add_n(gaps) / len(gaps)

    def update_teacher(self):
        """Move the EMA teacher toward the student; returns the decay used."""
        if self.teacher is None:
            return tf.constant(0.0)
        step = tf.cast(self.optimizer.iterations, tf.float32)
        decay = tf.minimum(self.ema_decay, 1.0 - 1.0 / (step + 1.0))
        for t, s in zip(self.teacher.trainable_variables, self.trainable_variables):
            t.assign(decay * t + (1.0 - decay) * s)
        return decay

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, names = self._balancer, self.task_names
        variables = self.trainable_variables
        cons_weight = self.consistency_weight * self.consistency_ramp()
        target = self.teacher_outputs(x)
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            y_pred = {k: pred[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            student = pred if self.student_dropout else self(x, training=False)
            cons_loss, kept = self.consistency_loss(target, student, y)
            objective = bal.objective(losses, has_labels) + cons_weight * cons_loss
        grads = tape.gradient(objective, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        new_pred = self(x, training=True)
        new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
        bal.update(losses, new_losses, has_labels)
        decay = self.update_teacher()

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        out = {'loss': loss}
        for tracker, v in ((self.consistency_loss_tracker, cons_loss),
                           (self.consistency_weight_tracker, cons_weight),
                           (self.consistency_kept_tracker, kept),
                           (self.teacher_gap_tracker, self.teacher_gap(target, student)),
                           (self.ema_decay_tracker, decay)):
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        return out

    def test_step(self, data):
        """Val logs compare the teacher with the student, both with dropout off
        (so 0 for the current-weights teacher)."""
        x, y = data[0], data[1]
        target, student = self.teacher_outputs(x), self(x, training=False)
        cons_loss, kept = self.consistency_loss(target, student, y)
        for tracker, v in ((self.consistency_loss_tracker, cons_loss),
                           (self.consistency_kept_tracker, kept),
                           (self.teacher_gap_tracker, self.teacher_gap(target, student))):
            tracker.update_state(v)
        return MultiTaskModel.test_step(self, data)


class TeacherConsistencyTrainer(BalancedLossTrainer):
    """BalancedLossTrainer (FAMO) for a TeacherConsistencyModel: builds the
    EMA teacher as a second model of the same config (copied from the student
    once built), then attaches the consistency settings, with the ramp sized
    in epochs of the train dataset."""

    def __init__(self, *args, teacher: str, student_dropout: bool, consistency_weight: float,
                 consistency_rampup_epochs: float = 1.0, consistency_keep: float = 0.0,
                 consistency_floor: float = 0.0,
                 consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                 ema_decay: float = 0.99, **kwargs):
        super().__init__(*args, balancer="famo", **kwargs)
        if not isinstance(self.model, TeacherConsistencyModel):
            raise TypeError("TeacherConsistencyTrainer needs a TeacherConsistencyModel")
        if teacher not in CONSISTENCY_TEACHERS:
            raise ValueError(f"teacher must be one of {CONSISTENCY_TEACHERS}, got {teacher!r}")
        self.teacher_name, self.student_dropout = teacher, student_dropout
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.consistency_keep, self.consistency_floor = consistency_keep, consistency_floor
        self.consistency_skip, self.ema_decay = consistency_skip, ema_decay

    def fit(self, train_dataset, *args, **kwargs):
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("The consistency ramp needs a train dataset of known length")
        x = next(iter(train_dataset))[0]
        self.model(x, training=False)  # create all variables
        self.model._balancer = FAMO(len(self.model.task_names) - len(self.pooled_tasks) + 1)
        teacher = None
        if self.teacher_name == "ema":
            teacher = MultiTaskModel(self.model.config, dropout_rate=self.model.dropout_rate)
            teacher(x, training=False)
        self.model.attach_teacher(teacher, self.student_dropout, self.consistency_weight,
                                  round(self.consistency_rampup_epochs * steps),
                                  self.consistency_keep, self.consistency_floor,
                                  self.consistency_skip, self.ema_decay)
        return PooledOutputLossTrainer.fit(self, train_dataset, *args, **kwargs)
