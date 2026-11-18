"""Adaptive task balancing: FAMO and DB-MTL.
"""

from __future__ import annotations

from typing import List

import numpy as np
import tensorflow as tf

from multitask import MultiTaskModel

from multitask.train.trainer import PooledOutputLossTrainer

# ---------------------------------------------------------------------------
# Adaptive task balancing: FAMO and DB-MTL
# ---------------------------------------------------------------------------

class FAMO:
    """Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023).

    Task weights are z = softmax(w). The model step minimises
    sum_t z_t log l_t / c with c = sum_t z_t / l_t held fixed. After the step
    the same batch is scored again and w moves towards the tasks whose loss
    fell least. ``w_lr`` and ``gamma`` (Adam weight decay on w) are the
    authors' defaults.
    """

    def __init__(self, num_tasks: int, w_lr: float = 0.025, gamma: float = 0.01,
                 eps: float = 1e-8):
        self.w_lr, self.gamma, self.eps = w_lr, gamma, eps
        self.w = tf.Variable(tf.zeros([num_tasks]), trainable=False, name="famo_w")
        # Adam state for w, kept by hand so no optimizer is tied to the model.
        self.m = tf.Variable(tf.zeros([num_tasks]), trainable=False, name="famo_m")
        self.v = tf.Variable(tf.zeros([num_tasks]), trainable=False, name="famo_v")
        self.t = tf.Variable(0.0, trainable=False, name="famo_t")

    def objective(self, losses, has_labels):
        z = tf.nn.softmax(self.w) * has_labels
        d = losses + self.eps
        c = tf.stop_gradient(tf.reduce_sum(z / d))
        return tf.reduce_sum(z * tf.math.log(d)) / c

    def update(self, prev_losses, new_losses, has_labels):
        delta = (tf.math.log(prev_losses + self.eps)
                 - tf.math.log(new_losses + self.eps)) * has_labels
        z = tf.nn.softmax(self.w)
        # Gradient of z . delta with respect to w, plus Adam-style weight decay.
        g = z * (delta - tf.reduce_sum(z * delta)) + self.gamma * self.w
        self.t.assign_add(1.0)
        self.m.assign(0.9 * self.m + 0.1 * g)
        self.v.assign(0.999 * self.v + 0.001 * tf.square(g))
        m_hat = self.m / (1.0 - tf.pow(0.9, self.t))
        v_hat = self.v / (1.0 - tf.pow(0.999, self.t))
        self.w.assign_sub(self.w_lr * m_hat / (tf.sqrt(v_hat) + 1e-8))


class DBMTL:
    """Dual-Balancing Multi-Task Learning (Lin et al., 2023).

    Loss scale: each task trains on log(l_t). Gradient scale: each task's
    gradient on the shared parameters is kept as an exponential moving average
    (``beta``), and every average is rescaled to the largest one's norm before
    they are summed. Task-specific parameters get the plain sum of log-loss
    gradients, as in LibMTL.
    """

    def __init__(self, shared_vars, num_tasks: int, beta: float = 0.9, eps: float = 1e-8):
        self.shared_vars = list(shared_vars)
        self.beta, self.eps = beta, eps
        self.shapes = [v.shape for v in self.shared_vars]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.buffer = tf.Variable(tf.zeros([num_tasks, sum(self.sizes)]),
                                  trainable=False, name="dbmtl_grad_ema")

    def shared_gradients(self, tape, log_losses, has_labels):
        """``log_losses`` is a list of per-task scalars computed inside ``tape``."""
        rows = []
        for log_loss in log_losses:
            grads = tape.gradient(log_loss, self.shared_vars,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
            rows.append(tf.concat(
                [tf.reshape(tf.convert_to_tensor(g), [-1]) for g in grads], axis=0))
        has = has_labels[:, None]
        # A task with no labels in this batch keeps its previous average.
        ema = has * (self.beta * self.buffer + (1.0 - self.beta) * tf.stack(rows)) \
            + (1.0 - has) * self.buffer
        self.buffer.assign(ema)
        norms = tf.norm(ema, axis=1)
        alpha = tf.reduce_max(norms) / (norms + self.eps)
        flat = tf.reduce_sum(alpha[:, None] * ema, axis=0)
        return [tf.reshape(g, s) for g, s in zip(tf.split(flat, self.sizes), self.shapes)]


def shared_variables(model) -> List[tf.Variable]:
    """Everything outside the branch, output and masked-word head layers: the
    encoder and the shared dense layers."""
    own_layers = [layer for level in model.branch_layers for branch in level for layer in branch]
    own_layers += list(model.output_layers.values())
    if getattr(model, "mlm_head", None) is not None:  # only the masked-word loss reaches it
        own_layers.append(model.mlm_head)
    own = {id(v) for layer in own_layers for v in layer.trainable_variables}
    shared = [v for v in model.trainable_variables if id(v) not in own]
    if not shared or len(shared) == len(model.trainable_variables):
        raise RuntimeError("Could not split shared from task parameters")
    return shared


class BalancedMultiTaskModel(MultiTaskModel):
    """MultiTaskModel whose training step weights the per-task losses with
    FAMO or DB-MTL (``_balancer``, set by BalancedLossTrainer) instead of the
    fixed nu_k / |T_k| weighting. The reported and validation loss is still
    the fixed weighted loss, so early stopping is unchanged."""

    def train_step(self, data):
        x, y = data[0], data[1]
        bal = self._balancer
        names = self.task_names
        variables = self.trainable_variables
        with tf.GradientTape(persistent=isinstance(bal, DBMTL)) as tape:
            y_pred = self(x, training=True)
            y_pred = {k: y_pred[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            if isinstance(bal, FAMO):
                objective = bal.objective(losses, has_labels)
            else:
                # Split per task inside the tape: ops outside it get no gradient.
                log_losses = tf.unstack(tf.math.log(losses + bal.eps))
                own_objective = tf.add_n(
                    [l * h for l, h in zip(log_losses, tf.unstack(has_labels))])

        if isinstance(bal, FAMO):
            grads = tape.gradient(objective, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            new_pred = self(x, training=True)
            new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
            bal.update(losses, new_losses, has_labels)
        else:
            shared_ids = {id(v) for v in bal.shared_vars}
            own = [v for v in variables if id(v) not in shared_ids]
            own_grads = tape.gradient(own_objective, own)
            shared_grads = bal.shared_gradients(tape, log_losses, has_labels)
            del tape
            by_id = {id(v): g for v, g in zip(bal.shared_vars, shared_grads)}
            by_id.update({id(v): g for v, g in zip(own, own_grads)})
            self.optimizer.apply_gradients([(by_id[id(v)], v) for v in variables])

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        return {'loss': loss}


class BalancedLossTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a BalancedMultiTaskModel: builds the model on
    one batch, then attaches a FAMO or DB-MTL balancer over the 10 outputs
    (pooled propaganda + the non-pooled tasks)."""

    def __init__(self, *args, balancer: str, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, BalancedMultiTaskModel):
            raise TypeError("BalancedLossTrainer needs a BalancedMultiTaskModel")
        self.balancer_name = balancer

    def fit(self, train_dataset, *args, **kwargs):
        self.model(next(iter(train_dataset))[0], training=False)  # create all variables
        num_tasks = len(self.model.task_names) - len(self.pooled_tasks) + 1
        if self.balancer_name == "famo":
            bal = FAMO(num_tasks)
        elif self.balancer_name == "dbmtl":
            shared = shared_variables(self.model)
            print(f"DB-MTL: {len(shared)} shared variables, "
                  f"{sum(int(np.prod(v.shape)) for v in shared):,} parameters")
            bal = DBMTL(shared, num_tasks)
        else:
            raise ValueError(f"Unknown balancer {self.balancer_name!r}")
        print(f"Loss balancer: {self.balancer_name} over {num_tasks} tasks")
        self.model._balancer = bal
        return super().fit(train_dataset, *args, **kwargs)
