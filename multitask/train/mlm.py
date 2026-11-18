"""Masked-word (MLM) auxiliary loss, alone or with a balancer.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from multitask import MultiTaskModel

from multitask.train.balancers import DBMTL, FAMO, shared_variables
from multitask.train.trainer import PooledOutputLossTrainer

# ---------------------------------------------------------------------------
# Masked-word (MLM) auxiliary loss
# ---------------------------------------------------------------------------

# BERT's uncased vocabulary, which the distilbert-base-uncased tokenizer in
# load_splits shares. Ids below 999 are [PAD], [unusedN] and the special
# tokens, so random replacement words are drawn from 999 up.
PAD_ID, CLS_ID, SEP_ID, MASK_ID, FIRST_WORD_ID = 0, 101, 102, 103, 999
MLM_IGNORE = -100


def mask_tokens(input_ids, attention_mask, vocab_size: int, mask_prob: float = 0.15):
    """BERT's masking rule. Picks ~``mask_prob`` of the real word tokens (never
    padding, [CLS] or [SEP]); of those, 80% become [MASK], 10% a random word
    and 10% stay as they are. Returns (masked ids, answers), where answers hold
    the original id at picked positions and MLM_IGNORE everywhere else."""
    ids = tf.convert_to_tensor(input_ids)
    candidate = (tf.cast(attention_mask, tf.bool) & tf.not_equal(ids, PAD_ID)
                 & tf.not_equal(ids, CLS_ID) & tf.not_equal(ids, SEP_ID))
    picked = candidate & (tf.random.uniform(tf.shape(ids)) < mask_prob)
    r = tf.random.uniform(tf.shape(ids))
    random_words = tf.random.uniform(tf.shape(ids), FIRST_WORD_ID, vocab_size, dtype=ids.dtype)
    masked = tf.where(picked & (r < 0.8), tf.cast(MASK_ID, ids.dtype), ids)
    masked = tf.where(picked & (r >= 0.8) & (r < 0.9), random_words, masked)
    answers = tf.where(picked, ids, tf.cast(MLM_IGNORE, ids.dtype))
    return masked, answers


class MLMHead(tf.keras.layers.Layer):
    """BERT's masked-word head: dense, activation, LayerNorm, then scores
    against the encoder's word-embedding table plus a per-word bias. The table
    is passed in on each call rather than stored, so it stays tied to (and is
    only tracked by) the encoder. Takes [positions, hidden]: LayerNormalization
    fixes its axis at build time, so the input rank must not change."""

    def __init__(self, hidden_size: int, vocab_size: int, activation="gelu",
                 layer_norm_eps: float = 1e-12, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")
        self.activation = tf.keras.activations.get(activation)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps, name="LayerNorm")

    def build(self, input_shape):
        self.bias = self.add_weight(name="bias", shape=[self.vocab_size], initializer="zeros")
        super().build(input_shape)

    def call(self, hidden, word_embeddings):
        h = self.layer_norm(self.activation(self.dense(hidden)))
        return tf.matmul(h, word_embeddings, transpose_b=True) + self.bias


def load_pretrained_mlm_head(model_name: str, from_pt: bool) -> MLMHead:
    """An MLMHead holding the checkpoint's own pretrained head weights."""
    from transformers import TFBertForMaskedLM
    from transformers.activations_tf import get_tf_activation

    try:
        src, info = TFBertForMaskedLM.from_pretrained(
            model_name, from_pt=from_pt, use_safetensors=False, output_loading_info=True)
    except TypeError:
        src, info = TFBertForMaskedLM.from_pretrained(
            model_name, from_pt=from_pt, output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "mlm" in k or "predictions" in k]
    if missing:
        raise RuntimeError(f"{model_name} has no pretrained MLM head weights: {missing}")
    cfg = src.config
    head = MLMHead(cfg.hidden_size, cfg.vocab_size, get_tf_activation(cfg.hidden_act),
                   cfg.layer_norm_eps, name="mlm_head")
    head(tf.zeros([1, cfg.hidden_size]), tf.zeros([cfg.vocab_size, cfg.hidden_size]))
    p = src.mlm.predictions
    head.dense.set_weights(p.transform.dense.get_weights())
    head.layer_norm.set_weights(p.transform.LayerNorm.get_weights())
    head.bias.assign(p.bias)
    return head


class NoMLMHead:
    """Stands in for the masked-word head on an encoder whose checkpoint has
    none (ELECTRA's discriminator). Only for ``mlm_lambda`` 0, where the
    masked-word loss has no weight anyway: masking still works (it only needs
    the vocabulary size), and the masked-word loss and accuracy are logged
    as 0."""

    trainable_variables: Tuple = ()

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size


class MaskedLMMultiTaskModel(MultiTaskModel):
    """MultiTaskModel trained on supervised loss + ``mlm_lambda`` x masked-word
    loss. The masked-word pass runs the encoder a second time on a masked copy
    of the batch and never reaches the heads. The validation loss, and so
    early stopping, is the supervised loss only; ``mlm_loss`` / ``mlm_acc``
    are logged for both splits. Call ``attach_mlm_head`` before training."""

    def attach_mlm_head(self, head: MLMHead, mlm_lambda: float, mask_prob: float = 0.15):
        self.mlm_head = head
        self.mlm_lambda = float(mlm_lambda)
        self.mlm_mask_prob = mask_prob
        self.mlm_loss_tracker = tf.keras.metrics.Mean(name="mlm_loss")
        self.mlm_acc_tracker = tf.keras.metrics.Mean(name="mlm_acc")

    def mask_batch(self, x):
        return mask_tokens(x["input_ids"], x["attention_mask"],
                           self.mlm_head.vocab_size, self.mlm_mask_prob)

    def mlm_loss(self, x, training, masking=None):
        """Mean cross-entropy and accuracy over this batch's masked positions.
        ``masking`` = (masked ids, answers) from ``mask_batch``; a new random
        masking is drawn when it's None."""
        masked, answers = masking if masking is not None else self.mask_batch(x)
        hidden = self.encoder({"input_ids": masked, "attention_mask": x["attention_mask"]},
                              training=training).last_hidden_state
        return self.mlm_loss_from_hidden(hidden, answers)

    def mlm_loss_from_hidden(self, hidden, answers):
        """``mlm_loss`` for encoder outputs already computed on the masked ids."""
        if isinstance(self.mlm_head, NoMLMHead):
            return tf.constant(0.0), tf.constant(0.0)
        # Score only the picked positions: the full [B, L, vocab] tensor is huge.
        picked = tf.not_equal(answers, MLM_IGNORE)
        logits = self.mlm_head(tf.boolean_mask(hidden, picked),
                               self.encoder.get_input_embeddings().weight)
        targets = tf.boolean_mask(answers, picked)
        n = tf.maximum(tf.cast(tf.size(targets), tf.float32), 1.0)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)) / n
        hits = tf.equal(tf.argmax(logits, axis=-1, output_type=targets.dtype), targets)
        return loss, tf.reduce_sum(tf.cast(hits, tf.float32)) / n

    def train_objective(self, x, y, training=True):
        """(total, supervised, masked-word loss, masked-word accuracy)."""
        y_pred = self(x, training=training)
        sup_loss = self._mtl_loss_fn(y, {k: y_pred[k] for k in self.task_names})
        mlm_loss, mlm_acc = self.mlm_loss(x, training)
        return sup_loss + self.mlm_lambda * mlm_loss, sup_loss, mlm_loss, mlm_acc

    def train_step(self, data):
        x, y = data[0], data[1]
        with tf.GradientTape() as tape:
            total, sup_loss, mlm_loss, mlm_acc = self.train_objective(x, y)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(sup_loss)
                break
        self.mlm_loss_tracker.update_state(mlm_loss)
        self.mlm_acc_tracker.update_state(mlm_acc)
        return {'loss': sup_loss, 'mlm_loss': self.mlm_loss_tracker.result(),
                'mlm_acc': self.mlm_acc_tracker.result()}

    def test_step(self, data):
        mlm_loss, mlm_acc = self.mlm_loss(data[0], training=False)
        self.mlm_loss_tracker.update_state(mlm_loss)
        self.mlm_acc_tracker.update_state(mlm_acc)
        return super().test_step(data)


class MaskedLMTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a MaskedLMMultiTaskModel: builds the model on
    one batch, then attaches the encoder checkpoint's pretrained MLM head."""

    def __init__(self, *args, mlm_lambda: float, mask_prob: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, MaskedLMMultiTaskModel):
            raise TypeError("MaskedLMTrainer needs a MaskedLMMultiTaskModel")
        self.mlm_lambda, self.mask_prob = mlm_lambda, mask_prob

    def prepare(self, train_dataset):
        """Create all variables, then attach the pretrained head."""
        self.model(next(iter(train_dataset))[0], training=False)
        ecfg = self.model.encoder_config
        enc_cfg = self.model.encoder.config
        if enc_cfg.model_type == "bert":
            head = load_pretrained_mlm_head(ecfg.encoder_identifier, ecfg.from_pt)
            source = f"pretrained head from {ecfg.encoder_identifier}"
        elif self.mlm_lambda == 0:
            head = NoMLMHead(enc_cfg.vocab_size)
            source = f"no head ({enc_cfg.model_type} checkpoint has none; loss logged as 0)"
        else:
            raise ValueError(f"A masked-word loss with weight {self.mlm_lambda:g} needs a BERT "
                             f"checkpoint with its pretrained head, not {enc_cfg.model_type}")
        self.model.attach_mlm_head(head, self.mlm_lambda, self.mask_prob)
        print(f"Masked-word loss: lambda={self.mlm_lambda:g}, mask prob={self.mask_prob:g}, "
              f"{source}")

    def fit(self, train_dataset, *args, **kwargs):
        self.prepare(train_dataset)
        return super().fit(train_dataset, *args, **kwargs)


MLM_MODES = ("fixed", "task", "cosine")


class BalancedMaskedLMModel(MaskedLMMultiTaskModel):
    """Masked-word loss combined with a loss balancer or a gradient gate.
    ``mlm_mode`` (set by BalancedMaskedLMTrainer):

    - "fixed": FAMO / DB-MTL (``_balancer``) over the 10 task losses, plus
      ``mlm_lambda`` x masked-word loss, as in MaskedLMMultiTaskModel.
    - "task": the masked-word loss is an 11th task under the balancer, which
      sets its weight; ``mlm_lambda`` is unused.
    - "cosine": no balancer. Fixed supervised loss, plus ``mlm_lambda`` x the
      masked-word gradient on the shared parameters only on steps where it
      has positive cosine with the supervised gradient there (Du et al.,
      2018, "Adapting Auxiliary Losses Using Gradient Similarity"). The
      masked-word head always gets its gradient: it can't affect the tasks.

    As in BalancedMultiTaskModel, the reported and validation loss is the
    fixed weighted supervised loss, so early stopping is unchanged."""

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, mode, lam = self._balancer, self.mlm_mode, self.mlm_lambda
        names = self.task_names
        variables = self.trainable_variables
        masking = self.mask_batch(x)
        with tf.GradientTape(persistent=not isinstance(bal, FAMO)) as tape:
            y_pred = self(x, training=True)
            y_pred = {k: y_pred[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            mlm_loss, mlm_acc = self.mlm_loss(x, True, masking)
            if mode == "task":
                losses = tf.concat([losses, mlm_loss[None]], 0)
                has_labels = tf.concat([has_labels, tf.ones([1], has_labels.dtype)], 0)
            if isinstance(bal, FAMO):
                objective = bal.objective(losses, has_labels)
                if mode == "fixed":
                    objective += lam * mlm_loss
            elif isinstance(bal, DBMTL):
                # Split per task inside the tape: ops outside it get no gradient.
                log_losses = tf.unstack(tf.math.log(losses + bal.eps))
                own_objective = tf.add_n(
                    [l * h for l, h in zip(log_losses, tf.unstack(has_labels))])
                if mode == "fixed":
                    own_objective += lam * mlm_loss
            else:
                sup_loss = self._mtl_loss_fn(y, y_pred)

        logs = {}
        if isinstance(bal, FAMO):
            grads = tape.gradient(objective, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            new_pred = self(x, training=True)
            new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
            if mode == "task":
                new_mlm, _ = self.mlm_loss(x, True, masking)  # same masking as before the step
                new_losses = tf.concat([new_losses, new_mlm[None]], 0)
            bal.update(losses, new_losses, has_labels)
            if mode == "task":
                logs["mlm_weight"] = tf.nn.softmax(bal.w)[-1]
        else:
            shared = self._mlm_shared
            shared_ids = {id(v) for v in shared}
            own = [v for v in variables if id(v) not in shared_ids]
            zero = tf.UnconnectedGradients.ZERO
            if isinstance(bal, DBMTL):
                own_grads = tape.gradient(own_objective, own, unconnected_gradients=zero)
                shared_grads = bal.shared_gradients(tape, log_losses, has_labels)
                if mode == "fixed":
                    mlm_grads = tape.gradient(mlm_loss, shared, unconnected_gradients=zero)
                    shared_grads = [g + lam * tf.convert_to_tensor(m)
                                    for g, m in zip(shared_grads, mlm_grads)]
            else:
                head_ids = {id(v) for v in self.mlm_head.trainable_variables}
                head = [v for v in own if id(v) in head_ids]
                grads = tape.gradient(sup_loss, own + shared, unconnected_gradients=zero)
                own_grads, sup_grads = grads[:len(own)], grads[len(own):]
                grads = tape.gradient(mlm_loss, head + shared, unconnected_gradients=zero)
                head_grads, mlm_grads = iter(grads[:len(head)]), grads[len(head):]
                own_grads = [lam * next(head_grads) if id(v) in head_ids else g
                             for v, g in zip(own, own_grads)]

                def flat(gs):
                    return tf.concat([tf.reshape(tf.convert_to_tensor(g), [-1]) for g in gs], 0)
                a, b = flat(sup_grads), flat(mlm_grads)
                cos = tf.reduce_sum(a * b) / (tf.norm(a) * tf.norm(b) + 1e-12)
                gate = tf.cast(cos > 0, tf.float32)
                shared_grads = [tf.convert_to_tensor(g) + lam * gate * tf.convert_to_tensor(m)
                                for g, m in zip(sup_grads, mlm_grads)]
                logs["mlm_cos"], logs["mlm_gate"] = cos, gate
            del tape
            by_id = {id(v): g for v, g in zip(shared, shared_grads)}
            by_id.update({id(v): g for v, g in zip(own, own_grads)})
            self.optimizer.apply_gradients([(by_id[id(v)], v) for v in variables])

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        self.mlm_loss_tracker.update_state(mlm_loss)
        self.mlm_acc_tracker.update_state(mlm_acc)
        out = {'loss': loss, 'mlm_loss': self.mlm_loss_tracker.result(),
               'mlm_acc': self.mlm_acc_tracker.result()}
        for k, v in logs.items():
            self.mlm_extra_trackers[k].update_state(v)
            out[k] = self.mlm_extra_trackers[k].result()
        return out


def attach_mlm_balancing(model: BalancedMaskedLMModel, balancer: str, mlm_mode: str,
                         num_tasks: int) -> None:
    """Set up a BalancedMaskedLMModel whose head is attached. ``num_tasks``
    counts the supervised losses (10 with pooled propaganda)."""
    m = model
    m.mlm_mode = mlm_mode
    m._mlm_shared = shared_variables(m)  # after attach, so the head is excluded
    num_tasks += mlm_mode == "task"
    if balancer == "famo":
        m._balancer = FAMO(num_tasks)
    elif balancer == "dbmtl":
        m._balancer = DBMTL(m._mlm_shared, num_tasks)
    else:
        m._balancer = None
    extra = {"task": ["mlm_weight"] if balancer == "famo" else [],
             "cosine": ["mlm_cos", "mlm_gate"]}.get(mlm_mode, [])
    m.mlm_extra_trackers = {}
    for k in extra:  # attributes, so Keras resets them every epoch
        setattr(m, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        m.mlm_extra_trackers[k] = getattr(m, f"{k}_tracker")
    print(f"Masked-word mode {mlm_mode!r}, balancer {balancer or 'none'} "
          f"over {num_tasks} losses, {len(m._mlm_shared)} shared variables")


class BalancedMaskedLMTrainer(MaskedLMTrainer):
    """MaskedLMTrainer for a BalancedMaskedLMModel: attaches the pretrained
    head, then a FAMO or DB-MTL balancer over the 10 outputs (11 with the
    masked-word loss in "task" mode), or none in "cosine" mode."""

    def __init__(self, *args, balancer: str, mlm_mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, BalancedMaskedLMModel):
            raise TypeError("BalancedMaskedLMTrainer needs a BalancedMaskedLMModel")
        if mlm_mode not in MLM_MODES:
            raise ValueError(f"Unknown mlm_mode {mlm_mode!r}")
        if (mlm_mode == "cosine") == bool(balancer):
            raise ValueError("mlm_mode 'cosine' takes no balancer; 'fixed' and 'task' need one")
        self.balancer_name, self.mlm_mode = balancer, mlm_mode

    def fit(self, train_dataset, *args, **kwargs):
        self.prepare(train_dataset)
        num_tasks = len(self.model.task_names) - len(self.pooled_tasks) + bool(self.pooled_tasks)
        attach_mlm_balancing(self.model, self.balancer_name, self.mlm_mode, num_tasks)
        return PooledOutputLossTrainer.fit(self, train_dataset, *args, **kwargs)
