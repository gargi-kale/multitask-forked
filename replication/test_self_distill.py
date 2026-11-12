"""Checks for per-task stopping with self-distillation (SelfDistillModel,
SelfDistillCallback, the soft targets in the pooled loss) in
test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_self_distill.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_gradient_probe import M, NAMES  # noqa: E402
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, ROW_ID_KEY, EpochScorer, PooledOutputLossTrainer, SelfDistillCallback,
    SelfDistillModel, binary_kl)
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

OUTPUTS = ["propaganda_max_pool", "wa", "d1", "d2"]
N = 64


def data(seed=0, row_ids=False):
    rng = np.random.default_rng(seed)
    ids = rng.integers(999, 30522, size=(N, 16)).astype(np.int32)
    ids[:, 0], ids[:, -1] = 101, 102
    x = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    y = {k: np.full((N, 1), M, np.float32) for k in NAMES}
    for k, rows in (("p1", range(32)), ("p2", range(32)), ("wa", range(32, 48)),
                    ("d1", range(48, 64)), ("d2", range(48, 64))):
        y[k][list(rows), 0] = rng.integers(0, 2, size=len(rows))
    if row_ids:
        y[ROW_ID_KEY] = np.arange(N, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    # Only train is shuffled, as in load_splits: the scorer reads val labels
    # and val predictions in separate passes.
    return (ds.shuffle(N, seed=seed) if row_ids else ds).batch(16)


def make():
    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )
    model = SelfDistillModel(cfg, dropout_rate=0.0)
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 2.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    model._mtl_loss_fn = loss_fn
    return model, trainer, loss_fn


def test_binary_kl_is_zero_at_the_target_and_positive_elsewhere():
    t = tf.constant([0.0, 0.2, 0.5, 0.9, 1.0])
    np.testing.assert_allclose(binary_kl(t, t).numpy(), 0.0, atol=1e-5)
    assert np.all(binary_kl(t, tf.constant([0.3, 0.6, 0.1, 0.4, 0.5])).numpy() > 0)


def test_soft_targets_off_change_nothing_and_on_replace_the_label_term():
    tf.random.set_seed(0)
    model, _, loss_fn = make()
    x, y = next(iter(data(row_ids=True)))
    pred = model(x, training=False)
    y_pred = {k: pred[k] for k in NAMES}
    plain = float(loss_fn(y, y_pred))
    rng = np.random.default_rng(1)
    rand = {k: (tf.constant(rng.uniform(size=16), tf.float32), 0.0) for k in OUTPUTS}
    assert float(loss_fn(y, y_pred, rand)) == plain

    # "wa" switched on, its target = the model's own scores: its term is 0,
    # every other term and the labelled-row count are unchanged.
    own = {"wa": (tf.reshape(y_pred["wa"], [-1]), 1.0)}
    y_no_wa = dict(y, wa=tf.fill(tf.shape(y["wa"]), M))
    n_all = sum(float(tf.reduce_sum(tf.cast(tf.not_equal(y[k], M), tf.float32)))
                for k in ("wa", "d1", "d2"))
    n_all += float(tf.reduce_sum(tf.cast(tf.reduce_max(tf.stack(
        [tf.reshape(y["p1"], [-1]), tf.reshape(y["p2"], [-1])], 1), 1) != M, tf.float32)))
    n_no_wa = n_all - float(tf.reduce_sum(tf.cast(tf.not_equal(y["wa"], M), tf.float32)))
    expected = float(loss_fn(y_no_wa, y_pred)) * n_no_wa / n_all
    np.testing.assert_allclose(float(loss_fn(y, y_pred, own)), expected, rtol=1e-4)


def test_score_table_is_not_part_of_the_model_weights():
    model, _, _ = make()
    x, _ = next(iter(data(row_ids=True)))
    model(x, training=False)
    before = len(model.get_weights())
    model.attach_self_distill(OUTPUTS, N)
    assert len(model.get_weights()) == before


class WeightLog(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([v.numpy() for v in self.model.trainable_variables])


@pytest.mark.parametrize("patience,signal", [(1, "f1"), (2, "f1"), (2, "loss")])
def test_outputs_switch_after_patience_with_their_best_epoch_scores(patience, signal):
    tf.random.set_seed(0)
    model, trainer, _ = make()
    train, val, test = data(0, row_ids=True), data(1), data(2)
    x, _ = next(iter(train))
    model(x, training=False)
    model.attach_self_distill(OUTPUTS, N)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    scorer = EpochScorer(val, test, monitor="f1", patience=1, outputs=OUTPUTS, stops=False)
    scorer.trainer = trainer
    sd = SelfDistillCallback(scorer, train, patience=patience, pooled_tasks=["p1", "p2"],
                             signal=signal)
    pre, post = WeightLog(), WeightLog()  # just before / after the switch callback
    epochs = 8
    model.fit(train, epochs=epochs, validation_data=val, verbose=0,
              callbacks=[scorer, pre, sd, post])
    recs, st = scorer.records, model._sd
    switched = {s["output"]: s for s in sd.switches}
    assert len(switched) == len(sd.switches)  # each output switches once
    for k in OUTPUTS:
        col = OUTPUTS.index(k)
        if k not in switched:
            assert st.on.numpy()[col] == 0.0 and len(recs) == epochs
            continue
        s = switched[k]
        n, b = s["switched_after_epoch"], s["best_epoch"]
        v = [r["val"][k][signal] for r in recs[:n]]
        assert b == 1 + int(np.argmin(v) if signal == "loss" else np.argmax(v))
        assert n - b == patience
        assert st.on.numpy()[col] == 1.0
        assert k in recs[n - 1]["distilled"] and all(k not in r["distilled"] for r in recs[:n - 1])
        # stored scores = the best epoch's weights on every train row
        current = [v.numpy() for v in model.trainable_variables]
        for v, w in zip(model.trainable_variables, pre.weights[b - 1]):
            v.assign(w)
        want = sd._score_train([k])[:, 0]
        for v, w in zip(model.trainable_variables, current):
            v.assign(w)
        np.testing.assert_allclose(st.scores.numpy()[:, col], want, atol=1e-5)
    if len(switched) == len(OUTPUTS):
        assert len(recs) == max(s["switched_after_epoch"] for s in sd.switches)
    # the switch-time weight swaps put the live weights back every epoch
    for w_pre, w_post in zip(pre.weights, post.weights):
        for a, b in zip(w_pre, w_post):
            np.testing.assert_array_equal(a, b)
    # the scorer's restore is unaffected by the score table
    for a, b in zip([v.numpy() for v in model.trainable_variables],
                    pre.weights[scorer.best_epoch - 1]):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# FAMO + filtered consistency (famo_consf1_m5) with self-distillation
# ---------------------------------------------------------------------------

def make_famo(model_cls, x, head, keep=0.1):
    from test_no_prop_overlap import attach_mlm_balancing

    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )
    model = model_cls(cfg, dropout_rate=0.0)
    model(x, training=False)
    for layer in model.encoder.submodules:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = 0.0
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 2.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    model.attach_mlm_head(head, 0.0, 0.05)
    attach_mlm_balancing(model, "famo", "fixed", 4)
    model.attach_label_aware(0.0, 1.0, 0, keep, None)
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss=model._mtl_loss_fn)
    return model


@pytest.fixture(scope="module")
def head():
    from test_no_prop_overlap import load_pretrained_mlm_head
    return load_pretrained_mlm_head("google/bert_uncased_L-2_H-128_A-2", from_pt=True)


def famo_batch():
    x, y = next(iter(data(row_ids=True).unbatch().batch(N)))
    return x, y


def test_famo_variant_with_nothing_switched_matches_famo_consf1(head):
    from test_no_prop_overlap import LabelAwareMaskedModel, SelfDistillFamoModel

    x, y = famo_batch()
    base = make_famo(LabelAwareMaskedModel, x, head)
    sd = make_famo(SelfDistillFamoModel, x, head)
    sd.set_weights(base.get_weights())
    sd.attach_self_distill(OUTPUTS, N)
    for m in (base, sd):
        tf.random.set_seed(7)
        m.train_step((x, y))
    for a, b in zip(base.get_weights(), sd.get_weights()):
        np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(base._balancer.w.numpy(), sd._balancer.w.numpy(), atol=1e-6)


def test_famo_variant_moves_switched_outputs_out_of_famo(head):
    from test_no_prop_overlap import SelfDistillFamoModel

    x, y = famo_batch()
    model = make_famo(SelfDistillFamoModel, x, head)
    model.attach_self_distill(OUTPUTS, N)
    pred = model(x, training=False)
    y_pred = {k: pred[k] for k in NAMES}
    assert float(model.distill_loss(y, y_pred)) == 0.0  # nothing switched

    st, col = model._sd, OUTPUTS.index("wa")
    rng = np.random.default_rng(3)
    scores = st.scores.numpy()
    scores[:, col] = rng.uniform(0.05, 0.95, size=N)
    st.scores.assign(scores)
    st.on.assign(tf.one_hot(col, len(OUTPUTS)))
    # distill loss = class-weighted mean KL over wa's labelled rows
    yw = np.asarray(y["wa"]).reshape(-1)
    rows = np.asarray(y[ROW_ID_KEY]).reshape(-1)
    lab = yw != M
    t, p = scores[rows, col], np.asarray(y_pred["wa"]).reshape(-1)
    kl = binary_kl(tf.constant(t), tf.constant(p)).numpy()
    w = np.where(yw == 0, 1.0, 2.0)
    want = (kl * w)[lab].sum() / lab.sum()
    np.testing.assert_allclose(float(model.distill_loss(y, y_pred)), want, rtol=1e-4)

    # With wa's target = its own scores the distill gradient is 0, so a step
    # must match a model whose batch has no wa labels: FAMO leaves wa out
    # (no FAMO weight, no update signal) and nothing else changes.
    scores[rows, col] = p
    st.scores.assign(scores)
    ref = make_famo(SelfDistillFamoModel, x, head)
    ref.set_weights(model.get_weights())
    ref.attach_self_distill(OUTPUTS, N)
    y_nowa = dict(y, wa=tf.fill(tf.shape(y["wa"]), M))
    tf.random.set_seed(5)
    model.train_step((x, y))
    tf.random.set_seed(5)
    ref.train_step((x, y_nowa))
    np.testing.assert_allclose(model._balancer.w.numpy(), ref._balancer.w.numpy(), atol=1e-5)
    # trainable only: the loss trackers differ, since wa's labels differ
    for a, b in zip(model.trainable_variables, ref.trainable_variables):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)


def test_scorer_records_each_outputs_own_val_loss():
    tf.random.set_seed(0)
    model, trainer, _ = make()
    train, val, test = data(0, row_ids=True), data(1), data(2)
    model(next(iter(train))[0], training=False)
    model.attach_self_distill(OUTPUTS, N)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    scorer = EpochScorer(val, test, monitor="f1", patience=1, outputs=OUTPUTS, stops=False)
    scorer.trainer = trainer
    model.fit(train, epochs=1, validation_data=val, verbose=0, callbacks=[scorer])
    rec = scorer.records[0]
    # the model's own per-task loss on val arrays collected here (the formula
    # itself is checked in test_pooled_output_loss.py): labels, predictions
    # and output names must line up
    xs, ys = zip(*list(val))
    y = {k: tf.constant(np.concatenate([np.asarray(b[k]) for b in ys])) for k in NAMES}
    pred = {k: tf.constant(np.concatenate([np.asarray(model(x, training=False)[k]) for x in xs]))
            for k in NAMES}
    losses, has = model._mtl_task_losses_fn(y, pred)
    want = dict(zip(model._mtl_task_loss_names, losses.numpy()))
    assert set(want) == set(OUTPUTS)
    for k in OUTPUTS:
        np.testing.assert_allclose(rec["val"][k]["loss"], want[k], rtol=1e-4)
