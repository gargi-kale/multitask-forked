"""Checks for BalancedMaskedLMModel (masked-word loss with FAMO / DB-MTL or a
gradient-similarity gate) in test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_mlm_balancing.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    DBMTL, ENCODER_CONFIGS, FAMO, BalancedMaskedLMModel, BalancedMultiTaskModel,
    PooledOutputLossTrainer, attach_mlm_balancing, load_pretrained_mlm_head, shared_variables,
)
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

M = -1.0
NAMES = ["p1", "p2", "wa", "d1", "d2"]
NUM_LOSSES = 4  # pooled propaganda, wa, d1, d2


@pytest.fixture(scope="module")
def head():
    return load_pretrained_mlm_head("google/bert_uncased_L-2_H-128_A-2", from_pt=True)


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    n, length = 32, 24
    ids = rng.integers(999, 30522, size=(n, length)).astype(np.int32)
    ids[:, 0], ids[:, -1] = 101, 102
    x = {"input_ids": tf.constant(ids), "attention_mask": tf.ones_like(tf.constant(ids))}
    y = {k: np.full((n, 1), M, np.float32) for k in NAMES}
    for k, rows in (("p1", range(16)), ("p2", range(16)), ("wa", range(16, 24)),
                    ("d1", range(24, 32)), ("d2", range(24, 32))):
        y[k][list(rows), 0] = rng.integers(0, 2, size=len(rows))
    return x, {k: tf.constant(v) for k, v in y.items()}


def make_model(batch, head, balancer, mode, lam, optimizer=None):
    """A built BalancedMaskedLMModel with encoder dropout off, so two training
    steps from the same weights can be compared exactly."""
    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )
    model = BalancedMaskedLMModel(cfg, dropout_rate=0.0)
    model(batch[0], training=False)
    for layer in model.encoder.submodules:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = 0.0
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 1.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    model.attach_mlm_head(head, lam)
    attach_mlm_balancing(model, balancer, mode, NUM_LOSSES)
    model.compile(optimizer=optimizer or tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    return model


def weights(model):
    return [v.numpy().copy() for v in model.trainable_variables]


def test_shared_variables_exclude_the_masked_word_head(batch, head):
    model = make_model(batch, head, "dbmtl", "fixed", 0.1)
    shared = {id(v) for v in shared_variables(model)}
    assert not {id(v) for v in model.mlm_head.trainable_variables} & shared
    assert {id(v) for v in model.encoder.trainable_variables} <= shared
    assert {id(v) for v in model._balancer.shared_vars} == shared


@pytest.mark.parametrize("balancer", ["famo", "dbmtl"])
def test_fixed_lambda_zero_matches_plain_balancer(batch, head, balancer):
    model = make_model(batch, head, balancer, "fixed", 0.0)
    w0 = weights(model)
    model.train_step(batch)
    ours = weights(model)

    for v, w in zip(model.trainable_variables, w0):
        v.assign(w)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    model._balancer = (FAMO(NUM_LOSSES) if balancer == "famo"
                       else DBMTL(shared_variables(model), NUM_LOSSES))
    BalancedMultiTaskModel.train_step(model, batch)
    for v, a, b in zip(model.trainable_variables, ours, weights(model)):
        np.testing.assert_allclose(a, b, atol=1e-6, err_msg=v.name)
    head_ids = {id(v) for v in model.mlm_head.trainable_variables}
    for v, a, b in zip(model.trainable_variables, w0, ours):
        if id(v) in head_ids:  # lambda 0: the head never moves
            np.testing.assert_array_equal(a, b, err_msg=v.name)


def test_task_mode_gives_the_masked_word_loss_its_own_weight(batch, head):
    tf.random.set_seed(0)
    model = make_model(batch, head, "famo", "task", 0.0)
    assert model._balancer.w.shape == [NUM_LOSSES + 1]
    w0 = weights(model)
    logs = model.train_step(batch)
    z = tf.nn.softmax(model._balancer.w).numpy()
    assert not np.allclose(z, z[0])  # the first update moves the weights
    np.testing.assert_allclose(float(logs["mlm_weight"]), z[-1], rtol=1e-6)
    # lambda is unused in task mode, so the head trains even with lambda 0
    head_ids = {id(v) for v in model.mlm_head.trainable_variables}
    assert any(not np.array_equal(a, v.numpy())
               for v, a in zip(model.trainable_variables, w0) if id(v) in head_ids)

    model = make_model(batch, head, "dbmtl", "task", 0.0)
    model.train_step(batch)
    buf = model._balancer.buffer.numpy()
    assert buf.shape[0] == NUM_LOSSES + 1 and np.any(buf[-1])


def test_cosine_gate_update(batch, head):
    # With SGD at lr 1 the step is exactly -gradient: shared parameters get
    # supervised + lambda x gate x masked-word, the head lambda x masked-word,
    # the task layers supervised only.
    lam = 0.5
    x, y = batch
    model = make_model(batch, head, "", "cosine", lam, tf.keras.optimizers.SGD(1.0))
    w0 = weights(model)
    tf.random.set_seed(3)
    masking = model.mask_batch(x)
    variables = model.trainable_variables
    zero = tf.UnconnectedGradients.ZERO
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x, training=True)
        sup = model._mtl_loss_fn(y, {k: y_pred[k] for k in model.task_names})
        mlm, _ = model.mlm_loss(x, True, masking)
    g_sup = [tf.convert_to_tensor(g).numpy()
             for g in tape.gradient(sup, variables, unconnected_gradients=zero)]
    g_mlm = [tf.convert_to_tensor(g).numpy()
             for g in tape.gradient(mlm, variables, unconnected_gradients=zero)]
    shared = {id(v) for v in model._mlm_shared}
    head_ids = {id(v) for v in model.mlm_head.trainable_variables}
    is_shared = [id(v) in shared for v in variables]
    a = np.concatenate([g.ravel() for g, s in zip(g_sup, is_shared) if s])
    b = np.concatenate([g.ravel() for g, s in zip(g_mlm, is_shared) if s])
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    tf.random.set_seed(3)  # the step draws the same masking
    logs = model.train_step(batch)
    np.testing.assert_allclose(float(logs["mlm_cos"]), cos, rtol=1e-4, atol=1e-6)
    assert float(logs["mlm_gate"]) == float(cos > 0)
    gate = float(cos > 0)
    for v, w, gs, gm, s in zip(variables, w0, g_sup, g_mlm, is_shared):
        if s:
            want = gs + lam * gate * gm
        elif id(v) in head_ids:
            want = lam * gm
        else:
            want = gs
        np.testing.assert_allclose(w - v.numpy(), want, atol=2e-5, rtol=1e-3, err_msg=v.name)


@pytest.mark.parametrize("balancer,mode", [("famo", "fixed"), ("dbmtl", "fixed"),
                                           ("famo", "task"), ("dbmtl", "task"), ("", "cosine")])
def test_fit_runs_in_graph_mode_and_logs(batch, head, balancer, mode):
    model = make_model(batch, head, balancer, mode, 0.03)
    ds = tf.data.Dataset.from_tensors(batch).repeat(3)
    hist = model.fit(ds, epochs=2, validation_data=tf.data.Dataset.from_tensors(batch),
                     verbose=0).history
    keys = ["loss", "mlm_loss", "mlm_acc", "val_loss", "val_mlm_acc"]
    keys += list(model.mlm_extra_trackers)
    for k in keys:
        assert k in hist and len(hist[k]) == 2 and all(np.isfinite(hist[k])), (k, hist)
    if mode == "cosine":
        assert all(0.0 <= g <= 1.0 for g in hist["mlm_gate"])
