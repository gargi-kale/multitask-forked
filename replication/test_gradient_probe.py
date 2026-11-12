"""Checks for GradientProbe and the shared/task parameter split in
test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_gradient_probe.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, GradientProbe, PooledOutputLossTrainer, shared_variables,
)
from multitask import MultiTaskModel  # noqa: E402
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

M = -1.0
NAMES = ["p1", "p2", "wa", "d1", "d2"]


def data(n=64, length=16, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(999, 30522, size=(n, length)).astype(np.int32)
    ids[:, 0], ids[:, -1] = 101, 102
    x = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    # rows 0-31 propaganda, 32-47 WA, 48-63 discourse, as in the real CSVs
    y = {k: np.full((n, 1), M, np.float32) for k in NAMES}
    for k, rows in (("p1", range(32)), ("p2", range(32)), ("wa", range(32, 48)),
                    ("d1", range(48, 64)), ("d2", range(48, 64))):
        y[k][list(rows), 0] = rng.integers(0, 2, size=len(rows))
    return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(n, seed=seed).batch(16)


def make_model():
    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )
    model = MultiTaskModel(cfg, dropout_rate=0.0)
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 1.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    model._mtl_loss_fn = loss_fn
    return model, loss_fn


def test_shared_variables_are_encoder_and_shared_layers():
    model, _ = make_model()
    ds = data()
    model(next(iter(ds))[0], training=False)
    shared = {id(v) for v in shared_variables(model)}
    assert {id(v) for v in model.encoder.trainable_variables} <= shared
    assert {id(v) for l in model.shared_layers for v in l.trainable_variables} <= shared
    for layer in model.output_layers.values():
        assert not {id(v) for v in layer.trainable_variables} & shared


def test_probe_records_valid_metrics():
    tf.random.set_seed(0)
    model, loss_fn = make_model()
    ds = data()
    model(next(iter(ds))[0], training=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
    probe = GradientProbe(ds, every=2, min_labels=16, lookahead_lr=1e-4)
    model.fit(ds, epochs=2, verbose=0, callbacks=[probe])

    assert probe.task_names == ["propaganda_max_pool", "wa", "d1", "d2"]
    # one before training, then every 2 of the 4 steps per epoch x 2 epochs
    assert [r["step"] for r in probe.records] == [0, 2, 4, 6, 8]
    assert [r["epoch"] for r in probe.records] == [0, 0, 0, 1, 1]
    for r in probe.records:
        cos, conf = np.array(r["cosine"]), np.array(r["conflict_rate"])
        assert cos.shape == (4, 4) and conf.shape == (4, 4)
        np.testing.assert_allclose(np.diag(cos), 1.0, atol=1e-5)
        np.testing.assert_allclose(cos, cos.T, atol=1e-5)
        assert np.all(np.abs(cos) <= 1 + 1e-5) and np.all((conf >= 0) & (conf <= 1))
        assert np.all(np.diag(conf) == 0) and all(n > 0 for n in r["norm"])
        aff = np.array(r["affinity"])
        assert aff.shape == (4, 4) and np.all(np.isfinite(aff)) and np.all(aff < 1)
    # training moves the shared weights, so the probe changes between records
    assert probe.records[0]["cosine"] != probe.records[-1]["cosine"]


def test_probe_builds_unbuilt_model():
    # The real pipeline calls fit on a model that was never called.
    model, loss_fn = make_model()
    ds = data()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
    probe = GradientProbe(ds, every=1000, min_labels=16)
    model.fit(ds.take(1), epochs=1, verbose=0, callbacks=[probe])
    assert len(probe.records) == 1 and all(n > 0 for n in probe.records[0]["norm"])


def test_affinity_step_is_undone():
    model, loss_fn = make_model()
    ds = data()
    model(next(iter(ds))[0], training=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0), loss=loss_fn)
    probe = GradientProbe(ds, every=1000, min_labels=16, lookahead_lr=1e-2)
    before = [v.numpy().copy() for v in model.trainable_variables]
    model.fit(ds.take(1), epochs=1, verbose=0, callbacks=[probe])  # lr 0: only the probe acts
    for b, v in zip(before, model.trainable_variables):
        np.testing.assert_array_equal(b, v.numpy(), err_msg=v.name)


def test_small_step_affinity_matches_first_order():
    # For a tiny step of length s along unit(g_i): 1 - L_j(after)/L_j
    # ~= s * |g_j| * cos(i, j) / L_j, and the diagonal is positive. Only
    # approximately: the max pool and ReLUs are not smooth.
    tf.random.set_seed(0)
    model, loss_fn = make_model()
    ds = data()
    model(next(iter(ds))[0], training=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0), loss=loss_fn)
    probe = GradientProbe(ds, every=1000, min_labels=16, lookahead_lr=1e-6)
    model.fit(ds.take(1), epochs=1, verbose=0, callbacks=[probe])
    r = probe.records[0]
    length = 1e-6 * np.sqrt(sum(int(np.prod(v.shape)) for v in probe.shared))
    pred = length * np.array(r["cosine"]) * np.array(r["norm"])[None, :] / np.array(r["loss"])[None, :]
    aff = np.array(r["affinity"])
    assert np.all(np.diag(aff) > 0), aff
    assert np.corrcoef(aff.ravel(), pred.ravel())[0, 1] > 0.95, (aff, pred)
    big = np.abs(pred) > 0.1 * np.abs(pred).max()
    np.testing.assert_array_equal(np.sign(aff[big]), np.sign(pred[big]))
