"""Checks for PooledOutputLossTrainer in test_no_prop_overlap.py.

    python -m pytest replication/test_pooled_output_loss.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import PooledOutputLossTrainer  # noqa: E402
from multitask import Trainer  # noqa: E402

M = -1.0
NAMES = ["p1", "p2", "p3", "wa", "d1", "d2"]
TYPES = {"p1": "prop", "p2": "prop", "p3": "prop", "wa": "wa", "d1": "disc", "d2": "disc"}
CW = [{0: 0.5, 1: 50.0}, {0: 0.6, 1: 3.0}, {0: 0.5, 1: 200.0},
      {0: 8.0, 1: 0.5}, {0: 2.0, 1: 0.6}, {0: 0.7, 1: 1.5}]
NU = [0.4, 0.4, 0.4, 1.1, 1.5, 1.5]
POOL_CW = {0: 0.8, 1: 1.4}

# rows 0-2 propaganda, 3-4 WA, 5-7 discourse (row 7 has d2 masked)
Y = {
    "p1": [1, 0, 0, M, M, M, M, M],
    "p2": [0, 0, 1, M, M, M, M, M],
    "p3": [0, 0, 0, M, M, M, M, M],
    "wa": [M, M, M, 1, 0, M, M, M],
    "d1": [M, M, M, M, M, 1, 0, 1],
    "d2": [M, M, M, M, M, 0, 1, M],
}
rng = np.random.default_rng(0)
P = {n: rng.uniform(0.05, 0.95, size=8).astype(np.float32) for n in NAMES}


def stub_model():
    cfg = SimpleNamespace(mask_value=M, get_task_type=lambda i: "binary", total_outputs=len(NAMES))
    return SimpleNamespace(task_names=NAMES, config=cfg)


def tensors(d):
    return {k: tf.constant(np.asarray(v, np.float32).reshape(-1, 1)) for k, v in d.items()}


def build(trainer_cls, **kw):
    t = trainer_cls(stub_model(), **kw)
    return t._build_loss_function(len(NAMES), CW, NU, task_type_mapping=TYPES)


def test_single_head_pool_matches_stock_library_loss():
    # "wa" is alone in its task-type, so pooling just that one head must
    # reproduce Trainer._build_loss_function exactly.
    stock = build(Trainer)(tensors(Y), tensors(P))
    pooled = build(PooledOutputLossTrainer, pooled_tasks=["wa"],
                   pooled_class_weights=CW[3])(tensors(Y), tensors(P))
    np.testing.assert_allclose(pooled.numpy(), stock.numpy(), rtol=1e-6)


def test_pooled_loss_matches_formula():
    loss = build(PooledOutputLossTrainer, pooled_tasks=["p1", "p2", "p3"],
                 pooled_class_weights=POOL_CW)(tensors(Y), tensors(P))

    def bce(y, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    total, n = 0.0, 0
    # non-pooled tasks: class weight * nu_k / |T_k| (|wa|=1, |disc|=2)
    for i, name in enumerate(NAMES[3:], start=3):
        size = 1 if TYPES[name] == "wa" else 2
        for y, p in zip(Y[name], P[name]):
            if y != M:
                total += bce(y, p) * CW[i][int(y)] * NU[i] / size
                n += 1
    # pooled propaganda: one task, nu_prop, pooled class weights
    for r in range(8):
        ys = [Y[t][r] for t in ("p1", "p2", "p3")]
        if all(y == M for y in ys):
            continue
        y, p = max(ys), max(P[t][r] for t in ("p1", "p2", "p3"))
        total += bce(y, p) * POOL_CW[int(y)] * NU[0]
        n += 1
    np.testing.assert_allclose(loss.numpy(), total / n, rtol=1e-5)


def test_gradient_reaches_only_max_head_per_row():
    loss_fn = build(PooledOutputLossTrainer, pooled_tasks=["p1", "p2", "p3"],
                    pooled_class_weights=POOL_CW)
    preds = {k: tf.Variable(v) for k, v in tensors(P).items()}
    with tf.GradientTape() as tape:
        loss = loss_fn(tensors(Y), preds)
    grads = tape.gradient(loss, [preds[t] for t in ("p1", "p2", "p3")])
    g = np.concatenate([gr.numpy() for gr in grads], axis=1)  # (rows, 3)
    stacked = np.stack([P[t] for t in ("p1", "p2", "p3")], axis=1)
    for r in range(3):  # propaganda rows
        nz = np.flatnonzero(g[r])
        assert list(nz) == [int(np.argmax(stacked[r]))]
    assert np.all(g[3:] == 0)  # rows without propaganda labels
