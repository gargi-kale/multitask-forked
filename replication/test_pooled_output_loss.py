"""Checks for PooledOutputLossTrainer in test_no_prop_overlap.py.

    python -m pytest replication/test_pooled_output_loss.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
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


def _bce_from_logit(y, z):
    p = 1 / (1 + np.exp(-z))
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), p


def test_zero_shift_and_gamma_leave_loss_unchanged():
    base = build(PooledOutputLossTrainer, pooled_tasks=["p1", "p2", "p3"],
                 pooled_class_weights=POOL_CW)(tensors(Y), tensors(P))
    opts = build(PooledOutputLossTrainer, pooled_tasks=["p1", "p2", "p3"],
                 pooled_class_weights=POOL_CW,
                 task_loss_options={"d1": (0.0, 0.0)})(tensors(Y), tensors(P))
    np.testing.assert_allclose(opts.numpy(), base.numpy(), rtol=1e-6)


def test_logit_adjust_and_focal_match_formula():
    shift, gamma = 1.3, 2.0
    kw = dict(pooled_tasks=["p1", "p2", "p3"], pooled_class_weights=POOL_CW)
    base = build(PooledOutputLossTrainer, **kw)(tensors(Y), tensors(P))
    loss = build(PooledOutputLossTrainer, task_loss_options={"d1": (shift, gamma)},
                 **kw)(tensors(Y), tensors(P))
    # Only the d1 term changes; n_valid is identical, so the difference in
    # summed loss is (new d1 sum - old d1 sum) / n.
    n = sum(1 for name in NAMES[3:] for y in Y[name] if y != M) + 3
    old = new = 0.0
    for y, p in zip(Y["d1"], P["d1"]):
        if y == M:
            continue
        w = CW[4][int(y)] * NU[4] / 2
        z = np.log(p / (1 - p))
        old += _bce_from_logit(y, z)[0] * w
        b, p_adj = _bce_from_logit(y, z + shift)
        p_correct = p_adj if y == 1 else 1 - p_adj
        new += b * (1 - p_correct) ** gamma * w
    np.testing.assert_allclose(loss.numpy() - base.numpy(), (new - old) / n, rtol=1e-4)


def test_famo_first_update_moves_weight_off_slow_tasks():
    from test_no_prop_overlap import FAMO
    famo = FAMO(3)
    prev = tf.constant([1.0, 1.0, 1.0])
    new = tf.constant([0.5, 0.9, 1.0])  # task 0 improved most, task 2 not at all
    famo.update(prev, new, tf.ones(3))
    delta = np.log(prev.numpy()) - np.log(new.numpy())
    g = (delta - delta.mean()) / 3  # z is uniform; w starts at 0, so no decay term
    # Adam's first step is lr * sign(g).
    np.testing.assert_allclose(famo.w.numpy(), -0.025 * np.sign(g), atol=1e-6)
    # Tasks 1 and 2 both improved less than average, so both gain equally.
    z = tf.nn.softmax(famo.w).numpy()
    assert z[2] == z[1] > z[0]


def test_dbmtl_rescales_each_task_gradient_to_the_largest():
    from test_no_prop_overlap import DBMTL
    v = tf.Variable([1.0, 2.0])
    db = DBMTL([v], num_tasks=2, beta=0.9)
    with tf.GradientTape(persistent=True) as tape:
        log_losses = [3.0 * v[0], 0.5 * v[1]]  # gradients (3, 0) and (0, 0.5)
    (g,) = db.shared_gradients(tape, log_losses, tf.ones(2))
    # EMA from zero is 0.1 * grad; both are scaled to the larger norm (0.3).
    np.testing.assert_allclose(g.numpy(), [0.3, 0.3], rtol=1e-5)
    np.testing.assert_allclose(db.buffer.numpy(), [[0.3, 0.0], [0.0, 0.05]], rtol=1e-5)


# ---------------------------------------------------------------------------
# Propaganda pooling options: noisy-OR, log-sum-exp, per-technique loss
# ---------------------------------------------------------------------------

PROP = ("p1", "p2", "p3")
POOL_KW = dict(pooled_tasks=list(PROP), pooled_class_weights=POOL_CW)


def _bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _pooled_part(pool_fn, extra=lambda r: 0.0):
    """(sum, rows) of the pooled propaganda term for a numpy pool function."""
    total, n = 0.0, 0
    for r in range(8):
        ys = [Y[t][r] for t in PROP]
        if all(y == M for y in ys):
            continue
        ps = np.array([P[t][r] for t in PROP], np.float64)
        y = max(ys)
        total += (_bce(y, pool_fn(ps)) * POOL_CW[int(y)] + extra(r)) * NU[0]
        n += 1
    return total, n


def _other_part():
    total, n = 0.0, 0
    for i, name in enumerate(NAMES[3:], start=3):
        size = 1 if TYPES[name] == "wa" else 2
        for y, p in zip(Y[name], P[name]):
            if y != M:
                total += _bce(y, p) * CW[i][int(y)] * NU[i] / size
                n += 1
    return total, n


def _noisy_or(ps):
    return 1 - np.prod(1 - ps)


def _lse(ps):
    z = np.log(ps / (1 - ps))
    return np.exp(z).sum() / (1 + np.exp(z).sum())


def test_pool_probabilities_values():
    from multitask.models.hierarchical import pool_probabilities
    ps = np.array([[0.1, 0.5, 0.2], [0.9, 0.9, 0.9]], np.float32)
    got = {s: pool_probabilities(tf.constant(ps), s, axis=1).numpy()
           for s in ("max", "noisy_or", "lse")}
    np.testing.assert_allclose(got["max"], ps.max(1))
    np.testing.assert_allclose(got["noisy_or"], [_noisy_or(r) for r in ps.astype(np.float64)], rtol=1e-5)
    np.testing.assert_allclose(got["lse"], [_lse(r) for r in ps.astype(np.float64)], rtol=1e-5)
    # Both soft pools sit at or above the max.
    assert np.all(got["noisy_or"] >= got["max"] - 1e-6) and np.all(got["lse"] >= got["max"] - 1e-6)


def test_pool_probabilities_stays_finite_at_saturation():
    from multitask.models.hierarchical import pool_probabilities
    ps = tf.constant([[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]], tf.float32)
    for s in ("noisy_or", "lse"):
        out = pool_probabilities(ps, s, axis=1).numpy()
        assert np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1))


@pytest.mark.parametrize("strategy,fn", [("noisy_or", _noisy_or), ("lse", _lse)])
def test_soft_pooled_loss_matches_formula(strategy, fn):
    loss = build(PooledOutputLossTrainer, pool_strategy=strategy, **POOL_KW)(tensors(Y), tensors(P))
    a, na = _other_part()
    b, nb = _pooled_part(fn)
    np.testing.assert_allclose(loss.numpy(), (a + b) / (na + nb), rtol=1e-5)


@pytest.mark.parametrize("strategy", ["noisy_or", "lse"])
def test_soft_pool_sends_gradient_to_every_head(strategy):
    loss_fn = build(PooledOutputLossTrainer, pool_strategy=strategy, **POOL_KW)
    preds = {k: tf.Variable(v) for k, v in tensors(P).items()}
    with tf.GradientTape() as tape:
        loss = loss_fn(tensors(Y), preds)
    g = np.concatenate([gr.numpy() for gr in tape.gradient(loss, [preds[t] for t in PROP])], axis=1)
    assert np.all(g[:3] != 0)   # every head, every propaganda row
    assert np.all(g[3:] == 0)   # rows without propaganda labels


def _technique_mean(r_rows=range(8)):
    """Mean over heads of each head's weighted-mean BCE over its labelled rows."""
    per_head = []
    for i, t in enumerate(PROP):
        num = den = 0.0
        for r in r_rows:
            y = Y[t][r]
            if y == M:
                continue
            w = CW[i][int(y)]
            num += w * _bce(y, P[t][r])
            den += w
        if den:
            per_head.append(num / den)
    return float(np.mean(per_head))


def test_technique_loss_matches_formula():
    lam = 0.3
    loss = build(PooledOutputLossTrainer, technique_loss_weight=lam,
                 technique_class_weights=CW[:3], **POOL_KW)(tensors(Y), tensors(P))
    tech = _technique_mean()
    a, na = _other_part()
    b, nb = _pooled_part(np.max, extra=lambda r: lam * tech)
    np.testing.assert_allclose(loss.numpy(), (a + b) / (na + nb), rtol=1e-5)


def test_zero_technique_weight_leaves_loss_unchanged():
    base = build(PooledOutputLossTrainer, **POOL_KW)(tensors(Y), tensors(P))
    zero = build(PooledOutputLossTrainer, technique_loss_weight=0.0,
                 technique_class_weights=CW[:3], **POOL_KW)(tensors(Y), tensors(P))
    np.testing.assert_allclose(zero.numpy(), base.numpy(), rtol=1e-6)


def test_technique_loss_trains_every_head_with_max_pool():
    loss_fn = build(PooledOutputLossTrainer, technique_loss_weight=0.1,
                    technique_class_weights=CW[:3], **POOL_KW)
    preds = {k: tf.Variable(v) for k, v in tensors(P).items()}
    with tf.GradientTape() as tape:
        loss = loss_fn(tensors(Y), preds)
    g = np.concatenate([gr.numpy() for gr in tape.gradient(loss, [preds[t] for t in PROP])], axis=1)
    assert np.all(g[:3] != 0) and np.all(g[3:] == 0)


def test_task_losses_put_technique_term_in_the_propaganda_task():
    lam = 0.3
    t = PooledOutputLossTrainer(stub_model(), technique_loss_weight=lam,
                                technique_class_weights=CW[:3], **POOL_KW)
    t._build_loss_function(len(NAMES), CW, NU, task_type_mapping=TYPES)
    losses, has = t.model._mtl_task_losses_fn(tensors(Y), tensors(P))
    b, nb = _pooled_part(np.max)
    expected0 = b / NU[0] / nb + lam * _technique_mean()
    np.testing.assert_allclose(losses.numpy()[0], expected0, rtol=1e-5)
    assert len(losses.numpy()) == 4  # pooled propaganda + wa + d1 + d2


def test_bad_pool_options_are_rejected():
    with pytest.raises(ValueError):
        PooledOutputLossTrainer(stub_model(), pool_strategy="mean", **POOL_KW)
    with pytest.raises(ValueError):
        PooledOutputLossTrainer(stub_model(), technique_loss_weight=0.1, **POOL_KW)


def test_soft_pool_labels_stay_any_technique():
    """The library's pooled labels for noisy_or / lse are the max of the
    constituent labels, never a pooled score of the labels."""
    from multitask.trainers.trainer import Trainer as LibTrainer
    labels = {t: np.asarray(Y[t], np.float32) for t in PROP}
    for strategy in ("max", "noisy_or", "lse"):
        cfg = SimpleNamespace(has_pools=True, pool_names=["pp"],
                              get_pool_config=lambda n, s=strategy: {"strategy": s, "tasks": list(PROP)})
        t = LibTrainer.__new__(LibTrainer)
        t.model, t.mask_value = SimpleNamespace(config=cfg), M
        np.testing.assert_array_equal(t._derive_pooled_labels(labels)["pp"],
                                      [1, 0, 1, M, M, M, M, M])
