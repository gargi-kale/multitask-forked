"""Checks for LabelAwareMaskedModel (famo_fixed plus the masked copy sent
through the whole model: supervised loss on it, and/or consistency with the
full-sentence answers) in test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_label_aware_masking.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, FAMO, BalancedMaskedLMModel, LabelAwareMaskedModel,
    PooledOutputLossTrainer, attach_mlm_balancing, confident_cells, load_pretrained_mlm_head,
)
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

M = -1.0
NAMES = ["p1", "p2", "wa", "d1", "d2"]
NUM_LOSSES = 4  # pooled propaganda, wa, d1, d2
UNLABELLED = list(range(28, 32))  # rows with no label at all
LAM = 0.03


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
                    ("d1", range(24, 28)), ("d2", range(24, 28))):
        y[k][list(rows), 0] = rng.integers(0, 2, size=len(rows))
    return x, {k: tf.constant(v) for k, v in y.items()}


def make_model(batch, head, label_w, cons_w, rampup=0, optimizer=None, mask_prob=0.15,
               keep=0.0, skip=None):
    """A built LabelAwareMaskedModel with encoder dropout off, so two training
    steps from the same weights can be compared exactly."""
    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )
    model = LabelAwareMaskedModel(cfg, dropout_rate=0.0)
    model(batch[0], training=False)
    for layer in model.encoder.submodules:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = 0.0
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 1.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    model.attach_mlm_head(head, LAM, mask_prob)
    attach_mlm_balancing(model, "famo", "fixed", NUM_LOSSES)
    model.attach_label_aware(label_w, cons_w, rampup, keep, skip)
    model.compile(optimizer=optimizer or tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    return model


def weights(model):
    return [v.numpy().copy() for v in model.trainable_variables]


def test_weights_zero_match_famo_fixed(batch, head):
    model = make_model(batch, head, 0.0, 0.0)
    w0 = weights(model)
    tf.random.set_seed(1)
    model.train_step(batch)
    ours = weights(model)

    for v, w in zip(model.trainable_variables, w0):
        v.assign(w)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    model._balancer = FAMO(NUM_LOSSES)
    tf.random.set_seed(1)  # the same masking
    BalancedMaskedLMModel.train_step(model, batch)
    for v, a, b in zip(model.trainable_variables, ours, weights(model)):
        np.testing.assert_allclose(a, b, atol=1e-6, err_msg=v.name)


def test_masked_label_loss_ignores_unlabelled_rows_and_consistency_uses_them(batch, head):
    model = make_model(batch, head, 1.0, 1.0)
    x, y = batch
    ids = x["input_ids"].numpy()
    changed = ids.copy()
    changed[UNLABELLED, 1:-1] = np.random.default_rng(5).integers(999, 30522, (4, ids.shape[1] - 2))
    x2 = {**x, "input_ids": tf.constant(changed)}

    def losses(inp):
        tf.random.set_seed(2)
        masking = model.mask_batch(inp)
        clean = model(inp, training=False)
        _, _, label, cons, _ = model.label_aware_losses(inp, y, clean, False, masking)
        return float(label), float(cons)

    (label_a, cons_a), (label_b, cons_b) = losses(x), losses(x2)
    np.testing.assert_allclose(label_a, label_b, rtol=1e-6)
    assert abs(cons_a - cons_b) > 1e-6


def test_consistency_target_gets_no_gradient(batch, head):
    model = make_model(batch, head, 0.0, 1.0)
    x, _ = batch
    clean = model(x, training=False)
    masked = model(x, training=False)
    keys = model._mtl_task_loss_names
    masked = {k: masked[k] * 0.9 + 0.05 for k in keys}  # differ from the targets
    targets = [clean[k] for k in keys]
    preds = [masked[k] for k in keys]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(targets + preds)
        loss, _ = model.consistency_loss(dict(zip(keys, targets)), dict(zip(keys, preds)))
    assert all(g is None for g in tape.gradient(loss, targets))
    assert all(g is not None and np.any(g.numpy()) for g in tape.gradient(loss, preds))


def test_nothing_hidden_gives_zero_consistency_and_the_plain_label_loss(batch, head):
    model = make_model(batch, head, 1.0, 1.0, mask_prob=0.0)
    x, y = batch
    clean = model(x, training=False)
    assert float(model.consistency_loss(clean, clean)[0]) == 0.0
    _, _, label, cons, kept = model.label_aware_losses(x, y, clean, False, model.mask_batch(x))
    np.testing.assert_allclose(float(cons), 0.0, atol=1e-6)
    assert float(kept) == 1.0  # no filter: every cell
    sup = model._mtl_loss_fn(y, {k: clean[k] for k in NAMES})
    np.testing.assert_allclose(float(label), float(sup), rtol=1e-5)


def test_consistency_ramp(batch, head):
    model = make_model(batch, head, 0.0, 1.0, rampup=4)
    got = []
    for step in range(6):
        model.optimizer.iterations.assign(step)
        got.append(float(model.consistency_ramp()))
    np.testing.assert_allclose(got, [0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    assert float(make_model(batch, head, 0.0, 1.0, rampup=0).consistency_ramp()) == 1.0


def test_step_is_the_weighted_sum_of_all_terms(batch, head):
    # With SGD at lr 1 the step is exactly -gradient of
    # FAMO objective + lambda x MLM + w_label x masked-label + w_cons x consistency.
    w_label, w_cons = 0.7, 0.4
    x, y = batch
    model = make_model(batch, head, w_label, w_cons, optimizer=tf.keras.optimizers.SGD(1.0))
    w0 = weights(model)
    variables = model.trainable_variables
    tf.random.set_seed(3)
    masking = model.mask_batch(x)
    with tf.GradientTape() as tape:
        clean = model(x, training=True)
        losses, has = model._mtl_task_losses_fn(y, {k: clean[k] for k in NAMES})
        mlm, _, label, cons, _ = model.label_aware_losses(x, y, clean, True, masking)
        obj = (model._balancer.objective(losses, has) + LAM * mlm
               + w_label * label + w_cons * cons)
    want = [tf.convert_to_tensor(g).numpy() for g in
            tape.gradient(obj, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)]
    tf.random.set_seed(3)  # the step draws the same masking
    model.train_step(batch)
    for v, w, g in zip(variables, w0, want):
        np.testing.assert_allclose(w - v.numpy(), g, atol=2e-5, rtol=1e-3, err_msg=v.name)


def test_fit_runs_in_graph_mode_and_logs(batch, head):
    model = make_model(batch, head, 0.5, 0.5, rampup=2)
    ds = tf.data.Dataset.from_tensors(batch).repeat(3)
    hist = model.fit(ds, epochs=2, validation_data=tf.data.Dataset.from_tensors(batch),
                     verbose=0).history
    keys = ["loss", "mlm_loss", "mlm_acc", "masked_label_loss", "consistency_loss",
            "consistency_weight", "consistency_kept", "val_loss", "val_mlm_acc", "val_masked_label_loss",
            "val_consistency_loss"]
    for k in keys:
        assert k in hist and len(hist[k]) == 2 and all(np.isfinite(hist[k])), (k, hist)
    # steps 0-2 then 3-5 with a 2-step ramp: means of (0, .5, 1) and (1, 1, 1)
    np.testing.assert_allclose(hist["consistency_weight"], [0.25, 0.5], rtol=1e-5)


def test_confident_cells_keep_both_ends_of_each_head():
    t = tf.constant([0.10, 0.90, 0.50, 0.45, 0.20, 0.80, 0.55, 0.60, 0.95, 0.05])
    every = tf.ones_like(t, dtype=tf.bool)
    keep = confident_cells(t, every, 0.2).numpy()  # 2 lowest and 2 highest
    assert keep.tolist() == [True, True, False, False, False, False, False, False, True, True]
    # Ineligible cells are never kept and do not count towards the ranking.
    eligible = tf.constant([True] * 5 + [False] * 5)
    keep = confident_cells(t, eligible, 0.2).numpy()  # 5 eligible -> 1 each side
    assert keep.tolist() == [True, True] + [False] * 8
    assert not confident_cells(t, tf.zeros_like(every), 0.2).numpy().any()


def test_confident_cells_floor_keeps_equal_sides_within_the_floor():
    t = tf.constant([0.05, 0.10, 0.15, 0.30, 0.50, 0.60, 0.70, 0.85, 0.95, 0.40])
    every = tf.ones_like(t, dtype=tf.bool)
    # share 0.3 -> 3 per side; floor 0.2 -> 3 low (<= 0.2) but 2 high (>= 0.8): 2 each
    keep = confident_cells(t, every, 0.3, floor=0.2).numpy()
    assert keep.tolist() == [True, True, False, False, False, False, False, True, True, False]
    # no high score reaches 0.9: the head keeps nothing
    assert not confident_cells(t, every, 0.3, floor=0.04).numpy().any()
    # no floor: unchanged, 3 per side
    assert confident_cells(t, every, 0.3).numpy().sum() == 6


def test_filter_uses_only_confident_cells_and_skips_marked_rows(batch, head):
    # d1 skips rows labelled for p1 (rows 0-15), as the IAC heads skip Propaganda rows.
    # Seeded: some inits left by earlier tests give many rows the same score, and
    # tied scores at a cut-off are all kept, which breaks the size check below.
    tf.random.set_seed(0)
    model = make_model(batch, head, 0.0, 1.0, keep=0.1, skip={"d1": ["p1"]})
    x, y = batch
    clean = model(x, training=False)
    keys = model._mtl_task_loss_names
    masked = {k: clean[k] * 0.5 + 0.25 for k in keys}  # every cell changes
    loss, kept = model.consistency_loss(clean, masked, y)

    n = 32
    eps = 1e-7
    total, used = 0.0, 0
    for k in keys:
        t = np.clip(clean[k].numpy().ravel(), eps, 1 - eps)
        p = np.clip(masked[k].numpy().ravel(), eps, 1 - eps)
        elig = np.ones(n, bool) if k != "d1" else np.arange(n) >= 16
        m = int(np.floor(0.1 * elig.sum()))
        order = np.sort(t[elig])
        keep = elig & ((t <= order[m - 1]) | (t >= order[::-1][m - 1]))
        kl = t * np.log(t / p) + (1 - t) * np.log((1 - t) / (1 - p))
        total += kl[keep].sum()
        used += keep.sum()
        if k == "d1":
            assert not keep[:16].any()
    np.testing.assert_allclose(float(loss), total / used, rtol=1e-4)
    np.testing.assert_allclose(float(kept), used / (n * len(keys)), rtol=1e-6)
    assert float(kept) < 0.3
