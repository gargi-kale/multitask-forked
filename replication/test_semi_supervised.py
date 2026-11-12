"""Checks for the semi-supervised terms (target rate + entropy on foreign
cells) and SemiSupervisedModel / SemiSupervisedTrainer in test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_semi_supervised.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, SSL_DEFAULT_RATES, SSL_POOL, MultiTaskModel, PooledOutputLossTrainer,
    SemiSupervisedModel, SemiSupervisedTrainer, parse_ssl_rates, semi_supervised_terms,
)
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

M = -1.0
NAMES = ["p1", "p2", "wa", "d1", "d2"]
POOLED = ["p1", "p2"]
HOME = {"Propaganda": POOLED, "IBM": ["wa"]}
# Rows 0-15 Propaganda (rows 0-3 also carry a rule-made d1 label), 16-23 IBM,
# 24-27 IAC with d labels, 28-31 IAC with no label at all.
PROP, IBM, IAC = range(0, 16), range(16, 24), range(24, 32)
PAIRS = [(SSL_POOL, "IAC", 0.3), (SSL_POOL, "IBM", 0.1),
         ("d1", "Propaganda", 0.1), ("d1", "IBM", 0.05)]


def labels(n=32, seed=0):
    rng = np.random.default_rng(seed)
    y = {k: np.full((n, 1), M, np.float32) for k in NAMES}
    for k, rows in (("p1", PROP), ("p2", PROP), ("wa", IBM), ("d1", range(24, 28)),
                    ("d2", range(24, 28))):
        y[k][list(rows), 0] = rng.integers(0, 2, size=len(rows))
    y["d1"][:4, 0] = 1.0
    return y


def foreign(y):
    """Expected foreign-cell masks per pair, written out by hand."""
    n = len(y["p1"])
    rows = lambda r: np.isin(np.arange(n), list(r))
    d1_missing = y["d1"][:, 0] == M
    return [rows(IAC), rows(IBM), rows(PROP) & d1_missing, rows(IBM) & d1_missing]


def terms(y, pred, pairs=PAIRS):
    return semi_supervised_terms({k: tf.constant(v) for k, v in y.items()}, pred, pairs,
                                 POOLED, HOME, "IAC", M)


def random_pred(n=32, seed=1):
    rng = np.random.default_rng(seed)
    return {k: tf.Variable(rng.uniform(0.05, 0.95, size=(n, 1)).astype(np.float32))
            for k in NAMES}


def test_parse_rates_and_defaults():
    pairs = parse_ssl_rates(SSL_DEFAULT_RATES)
    assert (SSL_POOL, "IAC", 0.30) in pairs and len(pairs) == 8
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_ssl_rates("d1@IAC=0")
    with pytest.raises(ValueError, match="only once"):
        parse_ssl_rates("d1@IAC=0.1,d1@IAC=0.2")
    with pytest.raises(ValueError, match="head@source=rate"):
        parse_ssl_rates("d1=0.1")


def test_extra_loss_is_zero_without_unlabelled_rows():
    y = {k: np.ones((32, 1), np.float32) for k in NAMES}  # every cell labelled
    rate, entropy, means, counts = terms(y, random_pred())
    assert float(rate) == 0.0 and float(entropy) == 0.0
    np.testing.assert_array_equal(counts.numpy(), 0.0)
    np.testing.assert_array_equal(means.numpy(), 0.0)


def test_sources_and_foreign_cells_are_read_from_the_labels():
    y = labels()
    _, _, _, counts = terms(y, random_pred())
    # Rule-labelled d1 rows (0-3) are not foreign for d1@Propaganda.
    np.testing.assert_array_equal(counts.numpy(), [8, 8, 12, 8])
    np.testing.assert_array_equal(counts.numpy(), [m.sum() for m in foreign(y)])


def test_rate_and_entropy_match_hand_computation():
    y, pred = labels(), random_pred()
    rate, entropy, means, _ = terms(y, pred)
    p = {k: v.numpy()[:, 0].astype(np.float64) for k, v in pred.items()}
    p[SSL_POOL] = np.maximum(p["p1"], p["p2"])
    kls, ents, ms = [], [], []
    for (head, _, r), m in zip(PAIRS, foreign(y)):
        q = p[head][m].mean()
        kls.append(r * np.log(r / q) + (1 - r) * np.log((1 - r) / (1 - q)))
        h = -(p[head] * np.log(p[head]) + (1 - p[head]) * np.log(1 - p[head]))
        ents.append(h[m].mean())
        ms.append(q)
    np.testing.assert_allclose(means.numpy(), ms, rtol=1e-5)
    np.testing.assert_allclose(float(rate), np.mean(kls), rtol=1e-4)
    np.testing.assert_allclose(float(entropy), np.mean(ents), rtol=1e-4)


def test_rate_is_zero_when_the_mean_hits_the_target():
    y = labels()
    pred = {k: tf.fill([32, 1], 0.3) for k in NAMES}
    rate, _, _, _ = terms(y, pred, pairs=[(SSL_POOL, "IAC", 0.3), ("d1", "IBM", 0.3)])
    assert abs(float(rate)) < 1e-6


def test_only_unlabelled_rows_get_gradient():
    y, pred = labels(), random_pred()
    with tf.GradientTape() as tape:
        rate, entropy, _, _ = terms(y, pred)
        total = rate + entropy
    grads = dict(zip(NAMES, tape.gradient(total, [pred[k] for k in NAMES])))
    m_pool_iac, m_pool_ibm, m_d1_prop, m_d1_ibm = foreign(y)
    # d1: gradient exactly on its foreign cells (Propaganda rows without the
    # rule label, IBM rows); none on labelled rows or on IAC rows.
    g = grads["d1"].numpy()[:, 0]
    assert np.all(g[m_d1_prop | m_d1_ibm] != 0)
    assert np.all(g[~(m_d1_prop | m_d1_ibm)] == 0)
    # The pool reaches p1/p2 only on IAC and IBM rows (never on the labelled
    # Propaganda rows), and through the max only the larger head of each row.
    gp = np.stack([grads["p1"].numpy()[:, 0], grads["p2"].numpy()[:, 0]], axis=1)
    pool_rows = m_pool_iac | m_pool_ibm
    assert np.all(gp[~pool_rows] == 0)
    assert np.all((gp[pool_rows] != 0).sum(axis=1) == 1)
    # Heads with no pair get nothing.
    for k in ("wa", "d2"):
        assert grads[k] is None or np.all(grads[k].numpy() == 0)


# --- the model and trainer ---------------------------------------------------

@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    ids = rng.integers(999, 30522, size=(32, 24)).astype(np.int32)
    ids[:, 0], ids[:, -1] = 101, 102
    x = {"input_ids": tf.constant(ids), "attention_mask": tf.ones_like(tf.constant(ids))}
    return x, {k: tf.constant(v) for k, v in labels().items()}


def config():
    return ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={SSL_POOL: {"tasks": POOLED, "strategy": "max",
                                 "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )


def make_model(batch, rate_w=1.0, entropy_w=0.0, start=0, lr=1e-3):
    model = SemiSupervisedModel(config(), dropout_rate=0.0)
    model(batch[0], training=False)
    for layer in model.encoder.submodules:  # deterministic passes
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = 0.0
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=POOLED,
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 1.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=model._mtl_loss_fn)
    model.attach_semi_supervised(PAIRS, POOLED, HOME, "IAC", rate_w, entropy_w, start)
    return model


def test_train_step_reports_the_supervised_loss_and_logs_the_terms(batch):
    model = make_model(batch, rate_w=1.0, entropy_w=0.5)
    x, y = batch
    pred = model(x, training=True)
    supervised = float(model._mtl_loss_fn(y, {k: pred[k] for k in NAMES}))
    rate, entropy, means, _ = model.ssl_terms(y, {k: pred[k] for k in NAMES})
    logs = model.train_step(batch)
    np.testing.assert_allclose(float(logs["loss"]), supervised, rtol=1e-5)
    np.testing.assert_allclose(float(logs["ssl_rate_loss"]), float(rate), rtol=1e-5)
    np.testing.assert_allclose(float(logs["ssl_entropy_loss"]), float(entropy), rtol=1e-5)
    np.testing.assert_allclose(float(logs["ssl_mean_propaganda_IAC"]), float(means[0]), rtol=1e-5)
    assert "ssl_mean_d1_Propaganda" in logs


def test_entropy_term_waits_for_its_start_step(batch):
    model = make_model(batch, rate_w=0.1, entropy_w=0.5, start=2)
    seen = [float(model.train_step(batch)["ssl_entropy_weight"]) for _ in range(3)]
    # Trackers average over the epoch: steps 0 and 1 log 0, step 2 logs 0.5.
    assert seen[0] == 0.0 and seen[1] == 0.0
    np.testing.assert_allclose(seen[2], 0.5 / 3, rtol=1e-6)


def test_extra_terms_change_the_step_only_through_foreign_cells(batch):
    """With every cell labelled, the extra terms are 0, so a step matches a
    plain MultiTaskModel step from the same weights."""
    x, _ = batch
    y = {k: tf.ones((32, 1)) for k in NAMES}
    model = make_model(batch, rate_w=1.0, entropy_w=1.0)
    plain = MultiTaskModel(config(), dropout_rate=0.0)
    plain(x, training=False)
    for layer in plain.encoder.submodules:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = 0.0
    assert len(plain.trainable_variables) == len(model.trainable_variables)
    for p, m in zip(plain.trainable_variables, model.trainable_variables):
        p.assign(m)
    plain._mtl_loss_fn = model._mtl_loss_fn
    plain.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    model.train_step((x, y))
    plain.train_step((x, y))
    for a, b in zip(model.trainable_variables, plain.trainable_variables):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)


def test_rate_term_pulls_the_foreign_mean_toward_the_target(batch):
    model = make_model(batch, rate_w=50.0, lr=3e-3)
    x, y = batch

    def gap():
        pred = model(x, training=False)
        _, _, means, _ = model.ssl_terms(y, {k: pred[k] for k in NAMES})
        return np.abs(means.numpy() - np.array([r for _, _, r in PAIRS]))

    before = gap()
    for _ in range(15):
        model.train_step(batch)
    after = gap()
    assert after.mean() < before.mean()


def test_trainer_needs_the_semi_supervised_model(batch):
    plain = MultiTaskModel(config(), dropout_rate=0.0)
    with pytest.raises(TypeError, match="SemiSupervisedModel"):
        SemiSupervisedTrainer(plain, TrainingConfig(), pooled_tasks=POOLED,
                              pooled_class_weights={0: 1.0, 1: 1.0}, ssl_pairs=PAIRS,
                              ssl_home_tasks=HOME, rate_weight=0.1)


def test_attach_refuses_unknown_heads_and_sources(batch):
    model = SemiSupervisedModel(config(), dropout_rate=0.0)
    with pytest.raises(ValueError, match="Unknown head"):
        model.attach_semi_supervised([("zz", "IAC", 0.1)], POOLED, HOME, "IAC", 1.0)
    with pytest.raises(ValueError, match="Unknown source"):
        model.attach_semi_supervised([("d1", "News", 0.1)], POOLED, HOME, "IAC", 1.0)
    with pytest.raises(ValueError, match="at least one"):
        model.attach_semi_supervised(PAIRS, POOLED, HOME, "IAC", 0.0, 0.0)
