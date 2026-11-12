"""Checks for EpochScorer (per-epoch val/test scores and early stopping on
val F1 / AUC) in test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_epoch_scorer.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_gradient_probe import data, make_model  # noqa: E402
from test_no_prop_overlap import EpochScorer, PooledOutputLossTrainer  # noqa: E402
from multitask.config import TrainingConfig  # noqa: E402

OUTPUTS = ["propaganda_max_pool", "wa", "d1", "d2"]


class WeightLog(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.get_weights())


def fit(monitor, patience, epochs=6, main_output=None):
    tf.random.set_seed(0)
    model, loss_fn = make_model()
    train, val, test = data(seed=0), data(seed=1), data(seed=2)
    model(next(iter(train))[0], training=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
    scorer = EpochScorer(val, test, monitor=monitor, patience=patience, outputs=OUTPUTS,
                         main_output=main_output)
    scorer.trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                             pooled_class_weights={0: 1.0, 1: 1.0})
    log = WeightLog()
    hist = model.fit(train, epochs=epochs, validation_data=val, verbose=0,
                     callbacks=[scorer, log]).history
    return model, scorer, log, hist


@pytest.mark.parametrize("monitor", ["f1", "loss"])
def test_records_scores_and_restores_the_best_epoch(monitor):
    model, scorer, log, hist = fit(monitor, patience=2)
    recs = scorer.records
    assert [r["epoch"] for r in recs] == list(range(1, len(recs) + 1))
    for r in recs:
        assert set(r["val"]) == set(r["test"]) == set(OUTPUTS)
        for m in ("f1", "auc"):
            assert 0.0 <= r[f"val_{m}"] <= 1.0 and 0.0 <= r[f"test_{m}"] <= 1.0
            np.testing.assert_allclose(r[f"val_{m}"], np.mean([s[m] for s in r["val"].values()]))
    np.testing.assert_allclose(hist["val_f1"], [r["val_f1"] for r in recs])
    np.testing.assert_allclose(hist["val_loss"], [r["val_loss"] for r in recs])

    key = "val_loss" if monitor == "loss" else "val_f1"
    vals = [r[key] for r in recs]
    best = 1 + int(np.argmin(vals) if monitor == "loss" else np.argmax(vals))
    assert scorer.best_epoch == best
    for a, b in zip(model.get_weights(), log.weights[best - 1]):
        np.testing.assert_array_equal(a, b)


def test_stops_only_once_every_metric_has_stalled():
    _, scorer, _, _ = fit("f1", patience=1, epochs=8)
    n = len(scorer.records)
    for m in ("loss", "f1", "auc"):
        assert n - scorer._best_by(m) >= 1 or n == 8
    if n < 8:  # one epoch earlier at least one metric was still improving
        scorer.records = scorer.records[:-1]
        assert any(n - 1 - scorer._best_by(m) < 1 for m in ("loss", "f1", "auc"))


def test_best_by_takes_the_first_best_epoch():
    scorer = EpochScorer(None, None, monitor="auc")
    scorer.records = [{"val_loss": 0.5, "val_f1": 0.6, "val_auc": 0.7},
                      {"val_loss": 0.4, "val_f1": 0.6, "val_auc": 0.6},
                      {"val_loss": 0.4, "val_f1": 0.5, "val_auc": 0.7}]
    assert [scorer._best_by(m) for m in ("loss", "f1", "auc")] == [2, 1, 1]
    with pytest.raises(ValueError):
        EpochScorer(None, None, monitor="accuracy")


def test_main_output_picks_and_stops_on_that_output_alone():
    model, scorer, log, _ = fit("f1", patience=1, epochs=8, main_output="wa")
    recs = scorer.records
    for r in recs:
        assert set(r["val"]) == set(OUTPUTS)  # every output still scored
        for m in ("f1", "auc"):
            assert r[f"val_main_{m}"] == r["val"]["wa"][m]
            assert r[f"test_main_{m}"] == r["test"]["wa"][m]
    best = 1 + int(np.argmax([r["val"]["wa"]["f1"] for r in recs]))
    assert scorer.best_epoch == best
    for a, b in zip(model.get_weights(), log.weights[best - 1]):
        np.testing.assert_array_equal(a, b)
    n = len(recs)
    if n < 8:  # stopped: the main output's F1 and AUC had both stalled
        for m in ("f1", "auc"):
            assert n - scorer._best_by(m) >= 1


def test_main_output_best_by_ignores_the_mean():
    scorer = EpochScorer(None, None, monitor="f1", main_output="wa")
    scorer.records = [{"val_loss": 0.5, "val_f1": 0.9, "val_auc": 0.9,
                       "val_main_f1": 0.5, "val_main_auc": 0.6},
                      {"val_loss": 0.4, "val_f1": 0.1, "val_auc": 0.1,
                       "val_main_f1": 0.7, "val_main_auc": 0.6}]
    assert [scorer._best_by(m) for m in ("f1", "auc")] == [2, 1]
    assert scorer._better(scorer.records[1], scorer.records[0])
    with pytest.raises(ValueError):
        EpochScorer(None, None, monitor="loss", main_output="wa")


def test_mask_sweep_rate_zero_changes_nothing_and_masking_does():
    from test_no_prop_overlap import mask_sweep

    _, scorer, _, _ = fit("f1", patience=2, epochs=2)
    res = mask_sweep(scorer, [0.0, 0.5], repeats=2)
    assert set(res["clean"]) == set(OUTPUTS) and res["by_rate"].keys() == {"0", "0.5"}
    for run in res["by_rate"]["0"]:
        for k in OUTPUTS:
            np.testing.assert_allclose(run[k]["f1"], res["clean"][k]["f1"], atol=1e-6)
            assert run[k]["flip_labelled"] == 0.0 and run[k]["shift_labelled"] < 1e-5
    shifts = [run[k]["shift_labelled"] for run in res["by_rate"]["0.5"] for k in OUTPUTS]
    assert max(shifts) > 1e-4
    for run in res["by_rate"]["0.5"]:
        for k in OUTPUTS:
            assert 0.0 <= run[k]["flip_labelled"] <= 1.0
            assert run[k]["flip_unlabelled"] is None or 0.0 <= run[k]["flip_unlabelled"] <= 1.0
