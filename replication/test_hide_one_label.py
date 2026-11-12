"""Checks for the hide-one-label pieces: load_splits(hide_train_columns=...,
mask_source_all_splits=...) in test_no_prop_overlap.py and the probes in
hidden_label_probe.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_hide_one_label.py

Downloads the distilbert-base-uncased tokenizer and
google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from hidden_label_probe import fit_probe, group_of, representations  # noqa: E402
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, MultiTaskModel, _any_technique, load_splits,
)
from multitask.config import ModelConfig  # noqa: E402

# rows 0-3 IAC (human labels), 4-7 Propaganda (rows 4-5 carry rule labels)
FRAME = pd.DataFrame({
    "text": [f"sentence {i}" for i in range(8)],
    "dataset": ["IAC"] * 4 + ["Propaganda"] * 4,
    "da": [1, 0, 0, 1, 1, 1, np.nan, np.nan],
    "p1": [-1, -1, -1, -1, 1, 1, 0, 0],
})


def write(tmp_path):
    for split in ("train", "val", "test"):
        FRAME.to_csv(tmp_path / f"no_prop_overlap_{split}.csv", index=False)
    return str(tmp_path)


def labels(ds):
    ys = [y for _, y in ds]
    return {k: np.concatenate([y[k].numpy() for y in ys]) for k in ys[0]}


def splits(tmp_path, **kw):
    return load_splits(["da", "p1"], encoder_config=ENCODER_CONFIGS["small-bert"], batch_size=8,
                       data_dir=write(tmp_path), max_length=8, seed=0, **kw)


def test_hidden_column_gone_from_train_only(tmp_path):
    train, val, test = splits(tmp_path, hide_train_columns=["p1"])
    tr = labels(train)
    assert np.all(tr["p1"] == -1)
    assert sorted(tr["da"].tolist()) == [-1, -1, 0, 0, 1, 1, 1, 1]  # other column untouched
    for ds in (val, test):
        np.testing.assert_array_equal(labels(ds)["p1"], FRAME["p1"].to_numpy())


def test_label_share_keeps_nested_seeded_subset_in_train_only(tmp_path):
    kept = {}
    for share in (0.25, 0.5):
        train, val, test = splits(tmp_path, train_label_share={"da": share})
        tr = labels(train)
        kept[share] = int((tr["da"] != -1).sum())
        assert sorted(tr["p1"].tolist()) == sorted(FRAME["p1"].tolist())  # other column untouched
        for ds in (val, test):
            np.testing.assert_array_equal(labels(ds)["da"], [1, 0, 0, 1, 1, 1, -1, -1])
    assert kept == {0.25: 2, 0.5: 3}  # round(share x 6 labelled rows)
    # same seed -> same rows, and the smaller share is a subset of the larger
    rows = np.flatnonzero(FRAME["da"].notna().to_numpy())
    order = np.random.default_rng(0).permutation(rows)
    train, _, _ = splits(tmp_path, train_label_share={"da": 0.5}, drop_unlabeled_train=True,
                         hide_train_columns=["p1"])
    assert sorted(labels(train)["da"].tolist()) == sorted(FRAME["da"].to_numpy()[order[:3]].tolist())


def test_repeat_train_repeats_kept_rows(tmp_path):
    train, val, _ = splits(tmp_path, train_label_share={"da": 0.5}, hide_train_columns=["p1"],
                           drop_unlabeled_train=True, repeat_train=4)
    assert len(labels(train)["da"]) == 12 and np.all(labels(train)["da"] != -1)
    assert len(labels(val)["da"]) == 8


def test_rule_labels_removed_from_every_split(tmp_path):
    for ds in splits(tmp_path, mask_train_source=("Propaganda", ["da"]),
                     mask_source_all_splits=True):
        lab = labels(ds)
        news = lab["p1"] != -1
        assert np.all(lab["da"][news] == -1)
        assert sorted(lab["da"][~news].tolist()) == [0, 0, 1, 1]
    # default: val/test keep the rule labels (the old --mask-news-discourse-train)
    _, val, _ = splits(tmp_path, mask_train_source=("Propaganda", ["da"]))
    np.testing.assert_array_equal(labels(val)["da"], [1, 0, 0, 1, 1, 1, -1, -1])


def test_any_technique():
    stacked = np.array([[-1, -1], [0, -1], [0, 1], [1, 1], [0, 0]], np.float32)
    np.testing.assert_array_equal(_any_technique(stacked), [-1, 0, 1, 1, 0])


NAMES = ["p1", "p2", "wa", "d1", "d2"]


@pytest.fixture(scope="module")
def model_and_inputs():
    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], branch_layer_sizes=[[12], [8]],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )
    model = MultiTaskModel(cfg, dropout_rate=0.0)
    rng = np.random.default_rng(0)
    ids = rng.integers(999, 30522, size=(20, 12)).astype(np.int32)
    ids[:, 0], ids[:, -1] = 101, 102
    x = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    model(x, training=False)
    return model, x


def test_group_of(model_and_inputs):
    model, _ = model_and_inputs
    assert [group_of(model, t) for t in NAMES] == [0, 0, 1, 2, 2]
    assert group_of(model, "propaganda_max_pool") == 0


def test_representations_are_the_models_own_path(model_and_inputs):
    """Each head's output, rebuilt from the 'group' representation through its
    second-level branch and output layer, equals the model's output: the
    probe reads the layers the model actually uses."""
    model, x = model_and_inputs
    ds = tf.data.Dataset.from_tensor_slices(x).batch(7)  # several batches
    reps = representations(model, ds)
    assert reps["encoder"].shape == (20, 128) and reps["shared"].shape == (20, 16)
    assert all(reps[f"group{g}"].shape == (20, 12) for g in range(3))
    out = model(x, training=False)
    for i, t in enumerate(NAMES):
        z = tf.constant(reps[f"group{group_of(model, t)}"])
        for layer in model.branch_layers[1][i]:
            z = layer(z, training=False)
        np.testing.assert_allclose(model.output_layers[t](z).numpy(), out[t].numpy(),
                                   rtol=1e-4, atol=1e-5)


def test_fit_probe_reads_a_linear_signal():
    rng = np.random.default_rng(0)

    def make(n):
        x = rng.normal(size=(n, 8)).astype(np.float32)
        return x, (x[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(int)

    s = fit_probe(*make(400), *make(200), *make(200), seed=0)
    assert s["test"]["auc"] > 0.97 and s["test"]["f1"] > 0.9 and s["n_train"] == 400
    noise = fit_probe(rng.normal(size=(400, 8)), rng.integers(0, 2, 400),
                      rng.normal(size=(200, 8)), rng.integers(0, 2, 200),
                      rng.normal(size=(200, 8)), rng.integers(0, 2, 200), seed=0)
    assert 0.35 < noise["test"]["auc"] < 0.65
