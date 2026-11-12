"""Checks for --propaganda-zero-fill (missing train labels become 0) in
test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_zero_fill.py

Downloads the distilbert-base-uncased tokenizer, so it needs network access.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, compute_full_train_binary_class_weights,
    compute_full_train_pooled_class_weights, load_splits,
)

# rows 0-3 propaganda (labelled), 4-7 another source (propaganda labels missing)
FRAME = pd.DataFrame({
    "text": [f"sentence {i}" for i in range(8)],
    "p1": [1, 0, 0, 0, -1, -1, -1, -1],
    "p2": [0, 1, 0, 0, -1, -1, -1, -1],
    "wa": [-1, -1, -1, -1, 1, 0, 1, 1],
})


def write(tmp_path):
    for split in ("train", "val", "test"):
        FRAME.to_csv(tmp_path / f"no_prop_overlap_{split}.csv", index=False)
    return str(tmp_path)


def labels(ds):
    ys = [y for _, y in ds]
    return {k: np.concatenate([y[k].numpy() for y in ys]) for k in ys[0]}


def test_only_named_train_labels_are_filled(tmp_path):
    train, val, test = load_splits(
        ["p1", "p2", "wa"], encoder_config=ENCODER_CONFIGS["small-bert"], batch_size=8,
        data_dir=write(tmp_path), max_length=8, seed=0, zero_fill_train=["p1", "p2"])
    tr = labels(train)
    order = np.argsort(np.where(tr["wa"] == -1, 0, 1), kind="stable")  # train is shuffled
    assert sorted(tr["p1"].tolist()) == [0] * 7 + [1]
    assert sorted(tr["wa"].tolist()) == [-1] * 4 + [0, 1, 1, 1]  # not named, still masked
    assert np.all(tr["p2"][order][4:] == 0)
    for ds in (val, test):  # still masked outside train
        np.testing.assert_array_equal(labels(ds)["p1"], FRAME["p1"].to_numpy())


def test_class_weights_follow_the_filled_labels(tmp_path):
    d = write(tmp_path)
    w = compute_full_train_binary_class_weights(data_dir=d, task_names=["p1", "wa"],
                                                zero_fill=["p1"])
    assert w[0] == {0: 8 / 14, 1: 8 / 2}  # 7 zeros, 1 one over all 8 rows
    assert w[1] == {0: 4 / 2, 1: 4 / 6}  # unchanged: 4 labelled rows
    assert compute_full_train_pooled_class_weights(data_dir=d, task_names=["p1", "p2"]) == \
        {0: 4 / 4, 1: 4 / 4}  # 2 of 4 propaganda rows have a technique
    assert compute_full_train_pooled_class_weights(
        data_dir=d, task_names=["p1", "p2"], zero_fill=True) == {0: 8 / 12, 1: 8 / 4}
