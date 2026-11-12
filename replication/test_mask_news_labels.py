"""Checks for --mask-news-discourse-train (the rule-made IAC labels on
Propaganda rows are masked in train only) in test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_mask_news_labels.py

Downloads the distilbert-base-uncased tokenizer, so it needs network access.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, compute_full_train_binary_class_weights, load_splits,
)

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


def test_news_labels_masked_in_train_only(tmp_path):
    train, val, test = load_splits(
        ["da", "p1"], encoder_config=ENCODER_CONFIGS["small-bert"], batch_size=8,
        data_dir=write(tmp_path), max_length=8, seed=0,
        mask_train_source=("Propaganda", ["da"]))
    tr = labels(train)
    news = tr["p1"] != -1  # train is shuffled; Propaganda rows have p1 labels
    assert np.all(tr["da"][news] == -1)
    assert sorted(tr["da"][~news].tolist()) == [0, 0, 1, 1]  # IAC labels untouched
    assert sorted(tr["p1"][news].tolist()) == [0, 0, 1, 1]  # technique labels untouched
    for ds in (val, test):  # rule labels kept outside train
        np.testing.assert_array_equal(labels(ds)["da"], [1, 0, 0, 1, 1, 1, -1, -1])


def test_class_weights_follow_the_mask(tmp_path):
    d = write(tmp_path)
    w = compute_full_train_binary_class_weights(
        data_dir=d, task_names=["da", "p1"], mask_source=("Propaganda", ["da"]))
    assert w[0] == {0: 4 / 4, 1: 4 / 4}  # IAC rows only: 2 zeros, 2 ones
    assert w[1] == {0: 4 / 4, 1: 4 / 4}  # not named, unchanged
    assert compute_full_train_binary_class_weights(data_dir=d, task_names=["da"])[0] == \
        {0: 6 / 4, 1: 6 / 8}  # without the mask: 2 zeros, 4 ones
