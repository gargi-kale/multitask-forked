"""Checks for --stratify-batches (dealt_batches) in test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_stratify_batches.py

Downloads the distilbert-base-uncased tokenizer, so it needs network access.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, batch_strata, dealt_batches, load_splits,
)

N, BATCH = 1003, 16
RNG = np.random.default_rng(0)
SOURCE = RNG.choice(["a", "b", "c"], N, p=[0.5, 0.3, 0.2])
# t1 only on source a, t2 on b, t3 on c and part of a; the pool covers p1 and p2.
T1 = np.where(SOURCE == "a", RNG.integers(0, 2, N), -1).astype(np.float32)
T2 = np.where(SOURCE == "b", (RNG.random(N) < 0.1).astype(int), -1).astype(np.float32)
T3 = np.where((SOURCE == "c") | (RNG.random(N) < 0.2), RNG.integers(0, 2, N), -1).astype(np.float32)
P1 = np.where(SOURCE == "c", (RNG.random(N) < 0.3).astype(int), -1).astype(np.float32)
P2 = np.where(SOURCE == "c", (RNG.random(N) < 0.3).astype(int), -1).astype(np.float32)
LEVELS = batch_strata(SOURCE, [T1, T2, T3, np.maximum(P1, P2)])


def test_every_row_once_in_even_batches():
    batches = dealt_batches(LEVELS, BATCH, np.random.default_rng(1))
    assert len(batches) == int(np.ceil(N / BATCH))
    sizes = [len(b) for b in batches]
    assert max(sizes) - min(sizes) <= 1
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(N))


def test_each_group_spreads_within_one_row_at_every_level():
    batches = dealt_batches(LEVELS, BATCH, np.random.default_rng(1))
    for level in LEVELS:
        counts = np.stack([np.bincount(level[b], minlength=level.max() + 1) for b in batches])
        assert (counts.max(0) - counts.min(0)).max() <= 1


def test_fresh_batches_each_epoch():
    rng = np.random.default_rng(1)
    first, second = dealt_batches(LEVELS, BATCH, rng), dealt_batches(LEVELS, BATCH, rng)
    assert not np.array_equal(first[0], second[0])


def test_strata_nest_and_count_masked_labels_as_missing():
    source, pattern, vector = batch_strata(
        np.array(["a", "a", "b", "b"]),
        [np.array([1, 0, -1, -1.]), np.array([-1, -1, 1, 1.])])
    assert len(set(source)) == 2
    assert pattern[0] == pattern[1] and pattern[2] == pattern[3] and pattern[0] != pattern[2]
    assert vector[0] != vector[1] and vector[2] == vector[3]


def write(tmp_path):
    frame = pd.DataFrame({"text": [f"row number {i}" for i in range(N)], "dataset": SOURCE,
                          "t1": T1, "t2": T2, "t3": T3, "p1": P1, "p2": P2})
    for split in ("train", "val", "test"):
        frame.to_csv(tmp_path / f"no_prop_overlap_{split}.csv", index=False)
    return str(tmp_path), frame


def splits(tmp_path):
    d, frame = write(tmp_path)
    return frame, load_splits(
        ["t1", "t2", "t3", "p1", "p2"], encoder_config=ENCODER_CONFIGS["small-bert"],
        batch_size=BATCH, data_dir=d, max_length=8, seed=0,
        stratify_on=[["t1"], ["t2"], ["t3"], ["p1", "p2"]])


def test_load_splits_keeps_rows_and_labels_aligned(tmp_path):
    frame, (train, val, test) = splits(tmp_path)
    assert len(train) == int(np.ceil(N / BATCH))  # the Trainer reads this for warmup

    # Same text must carry the same labels as the unshuffled val split.
    def by_ids(ds):
        rows = {}
        for x, y in ds:
            ids = x["input_word_ids" if "input_word_ids" in x else "input_ids"].numpy()
            labels = {k: v.numpy() for k, v in y.items()}
            for i, row in enumerate(ids.tolist()):
                rows[tuple(row)] = {k: float(v[i]) for k, v in labels.items()}
        return rows

    train_rows, val_rows = by_ids(train), by_ids(val)
    assert len(train_rows) == N
    assert train_rows == val_rows


def test_load_splits_balances_batches_and_reshuffles(tmp_path):
    _, (train, val, _) = splits(tmp_path)
    first = [y["t1"].numpy() for _, y in train]
    second = [y["t1"].numpy() for _, y in train]
    labelled = [int((b != -1).sum()) for b in first]
    assert max(labelled) - min(labelled) <= 1
    assert not all(np.array_equal(a, b) for a, b in zip(first, second))
    # val keeps the CSV order
    np.testing.assert_array_equal(
        np.concatenate([y["t1"].numpy() for _, y in val]), T1)
