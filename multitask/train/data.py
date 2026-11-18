"""Loading train/val/test splits, batch builders and class weights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer

from multitask.config import EncoderConfig, EncoderInputType

from multitask.train.encoders import ENCODER_CONFIGS

# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------

def batch_strata(sources: np.ndarray, outputs: Sequence[np.ndarray],
                 mask_value: float = -1.0) -> List[np.ndarray]:
    """Group ids for ``dealt_batches``, coarse to fine: source corpus, which
    outputs are labelled, and the full label vector."""
    y = np.stack(outputs, axis=1)
    group_ids = lambda m: np.unique(m, axis=0, return_inverse=True)[1].ravel()
    return [pd.factorize(sources)[0], group_ids(y != mask_value), group_ids(y)]


def dealt_batches(levels: Sequence[np.ndarray], batch_size: int,
                  rng: np.random.Generator) -> List[np.ndarray]:
    """One epoch of index batches. ``levels`` run coarse to fine: rows sorted by
    them, randomly within each group, are dealt out to batches like cards, so
    each group splits evenly across batches (within one row). Groups keep their
    natural share, so the class weights and task weights still apply."""
    n = len(levels[0])
    keys = [rng.random(n)] + [rng.permutation(lv.max() + 1)[lv] for lv in reversed(levels)]
    n_batches = int(np.ceil(n / batch_size))
    batch = np.empty(n, dtype=np.int64)
    batch[np.lexsort(keys)] = (np.arange(n) + rng.integers(n_batches)) % n_batches
    return np.split(np.argsort(batch), np.cumsum(np.bincount(batch))[:-1])


def length_grouped_batches(lengths: np.ndarray, sources: np.ndarray, batch_size: int,
                           chunk_batches: int, rng: np.random.Generator) -> List[np.ndarray]:
    """One epoch of index batches in which texts of similar length share a
    batch, so each batch can be padded to its own longest text instead of to
    ``max_length``.

    The rows are shuffled and cut into chunks of ``chunk_batches`` batches.
    Inside a chunk every row gets its length rank *within its own source*, as
    a share of that source's rows in the chunk; the chunk is sorted by that
    share and cut into batches. Batch i of a chunk therefore holds the i-th
    length slice of every source: lengths are similar within a source, and
    each batch keeps the chunk's mix of sources (plain sorting by length would
    put the long forum posts in batches of their own). The batches of the
    epoch are then shuffled. Every row appears exactly once."""
    n = len(lengths)
    perm = rng.permutation(n)
    batches: List[np.ndarray] = []
    for start in range(0, n, batch_size * chunk_batches):
        idx = perm[start:start + batch_size * chunk_batches]
        share = np.empty(len(idx))
        for s in np.unique(sources[idx]):
            rows = np.flatnonzero(sources[idx] == s)
            # stable sort of shuffled rows: ties in length stay in random order
            rows = rows[np.argsort(lengths[idx[rows]], kind="stable")]
            share[rows] = (np.arange(len(rows)) + 0.5) / len(rows)
        idx = idx[np.argsort(share, kind="stable")]
        batches += [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def trim_padding(x, y):
    """Cut a tokenized batch down to its longest text. The tokenizer pads on
    the right and the encoder ignores padded positions, so the model's outputs
    are unchanged; only the work on padding is saved."""
    longest = tf.reduce_max(tf.reduce_sum(tf.cast(x["attention_mask"], tf.int32), axis=1))
    longest = tf.maximum(longest, 1)
    return {k: v[:, :longest] for k, v in x.items()}, y


# Label key holding each train row's position (load_splits(train_row_ids=True)).
ROW_ID_KEY = "__row"


def load_splits(
    label_columns: Sequence[str] = (),
    *,
    multilabel_groups: Optional[Dict[str, Sequence[str]]] = None,
    encoder_config: Optional[EncoderConfig] = None,
    text_column: str = "text",
    batch_size: int = 32,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 96,
    data_dir: str = "data",
    prefix: str = "no_prop_overlap_",
    na_strings: Tuple[str, ...] = ("NA", "na", "NaN", ""),
    mask_value: float = -1.0,
    label_column_map: Optional[Dict[str, str]] = None,
    seed: Optional[int] = None,
    drop_unlabeled_train: bool = False,
    zero_fill_train: Sequence[str] = (),
    stratify_on: Sequence[Sequence[str]] = (),
    mask_train_source: Optional[Tuple[str, Sequence[str]]] = None,
    mask_source_all_splits: bool = False,
    hide_train_columns: Sequence[str] = (),
    train_row_ids: bool = False,
    train_label_share: Optional[Dict[str, float]] = None,
    repeat_train: int = 1,
    embedding_source: Optional[Any] = None,
    length_chunk_batches: int = 0,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """``drop_unlabeled_train`` removes train rows where every requested label
    is missing (they add no loss, only steps). Val/test keep every row so that
    saved predictions stay aligned with the CSVs. ``zero_fill_train`` names
    label columns whose missing train labels become 0 instead of masked; val
    and test stay masked, so they are still scored on labelled rows only.
    ``stratify_on`` replaces the train shuffle with ``dealt_batches``: one entry
    per output, each a list of label columns whose max is that output's label.
    ``mask_train_source`` = (source, columns) masks those label columns on the
    train rows whose ``dataset`` is that source (e.g. the rule-made IAC labels
    on Propaganda rows); val and test keep them, unless
    ``mask_source_all_splits`` masks them in every split. ``hide_train_columns``
    masks those label columns on every train row (hide-one-label runs); val and
    test keep them. ``train_row_ids`` adds each train row's position as the
    label ``ROW_ID_KEY`` (int32), so stored per-row scores can be looked up
    after shuffling (self-distillation). ``train_label_share`` = {column:
    share} keeps that share of the column's train labels and masks the rest
    (label-budget runs); the kept rows are the first ones of a permutation
    seeded by ``seed``, so for one seed a smaller share is a subset of a
    larger one, whatever else the run trains on. ``repeat_train`` > 1 repeats
    the train rows that many times (after ``drop_unlabeled_train``), so a run
    on few rows takes as many steps per epoch as one on the full set.
    ``embedding_source`` (texts -> float32 [n, dim]) replaces tokenizing: the
    model's input is then each row's pre-computed embedding (see
    ``openai_embedding_source``). ``length_chunk_batches`` > 0 pads every
    batch to its own longest text instead of to ``max_length``
    (``trim_padding``): train batches group texts of similar length
    (``length_grouped_batches``, chunks of that many batches, sources read
    from the ``dataset`` column), val and test keep the CSV order."""
    multilabel_groups = multilabel_groups or {}
    if length_chunk_batches:
        if stratify_on or embedding_source is not None or (
                encoder_config is not None
                and encoder_config.encoder_input_type != EncoderInputType.HUGGINGFACE_TOKENS):
            raise ValueError("length_chunk_batches needs a HuggingFace token encoder and "
                             "can't be combined with stratify_on")
    label_column_map = label_column_map or {}
    encoder_config = encoder_config or ENCODER_CONFIGS["distilbert"]
    tokenizer = (None if embedding_source is not None
                 else AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False))

    def _to_float(s: pd.Series) -> np.ndarray:
        s = s.replace(list(na_strings), np.nan)
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        return np.where(np.isnan(vals), mask_value, vals)

    def build(split: str) -> tf.data.Dataset:
        df = pd.read_csv(f"{data_dir}/{prefix}{split}.csv")
        labels: Dict[str, np.ndarray] = {}

        for col in label_columns:
            source_col = label_column_map.get(col, col)
            labels[col] = _to_float(df[source_col])
            if split == "train" and col in zero_fill_train:
                labels[col] = np.where(labels[col] == mask_value, 0.0, labels[col]).astype(np.float32)

        if mask_train_source and (split == "train" or mask_source_all_splits):
            source, cols = mask_train_source
            rows = (df["dataset"] == source).to_numpy()
            n_masked = 0
            for col in cols:
                n_masked += int((rows & (labels[col] != mask_value)).sum())
                labels[col] = np.where(rows, mask_value, labels[col]).astype(np.float32)
            print(f"{split}: masked {n_masked:,} labels in {len(cols)} columns on "
                  f"{int(rows.sum()):,} {source} rows")

        if split == "train" and hide_train_columns:
            n_hidden = 0
            for col in hide_train_columns:
                n_hidden += int((labels[col] != mask_value).sum())
                labels[col] = np.full_like(labels[col], mask_value)
            print(f"train: hid all {n_hidden:,} labels of {list(hide_train_columns)}")

        if split == "train" and train_label_share:
            for col, share in train_label_share.items():
                rows = np.flatnonzero(labels[col] != mask_value)
                order = np.random.default_rng(seed or 0).permutation(rows)
                n_keep = max(1, int(round(share * rows.size)))
                labels[col] = labels[col].copy()
                labels[col][order[n_keep:]] = mask_value
                kept = labels[col][order[:n_keep]]
                print(f"train: kept {n_keep:,} of {rows.size:,} labels of {col} "
                      f"({100 * share:g}%; {int((kept == 1).sum()):,} yes)")

        for name, cols in multilabel_groups.items():
            labels[name] = np.stack([_to_float(df[c]) for c in cols], axis=1)

        if split == "train" and drop_unlabeled_train:
            keep = np.zeros(len(df), dtype=bool)
            for arr in labels.values():
                valid = arr != mask_value
                keep |= valid if valid.ndim == 1 else valid.any(axis=1)
            print(f"train: keeping {keep.sum():,} of {len(df):,} rows with a label")
            df = df[keep].reset_index(drop=True)
            labels = {k: v[keep] for k, v in labels.items()}

        if split == "train" and repeat_train > 1:
            print(f"train: repeating {len(df):,} rows x{repeat_train}")
            df = pd.concat([df] * repeat_train, ignore_index=True)
            labels = {k: np.concatenate([v] * repeat_train) for k, v in labels.items()}

        if split == "train" and train_row_ids:
            labels[ROW_ID_KEY] = np.arange(len(df), dtype=np.int32)

        texts = ["" if pd.isna(x) else str(x) for x in df[text_column]]
        if embedding_source is not None:
            inputs = np.asarray(embedding_source(texts), dtype=np.float32)
            print(f"{split}: {inputs.shape[0]:,} pre-computed embeddings of "
                  f"{inputs.shape[1]:,} numbers", flush=True)
        else:
            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )

        if embedding_source is not None:
            pass
        elif encoder_config.encoder_input_type == EncoderInputType.TFHUB_TOKENS:
            input_type_ids = (
                enc["token_type_ids"].astype(np.int32)
                if "token_type_ids" in enc
                else np.zeros_like(enc["input_ids"], dtype=np.int32)
            )
            inputs = {
                "input_word_ids": enc["input_ids"].astype(np.int32),
                "input_mask": enc["attention_mask"].astype(np.int32),
                "input_type_ids": input_type_ids,
            }
        elif encoder_config.encoder_input_type == EncoderInputType.HUGGINGFACE_TOKENS:
            inputs = {
                "input_ids": enc["input_ids"].astype(np.int32),
                "attention_mask": enc["attention_mask"].astype(np.int32),
            }
        else:
            raise ValueError(
                f"load_splits only supports tokenized encoders in this script. "
                f"Got encoder_input_type={encoder_config.encoder_input_type!r}."
            )

        if split == "train" and stratify_on:
            levels = batch_strata(
                df["dataset"].to_numpy(),
                [np.max([labels[c] for c in cols], axis=0) for cols in stratify_on], mask_value)
            print("train: dealing batches over " + " > ".join(
                f"{lv.max() + 1} {name}" for lv, name in
                zip(levels, ("sources", "label patterns", "label vectors"))))
            rng = np.random.default_rng(seed)
            arrays = (inputs, labels)

            def batches():
                for idx in dealt_batches(levels, batch_size, rng):
                    yield tf.nest.map_structure(lambda a: a[idx], arrays)

            ds = tf.data.Dataset.from_generator(
                batches, output_signature=tf.nest.map_structure(
                    lambda a: tf.TensorSpec((None, *a.shape[1:]), a.dtype), arrays))
            # The Trainer sizes the warmup from the dataset's length.
            return ds.apply(tf.data.experimental.assert_cardinality(
                int(np.ceil(len(df) / batch_size)))).prefetch(tf.data.AUTOTUNE)

        if split == "train" and length_chunk_batches:
            lengths = inputs["attention_mask"].sum(axis=1)
            sources = (pd.factorize(df["dataset"])[0] if "dataset" in df
                       else np.zeros(len(df), dtype=np.int64))
            padded = len(df) * inputs["input_ids"].shape[1]
            rng = np.random.default_rng(seed)
            first = length_grouped_batches(lengths, sources, batch_size, length_chunk_batches,
                                           np.random.default_rng(seed))
            used = sum(int(lengths[b].max()) * len(b) for b in first)
            print(f"train: batches of similar length (chunks of {length_chunk_batches} batches, "
                  f"{int(sources.max()) + 1} sources): {100 * used / padded:.1f}% of the "
                  f"token positions of padding to {inputs['input_ids'].shape[1]}", flush=True)

            def grouped():
                for idx in length_grouped_batches(lengths, sources, batch_size,
                                                  length_chunk_batches, rng):
                    longest = max(int(lengths[idx].max()), 1)
                    yield ({k: v[idx, :longest] for k, v in inputs.items()},
                           {k: v[idx] for k, v in labels.items()})

            ds = tf.data.Dataset.from_generator(grouped, output_signature=(
                {k: tf.TensorSpec((None, None), v.dtype) for k, v in inputs.items()},
                {k: tf.TensorSpec((None, *v.shape[1:]), v.dtype) for k, v in labels.items()}))
            # The Trainer sizes the warmup from the dataset's length.
            return ds.apply(tf.data.experimental.assert_cardinality(
                int(np.ceil(len(df) / batch_size)))).prefetch(tf.data.AUTOTUNE)

        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if split == "train":
            # Full-length buffer: the train CSV is sorted by source corpus, so a
            # short buffer yields single-corpus batches for a whole third of an epoch.
            ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if length_chunk_batches:
            ds = ds.map(trim_padding, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    return build("train"), build("val"), build("test")


def compute_full_train_binary_class_weights(
    *,
    data_dir: str,
    task_names: Sequence[str],
    label_column_map: Optional[Dict[str, str]] = None,
    prefix: str = "no_prop_overlap_",
    mask_value: float = -1.0,
    na_strings: Tuple[str, ...] = ("NA", "na", "NaN", ""),
    zero_fill: Sequence[str] = (),
    mask_source: Optional[Tuple[str, Sequence[str]]] = None,
) -> List[Dict[int, float]]:
    """Compute binary class weights from the full train CSV.

    Uses the same weighting rule as ``Trainer._compute_class_weights`` for
    binary tasks: ``w_c = N_valid / (2 * count_c)``, with a 0.001 floor for
    absent classes and uniform ``{0: 1.0, 1: 1.0}`` when no valid labels exist.
    """
    label_column_map = label_column_map or {}
    train_path = f"{data_dir}/{prefix}train.csv"
    df = pd.read_csv(train_path)

    class_weights: List[Dict[int, float]] = []
    for task_name in task_names:
        source_col = label_column_map.get(task_name, task_name)
        s = df[source_col].replace(list(na_strings), np.nan)
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        vals = np.where(np.isnan(vals), mask_value, vals)
        if task_name in zero_fill:  # as load_splits(zero_fill_train=...)
            vals = np.where(vals == mask_value, 0.0, vals)
        if mask_source and task_name in mask_source[1]:  # as load_splits(mask_train_source=...)
            vals = np.where((df["dataset"] == mask_source[0]).to_numpy(), mask_value, vals)

        valid = vals[vals != mask_value]
        if valid.size == 0:
            class_weights.append({0: 1.0, 1: 1.0})
            continue

        class_counts = np.bincount(valid.astype(int), minlength=2)
        n_valid = int(valid.size)
        weights_for_task: Dict[int, float] = {}
        for class_idx in (0, 1):
            count = int(class_counts[class_idx])
            if count == 0:
                weights_for_task[class_idx] = 0.001
            else:
                weights_for_task[class_idx] = n_valid / (2.0 * count)
        class_weights.append(weights_for_task)

    return class_weights


def compute_full_train_pooled_class_weights(
    *,
    data_dir: str,
    task_names: Sequence[str],
    label_column_map: Optional[Dict[str, str]] = None,
    prefix: str = "no_prop_overlap_",
    mask_value: float = -1.0,
    na_strings: Tuple[str, ...] = ("NA", "na", "NaN", ""),
    zero_fill: bool = False,
) -> Dict[int, float]:
    """Binary class weights for the max over ``task_names`` (e.g. the pooled
    propaganda label), same rule as ``compute_full_train_binary_class_weights``.
    A row is masked only when every constituent column is missing."""
    label_column_map = label_column_map or {}
    df = pd.read_csv(f"{data_dir}/{prefix}train.csv")
    cols = [label_column_map.get(t, t) for t in task_names]
    vals = df[cols].replace(list(na_strings), np.nan).apply(pd.to_numeric, errors="coerce")
    vals = vals.where(vals != mask_value)
    if zero_fill:  # missing labels count as 0, as load_splits(zero_fill_train=...)
        vals = vals.fillna(0.0)
    pooled = vals.max(axis=1, skipna=True).dropna().to_numpy().astype(int)
    counts = np.bincount(pooled, minlength=2)
    return {c: (pooled.size / (2.0 * counts[c]) if counts[c] else 0.001) for c in (0, 1)}
