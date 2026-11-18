"""
End-to-end real-world experiments on the ``data/no_prop_overlap_*.csv`` datasets.

Dataset (see ``testing.md`` §4.3 and ``docs/DATA_FORMAT.md``):
- ``text``: input sentence / snippet.
- Propaganda techniques (multilabel, 18 columns): see
  :data:`PROPAGANDA_TECHNIQUE_COLUMNS`.
- Discourse dimensions (continuous, regression): see
  :data:`DISCOURSE_REGRESSION_COLUMNS`; binary versions with ``_bi`` suffix in
  :data:`DISCOURSE_BINARY_COLUMNS`.
- ``propaganda`` (binary/multiclass — inspect values before training),
  ``WA`` / ``WA_bi`` (argument quality)
- ``dataset`` (source id — stratification, not a label).
- ``NA`` entries mean "annotation doesn't apply" and are converted to
  ``mask_value=-1.0`` so the trainer masks them out of the loss.

Usage:
    python tests/test_no_prop_overlap.py <experiment> [options]

Experiments:
    propaganda          Propaganda techniques only (multilabel, 18-class head)
    propaganda_cols_binary
                        Propaganda technique columns as 18 separate binary heads
    discourse_dims      Discourse dimensions only (regression)
    discourse_binary    Binary discourse dimensions
    discourse_binary_single_task
                        Each binary discourse dimension trained as its own model
    propaganda_binary   Propaganda techniques + binary discourse dimensions
    all_task_types      All task types combined
    all_task_types_binary_prop
                        All task types with propaganda columns as binary heads (flat)
    all_task_types_binary_prop_grouped
                        Binary-propaganda all-task model with grouped hierarchy
    all_task_types_binary_prop_deep_grouped
                        Binary-propaganda all-task model with deep grouped hierarchy
    all_task_types_pair_by_dimension
                        Pair-by-dimension hierarchy (regression+binary sibling pairs)
    all_task_types_grouped
                        All task types with grouped hierarchy by task family
    all_task_types_deep_grouped
                        All task types with deep hierarchy (all -> some -> tasks)
    paper_replication   Propaganda binary cols + WA_bi + binary discourse, grouped hierarchy
                        Max-pooled propaganda
    all                 Run all experiments in sequence

Options:
    --data-dir DIR          Path to data directory (default: data)
    --epochs N              Training epochs (default: 10)
    --batch-size N          Batch size (default: 128)
    --learning-rate LR      Learning rate (default: 4e-5)
    --no-optimize-thresh    Disable threshold optimization
    --verbose N             Keras verbosity 0/1/2 (default: 1)
    --encoder {distilbert,tfhub-bert,small-bert,electra-small,electra-small-L2,electra-small-L4,electra-small-L6,openai-small,openai-large}
                            Encoder config to use (default: distilbert)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer

# NOTE: mixed_float16 makes model outputs float16, which breaks the trainer's
# loss arithmetic (float16 * float32 Mul dtype mismatch in dict_weighted_masked_loss).
# Keeping float32 globally avoids that without changing the trainer.
tf.keras.mixed_precision.set_global_policy("float32")

from multitask.config import (
    ModelConfig, EncoderConfig, TrainingConfig,
    EncoderIntegration, EncoderInputType,
)
from multitask import MultiTaskModel, Trainer
from multitask.models.hierarchical import pool_probabilities

# How the 18 technique heads are pooled into the propaganda output (see
# pool_probabilities). The output keeps the name propaganda_max_pool whatever
# the strategy, so reports and saved predictions stay comparable.
PROPAGANDA_POOLS = ("max", "noisy_or", "lse")

# ---------------------------------------------------------------------------
# Column groups (testing.md §4.3)
# ---------------------------------------------------------------------------

PROPAGANDA_TECHNIQUE_COLUMNS: Tuple[str, ...] = (
    "Loaded_Language", "Slogans", "Black-and-White_Fallacy", "Flag-Waving",
    "Name_Calling", "Whataboutism", "Causal_Oversimplification",
    "Exaggeration,Minimisation", "Doubt", "Appeal_to_Authority", "Repetition",
    "Appeal_to_fear-prejudice", "Red_Herring", "Thought-terminating_Cliches",
    "Reductio_ad_hitlerum", "Bandwagon",
    "Obfuscation,Intentional_Vagueness,Confusion", "Straw_Men",
)

DISCOURSE_REGRESSION_COLUMNS: Tuple[str, ...] = (
    "disagree_agree", "attacking_respectful", "emotion_fact", "nasty_nice",
    "personal_audience", "defeater_undercutter", "negotiate_attack",
    "questioning_asserting",
)

DISCOURSE_BINARY_COLUMNS: Tuple[str, ...] = tuple(
    f"{c}_bi" for c in DISCOURSE_REGRESSION_COLUMNS
)

# For experiments that model propaganda techniques as separate binary heads,
# sanitize task names so Keras layer scopes remain valid (e.g., commas are invalid).
def _sanitize_task_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", name)
    if not safe or not re.match(r"^[A-Za-z0-9.]", safe):
        safe = f"task_{safe}"
    return safe


PROPAGANDA_BINARY_TASK_NAMES: Tuple[str, ...] = tuple(
    _sanitize_task_name(col) for col in PROPAGANDA_TECHNIQUE_COLUMNS
)

if len(set(PROPAGANDA_BINARY_TASK_NAMES)) != len(PROPAGANDA_BINARY_TASK_NAMES):
    raise ValueError(
        "Sanitized propaganda binary task names are not unique; adjust naming."
    )

PROPAGANDA_BINARY_LABEL_MAP: Dict[str, str] = dict(
    zip(PROPAGANDA_BINARY_TASK_NAMES, PROPAGANDA_TECHNIQUE_COLUMNS)
)


# ---------------------------------------------------------------------------
# Encoder configs
# ---------------------------------------------------------------------------

ecfg_distilbert_uncased = EncoderConfig(
    encoder_identifier="distilbert-base-uncased",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True,
)

ecfg_tfhub_bert = EncoderConfig(
    encoder_identifier="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    encoder_input_type=EncoderInputType.TFHUB_TOKENS,
    encoder_integration=EncoderIntegration.TFHUB,
    trainable_encoder=True,
)

ecfg_small_bert = EncoderConfig(
    encoder_identifier="google/bert_uncased_L-2_H-128_A-2",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True,
    from_pt=True
)

# ELECTRA-small's discriminator: 12 layers, hidden 256, ~13.5M parameters. It
# uses BERT's uncased vocabulary (the vocab files are identical), so the
# tokenizer in load_splits and the special ids in mask_tokens apply unchanged.
# It has no pooler, so the model reads the [CLS] position of the last layer,
# and no masked-word head (see NoMLMHead).
ecfg_electra_small = EncoderConfig(
    encoder_identifier="google/electra-small-discriminator",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True,
)


def shortened_electra_small(num_layers: int) -> EncoderConfig:
    """ELECTRA-small cut down to its embeddings and first `num_layers` of 12
    transformer layers, which keep their pretrained weights; the layers above
    are dropped. There is no official ELECTRA checkpoint smaller than
    ELECTRA-small. Parameters: 2 layers ~5.6M (the small BERT has ~4.4M),
    4 layers ~7.2M, 6 layers ~8.7M, all 12 ~13.5M."""
    return EncoderConfig(
        encoder_identifier="google/electra-small-discriminator",
        encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
        encoder_integration=EncoderIntegration.HUGGINGFACE,
        trainable_encoder=True,
        encoder_kwargs={"num_hidden_layers": num_layers},
    )


# OpenAI embedding models (paper_replication only). The model is frozen, so its
# vectors are embedded once, stored (openai_embeddings.py, filled by
# `modal run modal_embed.py::embed`) and fed to the model as pre-computed
# inputs: there is no encoder in the model, and nothing below the shared layers
# trains. --encoder name: OpenAI model.
OPENAI_EMBEDDING_MODELS = {
    "openai-small": "text-embedding-3-small",  # 1,536 numbers per text
    "openai-large": "text-embedding-3-large",  # 3,072 numbers per text
}
OPENAI_EMBEDDING_DIMS = {"openai-small": 1536, "openai-large": 3072}


def precomputed_embeddings(embedding_dim: int) -> EncoderConfig:
    return EncoderConfig(
        encoder_input_type=EncoderInputType.PRECOMPUTED,
        embedding_dim=embedding_dim,
        trainable_encoder=False,
    )


def openai_embedding_source(encoder: str, cache_root: str):
    """texts -> float32 [n, dim] from the stored vectors of ``encoder``.

    OpenAI vectors have length 1, so each of their ``dim`` numbers is tiny
    (about 1 / sqrt(dim) = 0.026 for openai-small), far below the roughly
    unit-sized BERT outputs the dense layers' initialisation expects. The
    vectors are multiplied by sqrt(dim) so their numbers are about unit-sized;
    this rescales every text alike and changes no direction or similarity."""
    from openai_embeddings import load_embeddings

    model = OPENAI_EMBEDDING_MODELS[encoder]
    scale = np.float32(np.sqrt(OPENAI_EMBEDDING_DIMS[encoder]))

    def source(texts: Sequence[str]) -> np.ndarray:
        return load_embeddings(texts, model, cache_root) * scale

    return source


ENCODER_CONFIGS = {
    "distilbert": ecfg_distilbert_uncased,
    "tfhub-bert": ecfg_tfhub_bert,
    "small-bert": ecfg_small_bert,
    "electra-small": ecfg_electra_small,
    "electra-small-L2": shortened_electra_small(2),
    "electra-small-L4": shortened_electra_small(4),
    "electra-small-L6": shortened_electra_small(6),
    **{name: precomputed_embeddings(dim) for name, dim in OPENAI_EMBEDDING_DIMS.items()},
}

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


def binary_kl(t, p, eps=1e-7):
    """Per-row KL(Bernoulli(t) || Bernoulli(p)) = BCE(t, p) - H(t). Zero at
    p = t, and its gradient in the logit is p - t, as for BCE on a label."""
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    tc = tf.clip_by_value(t, eps, 1.0 - eps)
    return (t * (tf.math.log(tc) - tf.math.log(p))
            + (1.0 - t) * (tf.math.log(1.0 - tc) - tf.math.log(1.0 - p)))


class LayerwiseAdamW(tf.keras.optimizers.AdamW):
    """AdamW whose step differs per variable: the learning rate, and with it
    the decoupled weight decay (lr x decay x weight), is multiplied by
    ``lr_multiplier(variable)``. Adam's moment estimates don't depend on the
    learning rate, so a variable with multiplier m moves exactly m times as far
    as it would under plain AdamW from the same state.

    Keras builds each variable's update separately, reading ``learning_rate``
    while it does; the multiplier is switched in for the length of that
    variable's update."""

    _lr_scale = 1.0

    def __init__(self, *args, lr_multiplier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr_multiplier = lr_multiplier or (lambda variable: 1.0)

    @property
    def learning_rate(self):
        lr = tf.keras.optimizers.AdamW.learning_rate.fget(self)
        return lr if self._lr_scale == 1.0 else lr * self._lr_scale

    @learning_rate.setter
    def learning_rate(self, value):
        tf.keras.optimizers.AdamW.learning_rate.fset(self, value)

    def update_step(self, gradient, variable):
        self._lr_scale = float(self._lr_multiplier(variable))
        try:
            return super().update_step(gradient, variable)
        finally:
            self._lr_scale = 1.0

    def _apply_weight_decay(self, variables):
        # As tf_keras's Optimizer._apply_weight_decay, with the variable's rate.
        if self.weight_decay is None:
            return

        def distributed_apply_weight_decay(distribution, variables, **kwargs):
            def weight_decay_fn(variable):
                if self._use_weight_decay(variable):
                    lr = tf.cast(self.learning_rate, variable.dtype) * float(
                        self._lr_multiplier(variable))
                    wd = tf.cast(self.weight_decay, variable.dtype)
                    variable.assign_sub(variable * wd * lr)

            for variable in variables:
                distribution.extended.update(variable, weight_decay_fn, group=False)

        tf.__internal__.distribute.interim.maybe_merge_call(
            distributed_apply_weight_decay, self._distribution_strategy, variables)


# Variables left out of weight decay by --no-decay-bias-norm (regular
# expressions searched in the variable name): biases and LayerNorm scales and
# shifts, as in BERT's and ELECTRA's own fine-tuning code.
NO_DECAY_NAMES = (r"/bias", r"LayerNorm", r"layer_norm")


def layerwise_lr_groups(model, decay: float) -> Dict[int, Tuple[str, float]]:
    """{variable id: (group, learning-rate multiplier)} for every trainable
    variable of ``model``, following ELECTRA's layer-wise learning-rate decay
    (Clark et al., 2020; ``_get_layer_lrs`` in its finetune/optimization.py).
    With L encoder layers: the task layers (everything outside the encoder)
    train at the full rate, encoder layer i (0 = bottom) at decay^(L + 1 - i),
    the embeddings at decay^(L + 2). Any other encoder variable (BERT's
    pooler) sits just above the top layer, at decay^1."""
    encoder = model.encoder
    num_layers = int(encoder.config.num_hidden_layers)
    encoder_ids = {v._unique_id for v in encoder.trainable_variables}
    groups: Dict[int, Tuple[str, float]] = {}
    for v in model.trainable_variables:
        if v._unique_id not in encoder_ids:
            groups[v._unique_id] = ("task layers", 1.0)
            continue
        layer = re.search(r"/layer_\._(\d+)/", v.name)
        if layer:
            i = int(layer.group(1))
            if i >= num_layers:
                raise ValueError(f"{v.name}: layer {i} of a {num_layers}-layer encoder")
            groups[v._unique_id] = (f"encoder layer {i}", decay ** (num_layers + 1 - i))
        elif "/embeddings" in v.name:
            groups[v._unique_id] = ("embeddings", decay ** (num_layers + 2))
        elif "/encoder/" in v.name:
            raise ValueError(f"Can't place encoder variable {v.name} in a layer")
        else:
            groups[v._unique_id] = ("encoder, above the layers", decay)
    found = {g for g, _ in groups.values() if g.startswith("encoder layer")}
    if len(found) != num_layers:
        raise ValueError(f"Found variables for {len(found)} of {num_layers} encoder layers")
    return groups


def build_finetune_optimizer(model, learning_rate, weight_decay: float, *,
                             layer_lr_decay: float = 1.0, clipnorm: float = 0.0,
                             decay_bias_norm: bool = True) -> LayerwiseAdamW:
    """AdamW with the optional fine-tuning practices of --layer-lr-decay,
    --clipnorm and --no-decay-bias-norm. The per-variable rates are worked out
    at the first training step, when every model variable exists."""
    if not 0 < layer_lr_decay <= 1:
        raise ValueError("layer_lr_decay must be in (0, 1]")
    groups: Dict[int, Tuple[str, float]] = {}

    def multiplier(variable) -> float:
        if layer_lr_decay == 1.0:
            return 1.0
        if not groups:
            groups.update(layerwise_lr_groups(model, layer_lr_decay))
            count: Dict[Tuple[str, float], int] = {}
            for key in groups.values():
                count[key] = count.get(key, 0) + 1
            print(f"Layer-wise learning rates (decay {layer_lr_decay:g}):")
            for (group, m), n in sorted(count.items(), key=lambda kv: kv[0][1]):
                print(f"  x{m:.4f}  {group} ({n} variables)", flush=True)
        return groups[variable._unique_id][1]

    optimizer = LayerwiseAdamW(learning_rate=learning_rate, weight_decay=weight_decay,
                               global_clipnorm=clipnorm or None, lr_multiplier=multiplier)
    if not decay_bias_norm:
        optimizer.exclude_from_weight_decay(var_names=list(NO_DECAY_NAMES))
    print(f"Optimizer: AdamW, layer-wise lr decay {layer_lr_decay:g}, global gradient clip "
          f"{clipnorm or 'off'}, weight decay on biases and LayerNorm "
          f"{'on' if decay_bias_norm else 'off'}", flush=True)
    return optimizer


class PooledOutputLossTrainer(Trainer):
    """Trainer whose loss sees a pooled output instead of its constituent heads.

    The paper (Sec. 4.1) max-pools the 18 propaganda heads into one binary
    output and describes the network output as 10-dimensional, so the loss
    acts on that pooled output. The stock library loss instead trains each
    constituent head on its own label. Here the constituents are dropped from
    the loss and replaced by one term on max(constituent predictions) against
    max(constituent labels); that is the same tensor as the model's
    ``propaganda_max_pool`` output. The pooled term keeps its task-type's
    ν_k and counts as a single task (|T_k| = 1). Every other task is
    weighted exactly as in ``Trainer._build_loss_function``. Binary tasks only.

    ``pool_strategy`` ('max', 'noisy_or' or 'lse', see
    ``multitask.models.hierarchical.pool_probabilities``) pools the
    constituent predictions; it must match the model's pool config, since the
    model's pooled output is what gets scored. The pooled label is always the
    max of the constituent labels ("any technique present").

    ``technique_loss_weight`` > 0 adds that weight x the mean over the
    constituent heads of each head's class-weighted BCE on its own label, to
    the pooled term (so the balancers still see one propaganda task). Each
    head's term is a weighted mean (divided by the sum of its weights over the
    labelled rows, as PyTorch's weighted cross-entropy does): the rarest
    techniques have class weights in the hundreds, and dividing by the row
    count would let one positive dominate a batch. ``technique_class_weights``
    gives the per-head weights, in ``pooled_tasks`` order.

    ``task_loss_options`` maps a non-pooled task to ``(logit_shift, focal_gamma)``.
    A non-zero shift trains on ``z + logit_shift`` (logit adjustment; predictions
    stay on the raw ``z``), and a non-zero gamma multiplies the term by
    ``(1 - p_correct) ** gamma`` (focal loss).
    """

    def __init__(self, *args, pooled_tasks: Sequence[str],
                 pooled_class_weights: Dict[int, float],
                 task_loss_options: Optional[Dict[str, Tuple[float, float]]] = None,
                 pool_strategy: str = "max", technique_loss_weight: float = 0.0,
                 technique_class_weights: Optional[Sequence[Dict[int, float]]] = None,
                 finetune: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Keyword arguments of build_finetune_optimizer; None = the library's optimizer.
        self.finetune = dict(finetune) if finetune else None
        self.pooled_tasks = list(pooled_tasks)
        self.pooled_class_weights = pooled_class_weights
        self.task_loss_options = task_loss_options or {}
        if pool_strategy not in PROPAGANDA_POOLS:
            raise ValueError(f"pool_strategy must be one of {PROPAGANDA_POOLS}")
        if technique_loss_weight < 0:
            raise ValueError("technique_loss_weight must be >= 0")
        if technique_loss_weight and (technique_class_weights is None
                                      or len(technique_class_weights) != len(self.pooled_tasks)):
            raise ValueError("technique_loss_weight needs one class-weight dict per pooled task")
        self.pool_strategy = pool_strategy
        self.technique_loss_weight = float(technique_loss_weight)
        self.technique_class_weights = (
            [(float(w.get(0, 1.0)), float(w.get(1, 1.0))) for w in technique_class_weights]
            if technique_class_weights is not None else None)

    def _build_optimizer(self, steps_per_epoch):
        if not self.finetune:
            return super()._build_optimizer(steps_per_epoch)
        if self.config.optimizer_type.lower() != "adamw":
            raise ValueError("The fine-tuning optimizer options need optimizer_type 'adamw'")
        return build_finetune_optimizer(
            self.model, self._build_lr_schedule(steps_per_epoch),
            self.config.weight_decay_rate, **self.finetune)

    def _build_loss_function(self, num_tasks, class_weights, task_weights,
                             task_type_mapping=None, mask_value=None):
        if mask_value is None:
            mask_value = self.mask_value
        names = list(self.model.task_names)
        types = self._resolve_task_type_mapping(task_type_mapping)
        pooled = set(self.pooled_tasks)

        # |T_k| without the pooled constituents; ν_k is per task-type, so any
        # constituent carries the pooled term's ν_k.
        type_sizes: Dict[str, int] = {}
        for n in names:
            if n not in pooled:
                type_sizes[types[n]] = type_sizes.get(types[n], 0) + 1
        terms = [
            (n, float(class_weights[i].get(0, 1.0)), float(class_weights[i].get(1, 1.0)),
             float(task_weights[i]) / type_sizes[types[n]],
             *self.task_loss_options.get(n, (0.0, 0.0)))
            for i, n in enumerate(names) if n not in pooled
        ]
        # No pooled tasks (a --tasks subset without propaganda): plain terms only.
        has_pool = bool(self.pooled_tasks)
        if has_pool:
            pool_factor = float(task_weights[names.index(self.pooled_tasks[0])])
            pw0 = float(self.pooled_class_weights[0])
            pw1 = float(self.pooled_class_weights[1])
            print(f"Pooled loss over {len(self.pooled_tasks)} heads ({self.pool_strategy}): "
                  f"class weights {{0: {pw0:.4f}, 1: {pw1:.4f}}}, nu_k={pool_factor:.6f}, |T_k|=1")
            if self.technique_loss_weight:
                print(f"  + {self.technique_loss_weight:g} x mean per-technique weighted BCE "
                      f"over {len(self.pooled_tasks)} heads")
        for n, w0, w1, f, shift, gamma in terms:
            extra = (f", logit shift={shift:+.4f}" if shift else "") + \
                    (f", focal gamma={gamma:g}" if gamma else "")
            print(f"  {n}: class weights {{0: {w0:.4f}, 1: {w1:.4f}}}, "
                  f"nu_k/|T_k|={f:.6f}{extra}")

        def term(y_true, y_pred, w0, w1, factor, shift=0.0, gamma=0.0, soft=None):
            """``soft`` = (target [B], on scalar 0/1): when on, each labelled
            row's BCE on its label is replaced by the binary KL from the soft
            target (self-distillation); rows, class weights and scale stay."""
            y = tf.reshape(tf.cast(y_true, tf.float32), [-1])
            p = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
            mask = tf.cast(tf.not_equal(y, mask_value), tf.float32)
            y = tf.where(mask > 0, y, tf.zeros_like(y))
            if shift or gamma:
                # The heads end in a sigmoid, so recover the raw score z from p.
                # Clipping at 1e-7 matches what Keras BCE does to p anyway.
                p = tf.clip_by_value(p, 1e-7, 1.0 - 1e-7)
                z = tf.math.log(p) - tf.math.log1p(-p) + shift
                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)
                if gamma:
                    p_adj = tf.sigmoid(z)
                    p_correct = tf.where(tf.equal(y, 1.0), p_adj, 1.0 - p_adj)
                    bce = bce * tf.pow(1.0 - p_correct, gamma)
            else:
                bce = tf.reshape(tf.keras.losses.binary_crossentropy(
                    tf.reshape(y, [-1, 1]), tf.reshape(p, [-1, 1])), [-1])
            if soft is not None:
                if shift or gamma:
                    raise ValueError("self-distillation does not support logit shift / focal loss")
                t = tf.reshape(tf.cast(soft[0], tf.float32), [-1])
                on = tf.cast(soft[1], tf.float32)
                bce = on * binary_kl(t, p) + (1.0 - on) * bce
            w = tf.where(tf.equal(y, 0.0), w0, w1)
            return tf.reduce_sum(bce * w * mask * factor), tf.reduce_sum(mask)

        def stack(d):
            return tf.stack(
                [tf.reshape(tf.cast(d[t], tf.float32), [-1]) for t in self.pooled_tasks], axis=1)

        def pool_labels(d):
            return tf.reduce_max(stack(d), axis=1)

        def pool(d):
            return pool_probabilities(stack(d), self.pool_strategy, axis=1)

        tech_w = self.technique_class_weights
        tech_lambda = self.technique_loss_weight

        def technique_loss(y_true_dict, y_pred_dict):
            """Mean over the constituent heads of each head's weighted-mean
            BCE on its own label; a head with no labelled row adds 0 and is
            left out of the mean."""
            total, heads = 0.0, 0.0
            for t, (w0, w1) in zip(self.pooled_tasks, tech_w):
                y = tf.reshape(tf.cast(y_true_dict[t], tf.float32), [-1])
                p = tf.reshape(tf.cast(y_pred_dict[t], tf.float32), [-1])
                mask = tf.cast(tf.not_equal(y, mask_value), tf.float32)
                y = tf.where(mask > 0, y, tf.zeros_like(y))
                bce = tf.reshape(tf.keras.losses.binary_crossentropy(
                    tf.reshape(y, [-1, 1]), tf.reshape(p, [-1, 1])), [-1])
                w = tf.where(tf.equal(y, 0.0), w0, w1) * mask
                wsum = tf.reduce_sum(w)
                total += tf.math.divide_no_nan(tf.reduce_sum(bce * w), wsum)
                heads += tf.cast(wsum > 0, tf.float32)
            return tf.math.divide_no_nan(total, heads)

        def pooled_term(y_true_dict, y_pred_dict, factor, soft=None):
            """(sum, labelled rows) of the pooled propaganda term, with the
            per-technique loss folded in as a mean scaled to the same rows."""
            s, c = term(pool_labels(y_true_dict), pool(y_pred_dict), pw0, pw1, factor,
                        soft=soft)
            if tech_lambda:
                s += tech_lambda * factor * technique_loss(y_true_dict, y_pred_dict) * c
            return s, c

        def pooled_loss(y_true_dict, y_pred_dict, soft=None):
            """``soft`` (self-distillation) maps an output name (a task, or
            propaganda_max_pool for the pool) to (target [B], on)."""
            soft = soft or {}
            total, n_valid = 0.0, 0.0
            for n, w0, w1, f, shift, gamma in terms:
                s, c = term(y_true_dict[n], y_pred_dict[n], w0, w1, f, shift, gamma,
                            soft=soft.get(n))
                total += s
                n_valid += c
            if has_pool:
                # mask_value (-1) is below both classes, so the max is masked only
                # when every constituent label is masked.
                s, c = pooled_term(y_true_dict, y_pred_dict, pool_factor,
                                   soft=soft.get("propaganda_max_pool"))
                total += s
                n_valid += c
            return total / (n_valid + 1e-7)

        def task_losses(y_true_dict, y_pred_dict):
            """Per-task class-weighted mean loss, without nu_k or 1/|T_k|, for
            the loss balancers and the gradient probe. Returns (losses [T],
            has_labels [T]); the pooled propaganda output, if any, is task 0."""
            parts = [pooled_term(y_true_dict, y_pred_dict, 1.0)] if has_pool else []
            parts += [term(y_true_dict[n], y_pred_dict[n], w0, w1, 1.0, shift, gamma)
                      for n, w0, w1, _, shift, gamma in terms]
            sums = tf.stack([s for s, _ in parts])
            counts = tf.stack([c for _, c in parts])
            return sums / tf.maximum(counts, 1.0), tf.cast(counts > 0, tf.float32)

        self.model._mtl_task_losses_fn = task_losses
        self.model._mtl_pooled_tasks = list(self.pooled_tasks)
        self.model._mtl_task_loss_names = (
            (["propaganda_max_pool"] if has_pool else []) + [t[0] for t in terms])
        return pooled_loss


# ---------------------------------------------------------------------------
# Adaptive task balancing: FAMO and DB-MTL
# ---------------------------------------------------------------------------

class FAMO:
    """Fast Adaptive Multitask Optimization (Liu et al., NeurIPS 2023).

    Task weights are z = softmax(w). The model step minimises
    sum_t z_t log l_t / c with c = sum_t z_t / l_t held fixed. After the step
    the same batch is scored again and w moves towards the tasks whose loss
    fell least. ``w_lr`` and ``gamma`` (Adam weight decay on w) are the
    authors' defaults.
    """

    def __init__(self, num_tasks: int, w_lr: float = 0.025, gamma: float = 0.01,
                 eps: float = 1e-8):
        self.w_lr, self.gamma, self.eps = w_lr, gamma, eps
        self.w = tf.Variable(tf.zeros([num_tasks]), trainable=False, name="famo_w")
        # Adam state for w, kept by hand so no optimizer is tied to the model.
        self.m = tf.Variable(tf.zeros([num_tasks]), trainable=False, name="famo_m")
        self.v = tf.Variable(tf.zeros([num_tasks]), trainable=False, name="famo_v")
        self.t = tf.Variable(0.0, trainable=False, name="famo_t")

    def objective(self, losses, has_labels):
        z = tf.nn.softmax(self.w) * has_labels
        d = losses + self.eps
        c = tf.stop_gradient(tf.reduce_sum(z / d))
        return tf.reduce_sum(z * tf.math.log(d)) / c

    def update(self, prev_losses, new_losses, has_labels):
        delta = (tf.math.log(prev_losses + self.eps)
                 - tf.math.log(new_losses + self.eps)) * has_labels
        z = tf.nn.softmax(self.w)
        # Gradient of z . delta with respect to w, plus Adam-style weight decay.
        g = z * (delta - tf.reduce_sum(z * delta)) + self.gamma * self.w
        self.t.assign_add(1.0)
        self.m.assign(0.9 * self.m + 0.1 * g)
        self.v.assign(0.999 * self.v + 0.001 * tf.square(g))
        m_hat = self.m / (1.0 - tf.pow(0.9, self.t))
        v_hat = self.v / (1.0 - tf.pow(0.999, self.t))
        self.w.assign_sub(self.w_lr * m_hat / (tf.sqrt(v_hat) + 1e-8))


class DBMTL:
    """Dual-Balancing Multi-Task Learning (Lin et al., 2023).

    Loss scale: each task trains on log(l_t). Gradient scale: each task's
    gradient on the shared parameters is kept as an exponential moving average
    (``beta``), and every average is rescaled to the largest one's norm before
    they are summed. Task-specific parameters get the plain sum of log-loss
    gradients, as in LibMTL.
    """

    def __init__(self, shared_vars, num_tasks: int, beta: float = 0.9, eps: float = 1e-8):
        self.shared_vars = list(shared_vars)
        self.beta, self.eps = beta, eps
        self.shapes = [v.shape for v in self.shared_vars]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.buffer = tf.Variable(tf.zeros([num_tasks, sum(self.sizes)]),
                                  trainable=False, name="dbmtl_grad_ema")

    def shared_gradients(self, tape, log_losses, has_labels):
        """``log_losses`` is a list of per-task scalars computed inside ``tape``."""
        rows = []
        for log_loss in log_losses:
            grads = tape.gradient(log_loss, self.shared_vars,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
            rows.append(tf.concat(
                [tf.reshape(tf.convert_to_tensor(g), [-1]) for g in grads], axis=0))
        has = has_labels[:, None]
        # A task with no labels in this batch keeps its previous average.
        ema = has * (self.beta * self.buffer + (1.0 - self.beta) * tf.stack(rows)) \
            + (1.0 - has) * self.buffer
        self.buffer.assign(ema)
        norms = tf.norm(ema, axis=1)
        alpha = tf.reduce_max(norms) / (norms + self.eps)
        flat = tf.reduce_sum(alpha[:, None] * ema, axis=0)
        return [tf.reshape(g, s) for g, s in zip(tf.split(flat, self.sizes), self.shapes)]


def shared_variables(model) -> List[tf.Variable]:
    """Everything outside the branch, output and masked-word head layers: the
    encoder and the shared dense layers."""
    own_layers = [layer for level in model.branch_layers for branch in level for layer in branch]
    own_layers += list(model.output_layers.values())
    if getattr(model, "mlm_head", None) is not None:  # only the masked-word loss reaches it
        own_layers.append(model.mlm_head)
    own = {id(v) for layer in own_layers for v in layer.trainable_variables}
    shared = [v for v in model.trainable_variables if id(v) not in own]
    if not shared or len(shared) == len(model.trainable_variables):
        raise RuntimeError("Could not split shared from task parameters")
    return shared


class BalancedMultiTaskModel(MultiTaskModel):
    """MultiTaskModel whose training step weights the per-task losses with
    FAMO or DB-MTL (``_balancer``, set by BalancedLossTrainer) instead of the
    fixed nu_k / |T_k| weighting. The reported and validation loss is still
    the fixed weighted loss, so early stopping is unchanged."""

    def train_step(self, data):
        x, y = data[0], data[1]
        bal = self._balancer
        names = self.task_names
        variables = self.trainable_variables
        with tf.GradientTape(persistent=isinstance(bal, DBMTL)) as tape:
            y_pred = self(x, training=True)
            y_pred = {k: y_pred[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            if isinstance(bal, FAMO):
                objective = bal.objective(losses, has_labels)
            else:
                # Split per task inside the tape: ops outside it get no gradient.
                log_losses = tf.unstack(tf.math.log(losses + bal.eps))
                own_objective = tf.add_n(
                    [l * h for l, h in zip(log_losses, tf.unstack(has_labels))])

        if isinstance(bal, FAMO):
            grads = tape.gradient(objective, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            new_pred = self(x, training=True)
            new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
            bal.update(losses, new_losses, has_labels)
        else:
            shared_ids = {id(v) for v in bal.shared_vars}
            own = [v for v in variables if id(v) not in shared_ids]
            own_grads = tape.gradient(own_objective, own)
            shared_grads = bal.shared_gradients(tape, log_losses, has_labels)
            del tape
            by_id = {id(v): g for v, g in zip(bal.shared_vars, shared_grads)}
            by_id.update({id(v): g for v, g in zip(own, own_grads)})
            self.optimizer.apply_gradients([(by_id[id(v)], v) for v in variables])

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        return {'loss': loss}


class BalancedLossTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a BalancedMultiTaskModel: builds the model on
    one batch, then attaches a FAMO or DB-MTL balancer over the 10 outputs
    (pooled propaganda + the non-pooled tasks)."""

    def __init__(self, *args, balancer: str, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, BalancedMultiTaskModel):
            raise TypeError("BalancedLossTrainer needs a BalancedMultiTaskModel")
        self.balancer_name = balancer

    def fit(self, train_dataset, *args, **kwargs):
        self.model(next(iter(train_dataset))[0], training=False)  # create all variables
        num_tasks = len(self.model.task_names) - len(self.pooled_tasks) + 1
        if self.balancer_name == "famo":
            bal = FAMO(num_tasks)
        elif self.balancer_name == "dbmtl":
            shared = shared_variables(self.model)
            print(f"DB-MTL: {len(shared)} shared variables, "
                  f"{sum(int(np.prod(v.shape)) for v in shared):,} parameters")
            bal = DBMTL(shared, num_tasks)
        else:
            raise ValueError(f"Unknown balancer {self.balancer_name!r}")
        print(f"Loss balancer: {self.balancer_name} over {num_tasks} tasks")
        self.model._balancer = bal
        return super().fit(train_dataset, *args, **kwargs)


# ---------------------------------------------------------------------------
# Masked-word (MLM) auxiliary loss
# ---------------------------------------------------------------------------

# BERT's uncased vocabulary, which the distilbert-base-uncased tokenizer in
# load_splits shares. Ids below 999 are [PAD], [unusedN] and the special
# tokens, so random replacement words are drawn from 999 up.
PAD_ID, CLS_ID, SEP_ID, MASK_ID, FIRST_WORD_ID = 0, 101, 102, 103, 999
MLM_IGNORE = -100


def mask_tokens(input_ids, attention_mask, vocab_size: int, mask_prob: float = 0.15):
    """BERT's masking rule. Picks ~``mask_prob`` of the real word tokens (never
    padding, [CLS] or [SEP]); of those, 80% become [MASK], 10% a random word
    and 10% stay as they are. Returns (masked ids, answers), where answers hold
    the original id at picked positions and MLM_IGNORE everywhere else."""
    ids = tf.convert_to_tensor(input_ids)
    candidate = (tf.cast(attention_mask, tf.bool) & tf.not_equal(ids, PAD_ID)
                 & tf.not_equal(ids, CLS_ID) & tf.not_equal(ids, SEP_ID))
    picked = candidate & (tf.random.uniform(tf.shape(ids)) < mask_prob)
    r = tf.random.uniform(tf.shape(ids))
    random_words = tf.random.uniform(tf.shape(ids), FIRST_WORD_ID, vocab_size, dtype=ids.dtype)
    masked = tf.where(picked & (r < 0.8), tf.cast(MASK_ID, ids.dtype), ids)
    masked = tf.where(picked & (r >= 0.8) & (r < 0.9), random_words, masked)
    answers = tf.where(picked, ids, tf.cast(MLM_IGNORE, ids.dtype))
    return masked, answers


class MLMHead(tf.keras.layers.Layer):
    """BERT's masked-word head: dense, activation, LayerNorm, then scores
    against the encoder's word-embedding table plus a per-word bias. The table
    is passed in on each call rather than stored, so it stays tied to (and is
    only tracked by) the encoder. Takes [positions, hidden]: LayerNormalization
    fixes its axis at build time, so the input rank must not change."""

    def __init__(self, hidden_size: int, vocab_size: int, activation="gelu",
                 layer_norm_eps: float = 1e-12, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")
        self.activation = tf.keras.activations.get(activation)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps, name="LayerNorm")

    def build(self, input_shape):
        self.bias = self.add_weight(name="bias", shape=[self.vocab_size], initializer="zeros")
        super().build(input_shape)

    def call(self, hidden, word_embeddings):
        h = self.layer_norm(self.activation(self.dense(hidden)))
        return tf.matmul(h, word_embeddings, transpose_b=True) + self.bias


def load_pretrained_mlm_head(model_name: str, from_pt: bool) -> MLMHead:
    """An MLMHead holding the checkpoint's own pretrained head weights."""
    from transformers import TFBertForMaskedLM
    from transformers.activations_tf import get_tf_activation

    try:
        src, info = TFBertForMaskedLM.from_pretrained(
            model_name, from_pt=from_pt, use_safetensors=False, output_loading_info=True)
    except TypeError:
        src, info = TFBertForMaskedLM.from_pretrained(
            model_name, from_pt=from_pt, output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "mlm" in k or "predictions" in k]
    if missing:
        raise RuntimeError(f"{model_name} has no pretrained MLM head weights: {missing}")
    cfg = src.config
    head = MLMHead(cfg.hidden_size, cfg.vocab_size, get_tf_activation(cfg.hidden_act),
                   cfg.layer_norm_eps, name="mlm_head")
    head(tf.zeros([1, cfg.hidden_size]), tf.zeros([cfg.vocab_size, cfg.hidden_size]))
    p = src.mlm.predictions
    head.dense.set_weights(p.transform.dense.get_weights())
    head.layer_norm.set_weights(p.transform.LayerNorm.get_weights())
    head.bias.assign(p.bias)
    return head


class NoMLMHead:
    """Stands in for the masked-word head on an encoder whose checkpoint has
    none (ELECTRA's discriminator). Only for ``mlm_lambda`` 0, where the
    masked-word loss has no weight anyway: masking still works (it only needs
    the vocabulary size), and the masked-word loss and accuracy are logged
    as 0."""

    trainable_variables: Tuple = ()

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size


class MaskedLMMultiTaskModel(MultiTaskModel):
    """MultiTaskModel trained on supervised loss + ``mlm_lambda`` x masked-word
    loss. The masked-word pass runs the encoder a second time on a masked copy
    of the batch and never reaches the heads. The validation loss, and so
    early stopping, is the supervised loss only; ``mlm_loss`` / ``mlm_acc``
    are logged for both splits. Call ``attach_mlm_head`` before training."""

    def attach_mlm_head(self, head: MLMHead, mlm_lambda: float, mask_prob: float = 0.15):
        self.mlm_head = head
        self.mlm_lambda = float(mlm_lambda)
        self.mlm_mask_prob = mask_prob
        self.mlm_loss_tracker = tf.keras.metrics.Mean(name="mlm_loss")
        self.mlm_acc_tracker = tf.keras.metrics.Mean(name="mlm_acc")

    def mask_batch(self, x):
        return mask_tokens(x["input_ids"], x["attention_mask"],
                           self.mlm_head.vocab_size, self.mlm_mask_prob)

    def mlm_loss(self, x, training, masking=None):
        """Mean cross-entropy and accuracy over this batch's masked positions.
        ``masking`` = (masked ids, answers) from ``mask_batch``; a new random
        masking is drawn when it's None."""
        masked, answers = masking if masking is not None else self.mask_batch(x)
        hidden = self.encoder({"input_ids": masked, "attention_mask": x["attention_mask"]},
                              training=training).last_hidden_state
        return self.mlm_loss_from_hidden(hidden, answers)

    def mlm_loss_from_hidden(self, hidden, answers):
        """``mlm_loss`` for encoder outputs already computed on the masked ids."""
        if isinstance(self.mlm_head, NoMLMHead):
            return tf.constant(0.0), tf.constant(0.0)
        # Score only the picked positions: the full [B, L, vocab] tensor is huge.
        picked = tf.not_equal(answers, MLM_IGNORE)
        logits = self.mlm_head(tf.boolean_mask(hidden, picked),
                               self.encoder.get_input_embeddings().weight)
        targets = tf.boolean_mask(answers, picked)
        n = tf.maximum(tf.cast(tf.size(targets), tf.float32), 1.0)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits)) / n
        hits = tf.equal(tf.argmax(logits, axis=-1, output_type=targets.dtype), targets)
        return loss, tf.reduce_sum(tf.cast(hits, tf.float32)) / n

    def train_objective(self, x, y, training=True):
        """(total, supervised, masked-word loss, masked-word accuracy)."""
        y_pred = self(x, training=training)
        sup_loss = self._mtl_loss_fn(y, {k: y_pred[k] for k in self.task_names})
        mlm_loss, mlm_acc = self.mlm_loss(x, training)
        return sup_loss + self.mlm_lambda * mlm_loss, sup_loss, mlm_loss, mlm_acc

    def train_step(self, data):
        x, y = data[0], data[1]
        with tf.GradientTape() as tape:
            total, sup_loss, mlm_loss, mlm_acc = self.train_objective(x, y)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(sup_loss)
                break
        self.mlm_loss_tracker.update_state(mlm_loss)
        self.mlm_acc_tracker.update_state(mlm_acc)
        return {'loss': sup_loss, 'mlm_loss': self.mlm_loss_tracker.result(),
                'mlm_acc': self.mlm_acc_tracker.result()}

    def test_step(self, data):
        mlm_loss, mlm_acc = self.mlm_loss(data[0], training=False)
        self.mlm_loss_tracker.update_state(mlm_loss)
        self.mlm_acc_tracker.update_state(mlm_acc)
        return super().test_step(data)


class MaskedLMTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a MaskedLMMultiTaskModel: builds the model on
    one batch, then attaches the encoder checkpoint's pretrained MLM head."""

    def __init__(self, *args, mlm_lambda: float, mask_prob: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, MaskedLMMultiTaskModel):
            raise TypeError("MaskedLMTrainer needs a MaskedLMMultiTaskModel")
        self.mlm_lambda, self.mask_prob = mlm_lambda, mask_prob

    def prepare(self, train_dataset):
        """Create all variables, then attach the pretrained head."""
        self.model(next(iter(train_dataset))[0], training=False)
        ecfg = self.model.encoder_config
        enc_cfg = self.model.encoder.config
        if enc_cfg.model_type == "bert":
            head = load_pretrained_mlm_head(ecfg.encoder_identifier, ecfg.from_pt)
            source = f"pretrained head from {ecfg.encoder_identifier}"
        elif self.mlm_lambda == 0:
            head = NoMLMHead(enc_cfg.vocab_size)
            source = f"no head ({enc_cfg.model_type} checkpoint has none; loss logged as 0)"
        else:
            raise ValueError(f"A masked-word loss with weight {self.mlm_lambda:g} needs a BERT "
                             f"checkpoint with its pretrained head, not {enc_cfg.model_type}")
        self.model.attach_mlm_head(head, self.mlm_lambda, self.mask_prob)
        print(f"Masked-word loss: lambda={self.mlm_lambda:g}, mask prob={self.mask_prob:g}, "
              f"{source}")

    def fit(self, train_dataset, *args, **kwargs):
        self.prepare(train_dataset)
        return super().fit(train_dataset, *args, **kwargs)


MLM_MODES = ("fixed", "task", "cosine")


class BalancedMaskedLMModel(MaskedLMMultiTaskModel):
    """Masked-word loss combined with a loss balancer or a gradient gate.
    ``mlm_mode`` (set by BalancedMaskedLMTrainer):

    - "fixed": FAMO / DB-MTL (``_balancer``) over the 10 task losses, plus
      ``mlm_lambda`` x masked-word loss, as in MaskedLMMultiTaskModel.
    - "task": the masked-word loss is an 11th task under the balancer, which
      sets its weight; ``mlm_lambda`` is unused.
    - "cosine": no balancer. Fixed supervised loss, plus ``mlm_lambda`` x the
      masked-word gradient on the shared parameters only on steps where it
      has positive cosine with the supervised gradient there (Du et al.,
      2018, "Adapting Auxiliary Losses Using Gradient Similarity"). The
      masked-word head always gets its gradient: it can't affect the tasks.

    As in BalancedMultiTaskModel, the reported and validation loss is the
    fixed weighted supervised loss, so early stopping is unchanged."""

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, mode, lam = self._balancer, self.mlm_mode, self.mlm_lambda
        names = self.task_names
        variables = self.trainable_variables
        masking = self.mask_batch(x)
        with tf.GradientTape(persistent=not isinstance(bal, FAMO)) as tape:
            y_pred = self(x, training=True)
            y_pred = {k: y_pred[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            mlm_loss, mlm_acc = self.mlm_loss(x, True, masking)
            if mode == "task":
                losses = tf.concat([losses, mlm_loss[None]], 0)
                has_labels = tf.concat([has_labels, tf.ones([1], has_labels.dtype)], 0)
            if isinstance(bal, FAMO):
                objective = bal.objective(losses, has_labels)
                if mode == "fixed":
                    objective += lam * mlm_loss
            elif isinstance(bal, DBMTL):
                # Split per task inside the tape: ops outside it get no gradient.
                log_losses = tf.unstack(tf.math.log(losses + bal.eps))
                own_objective = tf.add_n(
                    [l * h for l, h in zip(log_losses, tf.unstack(has_labels))])
                if mode == "fixed":
                    own_objective += lam * mlm_loss
            else:
                sup_loss = self._mtl_loss_fn(y, y_pred)

        logs = {}
        if isinstance(bal, FAMO):
            grads = tape.gradient(objective, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            new_pred = self(x, training=True)
            new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
            if mode == "task":
                new_mlm, _ = self.mlm_loss(x, True, masking)  # same masking as before the step
                new_losses = tf.concat([new_losses, new_mlm[None]], 0)
            bal.update(losses, new_losses, has_labels)
            if mode == "task":
                logs["mlm_weight"] = tf.nn.softmax(bal.w)[-1]
        else:
            shared = self._mlm_shared
            shared_ids = {id(v) for v in shared}
            own = [v for v in variables if id(v) not in shared_ids]
            zero = tf.UnconnectedGradients.ZERO
            if isinstance(bal, DBMTL):
                own_grads = tape.gradient(own_objective, own, unconnected_gradients=zero)
                shared_grads = bal.shared_gradients(tape, log_losses, has_labels)
                if mode == "fixed":
                    mlm_grads = tape.gradient(mlm_loss, shared, unconnected_gradients=zero)
                    shared_grads = [g + lam * tf.convert_to_tensor(m)
                                    for g, m in zip(shared_grads, mlm_grads)]
            else:
                head_ids = {id(v) for v in self.mlm_head.trainable_variables}
                head = [v for v in own if id(v) in head_ids]
                grads = tape.gradient(sup_loss, own + shared, unconnected_gradients=zero)
                own_grads, sup_grads = grads[:len(own)], grads[len(own):]
                grads = tape.gradient(mlm_loss, head + shared, unconnected_gradients=zero)
                head_grads, mlm_grads = iter(grads[:len(head)]), grads[len(head):]
                own_grads = [lam * next(head_grads) if id(v) in head_ids else g
                             for v, g in zip(own, own_grads)]

                def flat(gs):
                    return tf.concat([tf.reshape(tf.convert_to_tensor(g), [-1]) for g in gs], 0)
                a, b = flat(sup_grads), flat(mlm_grads)
                cos = tf.reduce_sum(a * b) / (tf.norm(a) * tf.norm(b) + 1e-12)
                gate = tf.cast(cos > 0, tf.float32)
                shared_grads = [tf.convert_to_tensor(g) + lam * gate * tf.convert_to_tensor(m)
                                for g, m in zip(sup_grads, mlm_grads)]
                logs["mlm_cos"], logs["mlm_gate"] = cos, gate
            del tape
            by_id = {id(v): g for v, g in zip(shared, shared_grads)}
            by_id.update({id(v): g for v, g in zip(own, own_grads)})
            self.optimizer.apply_gradients([(by_id[id(v)], v) for v in variables])

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        self.mlm_loss_tracker.update_state(mlm_loss)
        self.mlm_acc_tracker.update_state(mlm_acc)
        out = {'loss': loss, 'mlm_loss': self.mlm_loss_tracker.result(),
               'mlm_acc': self.mlm_acc_tracker.result()}
        for k, v in logs.items():
            self.mlm_extra_trackers[k].update_state(v)
            out[k] = self.mlm_extra_trackers[k].result()
        return out


def attach_mlm_balancing(model: BalancedMaskedLMModel, balancer: str, mlm_mode: str,
                         num_tasks: int) -> None:
    """Set up a BalancedMaskedLMModel whose head is attached. ``num_tasks``
    counts the supervised losses (10 with pooled propaganda)."""
    m = model
    m.mlm_mode = mlm_mode
    m._mlm_shared = shared_variables(m)  # after attach, so the head is excluded
    num_tasks += mlm_mode == "task"
    if balancer == "famo":
        m._balancer = FAMO(num_tasks)
    elif balancer == "dbmtl":
        m._balancer = DBMTL(m._mlm_shared, num_tasks)
    else:
        m._balancer = None
    extra = {"task": ["mlm_weight"] if balancer == "famo" else [],
             "cosine": ["mlm_cos", "mlm_gate"]}.get(mlm_mode, [])
    m.mlm_extra_trackers = {}
    for k in extra:  # attributes, so Keras resets them every epoch
        setattr(m, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        m.mlm_extra_trackers[k] = getattr(m, f"{k}_tracker")
    print(f"Masked-word mode {mlm_mode!r}, balancer {balancer or 'none'} "
          f"over {num_tasks} losses, {len(m._mlm_shared)} shared variables")


class BalancedMaskedLMTrainer(MaskedLMTrainer):
    """MaskedLMTrainer for a BalancedMaskedLMModel: attaches the pretrained
    head, then a FAMO or DB-MTL balancer over the 10 outputs (11 with the
    masked-word loss in "task" mode), or none in "cosine" mode."""

    def __init__(self, *args, balancer: str, mlm_mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, BalancedMaskedLMModel):
            raise TypeError("BalancedMaskedLMTrainer needs a BalancedMaskedLMModel")
        if mlm_mode not in MLM_MODES:
            raise ValueError(f"Unknown mlm_mode {mlm_mode!r}")
        if (mlm_mode == "cosine") == bool(balancer):
            raise ValueError("mlm_mode 'cosine' takes no balancer; 'fixed' and 'task' need one")
        self.balancer_name, self.mlm_mode = balancer, mlm_mode

    def fit(self, train_dataset, *args, **kwargs):
        self.prepare(train_dataset)
        num_tasks = len(self.model.task_names) - len(self.pooled_tasks) + 1
        attach_mlm_balancing(self.model, self.balancer_name, self.mlm_mode, num_tasks)
        return PooledOutputLossTrainer.fit(self, train_dataset, *args, **kwargs)


# ---------------------------------------------------------------------------
# Label-aware masking: the masked copy goes through the whole model
# ---------------------------------------------------------------------------

LABEL_AWARE_TRACKERS = ("masked_label_loss", "consistency_loss", "consistency_weight",
                        "consistency_kept")


def confident_cells(t, eligible, share: float, floor: float = 0.0):
    """Per head, the eligible cells whose full-sentence score is among the
    lowest or highest ``share`` of eligible scores in the batch (at least one
    each side). Ranking within the head keeps both sides equal in size, which
    a single global cut-off does not on class-weighted, imbalanced heads.

    With ``floor`` > 0 a kept cell must also score <= ``floor`` (low side) or
    >= 1 - ``floor`` (high side). Both sides shrink to the smaller count, so
    they stay equal, and a head with no confident cells on one side keeps
    none at all in that batch."""
    n = tf.reduce_sum(tf.cast(eligible, tf.int32))
    k = tf.maximum(1, tf.cast(tf.floor(share * tf.cast(n, tf.float32)), tf.int32))
    k = tf.minimum(k, tf.maximum(n, 1))
    if floor:
        n_low = tf.reduce_sum(tf.cast(eligible & (t <= floor), tf.int32))
        n_high = tf.reduce_sum(tf.cast(eligible & (t >= 1.0 - floor), tf.int32))
        k = tf.minimum(k, tf.minimum(n_low, n_high))
    low = tf.sort(tf.where(eligible, t, 2.0))  # ineligible cells sort last
    high = tf.sort(tf.where(eligible, t, -2.0), direction="DESCENDING")
    i = tf.maximum(k - 1, 0)
    keep = (t <= low[i]) | (t >= high[i])
    return eligible & keep & (n > 0) & (k > 0)


def check_consistency_filter(keep: float, floor: float) -> None:
    if not 0.0 <= keep <= 0.5:
        raise ValueError("consistency_keep is a share per side, in [0, 0.5]")
    if not 0.0 <= floor < 0.5:
        raise ValueError("consistency_floor is a score distance from 0 or 1, in [0, 0.5)")
    if floor and not keep:
        raise ValueError("consistency_floor needs consistency_keep")


class ConsistencyMixin:
    """The consistency term shared by LabelAwareMaskedModel and
    TeacherConsistencyModel. Reads ``consistency_keep``, ``consistency_floor``,
    ``consistency_skip`` and ``consistency_rampup_steps`` from the model."""

    def consistency_loss(self, target, student, y=None):
        """(mean binary KL(target answer || student answer) over the cells
        used, share of cells used). The target answers are held fixed.
        Without ``consistency_keep`` every row of the 10 outputs is used; ``y``
        is needed only for ``consistency_skip``."""
        eps = 1e-7
        keep_share = getattr(self, "consistency_keep", 0.0)
        floor = getattr(self, "consistency_floor", 0.0)
        skip = getattr(self, "consistency_skip", {}) if y is not None else {}
        total, used, cells = 0.0, 0.0, 0.0
        for k in self._mtl_task_loss_names:
            t = tf.clip_by_value(tf.stop_gradient(
                tf.reshape(tf.cast(target[k], tf.float32), [-1])), eps, 1.0 - eps)
            p = tf.clip_by_value(tf.reshape(tf.cast(student[k], tf.float32), [-1]), eps, 1.0 - eps)
            kl = t * tf.math.log(t / p) + (1.0 - t) * tf.math.log((1.0 - t) / (1.0 - p))
            keep = tf.ones_like(t, dtype=tf.bool)
            if keep_share:
                for task in skip.get(k, []):
                    keep &= tf.equal(tf.reshape(tf.cast(y[task], tf.float32), [-1]),
                                     self.config.mask_value)
                keep = confident_cells(t, keep, keep_share, floor)
            w = tf.cast(keep, tf.float32)
            total += tf.reduce_sum(kl * w)
            used += tf.reduce_sum(w)
            cells += tf.cast(tf.size(t), tf.float32)
        return total / tf.maximum(used, 1.0), used / cells

    def consistency_ramp(self):
        """0 at the first step, rising linearly to 1 after the ramp."""
        if not self.consistency_rampup_steps:
            return tf.constant(1.0)
        step = tf.cast(self.optimizer.iterations, tf.float32)
        return tf.minimum(1.0, step / float(self.consistency_rampup_steps))


class LabelAwareMaskedModel(ConsistencyMixin, BalancedMaskedLMModel):
    """FAMO over the 10 task losses plus ``mlm_lambda`` x masked-word loss
    (famo_fixed), plus the masked copy of the batch sent up through the shared,
    branch and output layers. The masked copy reuses the masked-word encoder
    pass, so the extra cost is the small dense layers. Two extra terms, both
    outside FAMO like the masked-word loss:

    - ``masked_label_weight`` x the fixed-weight supervised loss on the masked
      copy against the real labels, so labelled cells only.
    - ``consistency_weight`` x the binary KL divergence from the full-sentence
      answers (no gradient) to the masked-copy answers, over every row and the
      10 outputs (pooled propaganda counted once), labelled or not. KL rather
      than BCE so it is 0 when the answers match; the gradient is the same.
      The weight rises linearly from 0 over ``consistency_rampup_steps``.
      With ``consistency_keep`` > 0 only confident targets count: per head,
      the cells whose full-sentence score is in the batch's lowest or highest
      ``consistency_keep`` share (``confident_cells``), skipping rows that
      have a label for any task in ``consistency_skip[head]`` (used to keep
      Propaganda rows, with their rule-made IAC labels, out of the IAC
      heads' targets). ``consistency_floor`` > 0 also requires a kept score
      to be <= floor or >= 1 - floor (``confident_cells``). Targets stay soft
      scores; nothing is sharpened.

    As in BalancedMultiTaskModel, the reported and validation loss is the
    fixed weighted supervised loss on full sentences, so early stopping is
    unchanged. Call ``attach_label_aware`` after ``attach_mlm_balancing``."""

    def attach_label_aware(self, masked_label_weight: float, consistency_weight: float,
                           consistency_rampup_steps: int = 0, consistency_keep: float = 0.0,
                           consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                           consistency_floor: float = 0.0):
        if not isinstance(self._balancer, FAMO) or self.mlm_mode != "fixed":
            raise ValueError("Label-aware masking is built on famo_fixed: FAMO, mlm_mode 'fixed'")
        check_consistency_filter(consistency_keep, consistency_floor)
        self.masked_label_weight = float(masked_label_weight)
        self.consistency_weight = float(consistency_weight)
        self.consistency_rampup_steps = int(consistency_rampup_steps)
        self.consistency_keep = float(consistency_keep)
        self.consistency_floor = float(consistency_floor)
        self.consistency_skip = {k: list(v) for k, v in (consistency_skip or {}).items()}
        for k in LABEL_AWARE_TRACKERS:  # attributes, so Keras resets them every epoch
            setattr(self, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        print(f"Label-aware masking: masked-label weight {self.masked_label_weight:g}, "
              f"consistency weight {self.consistency_weight:g} "
              f"(ramp over {self.consistency_rampup_steps} steps), mask prob "
              f"{self.mlm_mask_prob:g}, confident share per side {self.consistency_keep:g}, "
              f"floor {self.consistency_floor:g}, "
              f"{len(self.consistency_skip)} heads skip rows labelled for other tasks")

    def masked_pass(self, x, training, masking):
        """One encoder pass on the masked ids, giving (masked-word loss,
        masked-word accuracy, model outputs on the masked copy)."""
        masked, answers = masking
        enc = self.encoder({"input_ids": masked, "attention_mask": x["attention_mask"]},
                           training=training)
        mlm_loss, mlm_acc = self.mlm_loss_from_hidden(enc.last_hidden_state, answers)
        # Same choice as MultiTaskModel._encode_input for HuggingFace encoders.
        pooled = getattr(enc, "pooler_output", None)  # ELECTRA's output has no such field
        rep = pooled if pooled is not None else enc.last_hidden_state[:, 0, :]
        return mlm_loss, mlm_acc, self.forward_from_encoding(rep, training=training)

    def label_aware_losses(self, x, y, clean, training, masking):
        """(masked-word loss, accuracy, masked-label loss, consistency loss,
        share of cells the consistency loss used)."""
        mlm_loss, mlm_acc, masked_out = self.masked_pass(x, training, masking)
        label_loss = self._mtl_loss_fn(y, {k: masked_out[k] for k in self.task_names})
        cons_loss, kept = self.consistency_loss(clean, masked_out, y)
        return mlm_loss, mlm_acc, label_loss, cons_loss, kept

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, names = self._balancer, self.task_names
        variables = self.trainable_variables
        masking = self.mask_batch(x)
        cons_weight = self.consistency_weight * self.consistency_ramp()
        with tf.GradientTape() as tape:
            clean = self(x, training=True)
            y_pred = {k: clean[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            mlm_loss, mlm_acc, label_loss, cons_loss, kept = self.label_aware_losses(
                x, y, clean, True, masking)
            objective = (bal.objective(losses, has_labels) + self.mlm_lambda * mlm_loss
                         + self.masked_label_weight * label_loss + cons_weight * cons_loss)
        grads = tape.gradient(objective, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        new_pred = self(x, training=True)
        new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
        bal.update(losses, new_losses, has_labels)

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        out = {'loss': loss}
        for tracker, v in ((self.mlm_loss_tracker, mlm_loss), (self.mlm_acc_tracker, mlm_acc),
                           (self.masked_label_loss_tracker, label_loss),
                           (self.consistency_loss_tracker, cons_loss),
                           (self.consistency_weight_tracker, cons_weight),
                           (self.consistency_kept_tracker, kept)):
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        return out

    def test_step(self, data):
        x, y = data[0], data[1]
        clean = self(x, training=False)
        mlm_loss, mlm_acc, label_loss, cons_loss, kept = self.label_aware_losses(
            x, y, clean, False, self.mask_batch(x))
        for tracker, v in ((self.mlm_loss_tracker, mlm_loss), (self.mlm_acc_tracker, mlm_acc),
                           (self.masked_label_loss_tracker, label_loss),
                           (self.consistency_loss_tracker, cons_loss),
                           (self.consistency_kept_tracker, kept)):
            tracker.update_state(v)
        return MultiTaskModel.test_step(self, data)


class LabelAwareMaskedTrainer(BalancedMaskedLMTrainer):
    """BalancedMaskedLMTrainer (FAMO, mlm_mode 'fixed') for a
    LabelAwareMaskedModel: also sets the two label-aware weights, with the
    consistency ramp sized in epochs of the train dataset."""

    def __init__(self, *args, masked_label_weight: float, consistency_weight: float,
                 consistency_rampup_epochs: float = 1.0, consistency_keep: float = 0.0,
                 consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                 consistency_floor: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, LabelAwareMaskedModel):
            raise TypeError("LabelAwareMaskedTrainer needs a LabelAwareMaskedModel")
        if self.balancer_name != "famo" or self.mlm_mode != "fixed":
            raise ValueError("Label-aware masking is built on famo_fixed: FAMO, mlm_mode 'fixed'")
        self.masked_label_weight = masked_label_weight
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.consistency_keep, self.consistency_skip = consistency_keep, consistency_skip
        self.consistency_floor = consistency_floor

    def fit(self, train_dataset, *args, **kwargs):
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("The consistency ramp needs a train dataset of known length")
        self.prepare(train_dataset)
        num_tasks = len(self.model.task_names) - len(self.pooled_tasks) + 1
        attach_mlm_balancing(self.model, self.balancer_name, self.mlm_mode, num_tasks)
        self.model.attach_label_aware(self.masked_label_weight, self.consistency_weight,
                                      round(self.consistency_rampup_epochs * steps),
                                      self.consistency_keep, self.consistency_skip,
                                      self.consistency_floor)
        return PooledOutputLossTrainer.fit(self, train_dataset, *args, **kwargs)


# ---------------------------------------------------------------------------
# Teacher consistency: dropout views and an EMA (mean) teacher, no masking
# ---------------------------------------------------------------------------

TEACHER_TRACKERS = ("consistency_loss", "consistency_weight", "consistency_kept",
                    "teacher_gap", "ema_decay")
CONSISTENCY_TEACHERS = ("current", "ema")


class _Untracked:
    """Holds the EMA teacher so Keras does not track it as part of the
    student: its weights must stay out of ``trainable_variables``, the
    optimizer and ``get_weights``."""

    def __init__(self, model):
        self.model = model


class TeacherConsistencyModel(ConsistencyMixin, BalancedMultiTaskModel):
    """FAMO over the 10 task losses plus ``consistency_weight`` x the binary
    KL divergence from a teacher's answers (held fixed, dropout off) to the
    student's answers, with the confidence filter of LabelAwareMaskedModel.
    No masking and no masked-word loss. Two switches:

    - teacher "current": the model itself with dropout off; "ema": a copy
      whose weights are an exponential moving average of the student's,
      decay min(``ema_decay``, 1 - 1/(step + 1)) as in Mean Teacher (Tarvainen
      & Valpola, 2017), also run with dropout off.
    - ``student_dropout``: True scores the supervised pass (encoder dropout
      on, so no extra pass); False runs a second student pass with dropout off.

    "current" with ``student_dropout`` False compares the model with itself,
    so the loss is always 0 and it is refused. ``teacher_gap`` logs the mean
    |teacher - student| answer over every cell of the 10 outputs, before the
    filter. As in BalancedMultiTaskModel, the reported and validation loss is
    the fixed weighted supervised loss, so early stopping is unchanged. Call
    ``attach_teacher`` after the balancer is set."""

    def attach_teacher(self, teacher: Optional[MultiTaskModel], student_dropout: bool,
                       consistency_weight: float, consistency_rampup_steps: int = 0,
                       consistency_keep: float = 0.0, consistency_floor: float = 0.0,
                       consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                       ema_decay: float = 0.99):
        if not isinstance(self._balancer, FAMO):
            raise ValueError("Teacher consistency is built on FAMO")
        if teacher is None and not student_dropout:
            raise ValueError("The current-weights teacher needs student dropout: "
                             "otherwise both answers are identical and the loss is 0")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        check_consistency_filter(consistency_keep, consistency_floor)
        if teacher is not None:
            pairs = list(zip(teacher.trainable_variables, self.trainable_variables))
            if len(teacher.trainable_variables) != len(self.trainable_variables) or any(
                    t.shape != s.shape for t, s in pairs):
                raise ValueError("The teacher's weights do not line up with the student's")
            for t, s in pairs:
                t.assign(s)
            self._teacher = _Untracked(teacher)
        self.student_dropout = bool(student_dropout)
        self.ema_decay = float(ema_decay)
        self.consistency_weight = float(consistency_weight)
        self.consistency_rampup_steps = int(consistency_rampup_steps)
        self.consistency_keep = float(consistency_keep)
        self.consistency_floor = float(consistency_floor)
        self.consistency_skip = {k: list(v) for k, v in (consistency_skip or {}).items()}
        for k in TEACHER_TRACKERS:  # attributes, so Keras resets them every epoch
            setattr(self, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        print(f"Teacher consistency: teacher {'ema' if teacher is not None else 'current'}"
              f"{f' (decay {self.ema_decay:g})' if teacher is not None else ''}, student "
              f"dropout {'on' if self.student_dropout else 'off'}, weight "
              f"{self.consistency_weight:g} (ramp over {self.consistency_rampup_steps} steps), "
              f"confident share per side {self.consistency_keep:g}, floor "
              f"{self.consistency_floor:g}, {len(self.consistency_skip)} heads skip rows "
              "labelled for other tasks")

    @property
    def teacher(self):
        box = getattr(self, "_teacher", None)
        return box.model if box is not None else None

    def teacher_outputs(self, x):
        net = self.teacher if self.teacher is not None else self
        out = net(x, training=False)
        return {k: tf.stop_gradient(out[k]) for k in self._mtl_task_loss_names}

    def teacher_gap(self, target, student):
        gaps = [tf.reduce_mean(tf.abs(tf.cast(target[k], tf.float32)
                                      - tf.cast(student[k], tf.float32)))
                for k in self._mtl_task_loss_names]
        return tf.add_n(gaps) / len(gaps)

    def update_teacher(self):
        """Move the EMA teacher toward the student; returns the decay used."""
        if self.teacher is None:
            return tf.constant(0.0)
        step = tf.cast(self.optimizer.iterations, tf.float32)
        decay = tf.minimum(self.ema_decay, 1.0 - 1.0 / (step + 1.0))
        for t, s in zip(self.teacher.trainable_variables, self.trainable_variables):
            t.assign(decay * t + (1.0 - decay) * s)
        return decay

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, names = self._balancer, self.task_names
        variables = self.trainable_variables
        cons_weight = self.consistency_weight * self.consistency_ramp()
        target = self.teacher_outputs(x)
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            y_pred = {k: pred[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            student = pred if self.student_dropout else self(x, training=False)
            cons_loss, kept = self.consistency_loss(target, student, y)
            objective = bal.objective(losses, has_labels) + cons_weight * cons_loss
        grads = tape.gradient(objective, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        new_pred = self(x, training=True)
        new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
        bal.update(losses, new_losses, has_labels)
        decay = self.update_teacher()

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        out = {'loss': loss}
        for tracker, v in ((self.consistency_loss_tracker, cons_loss),
                           (self.consistency_weight_tracker, cons_weight),
                           (self.consistency_kept_tracker, kept),
                           (self.teacher_gap_tracker, self.teacher_gap(target, student)),
                           (self.ema_decay_tracker, decay)):
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        return out

    def test_step(self, data):
        """Val logs compare the teacher with the student, both with dropout off
        (so 0 for the current-weights teacher)."""
        x, y = data[0], data[1]
        target, student = self.teacher_outputs(x), self(x, training=False)
        cons_loss, kept = self.consistency_loss(target, student, y)
        for tracker, v in ((self.consistency_loss_tracker, cons_loss),
                           (self.consistency_kept_tracker, kept),
                           (self.teacher_gap_tracker, self.teacher_gap(target, student))):
            tracker.update_state(v)
        return MultiTaskModel.test_step(self, data)


class TeacherConsistencyTrainer(BalancedLossTrainer):
    """BalancedLossTrainer (FAMO) for a TeacherConsistencyModel: builds the
    EMA teacher as a second model of the same config (copied from the student
    once built), then attaches the consistency settings, with the ramp sized
    in epochs of the train dataset."""

    def __init__(self, *args, teacher: str, student_dropout: bool, consistency_weight: float,
                 consistency_rampup_epochs: float = 1.0, consistency_keep: float = 0.0,
                 consistency_floor: float = 0.0,
                 consistency_skip: Optional[Dict[str, Sequence[str]]] = None,
                 ema_decay: float = 0.99, **kwargs):
        super().__init__(*args, balancer="famo", **kwargs)
        if not isinstance(self.model, TeacherConsistencyModel):
            raise TypeError("TeacherConsistencyTrainer needs a TeacherConsistencyModel")
        if teacher not in CONSISTENCY_TEACHERS:
            raise ValueError(f"teacher must be one of {CONSISTENCY_TEACHERS}, got {teacher!r}")
        self.teacher_name, self.student_dropout = teacher, student_dropout
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.consistency_keep, self.consistency_floor = consistency_keep, consistency_floor
        self.consistency_skip, self.ema_decay = consistency_skip, ema_decay

    def fit(self, train_dataset, *args, **kwargs):
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("The consistency ramp needs a train dataset of known length")
        x = next(iter(train_dataset))[0]
        self.model(x, training=False)  # create all variables
        self.model._balancer = FAMO(len(self.model.task_names) - len(self.pooled_tasks) + 1)
        teacher = None
        if self.teacher_name == "ema":
            teacher = MultiTaskModel(self.model.config, dropout_rate=self.model.dropout_rate)
            teacher(x, training=False)
        self.model.attach_teacher(teacher, self.student_dropout, self.consistency_weight,
                                  round(self.consistency_rampup_epochs * steps),
                                  self.consistency_keep, self.consistency_floor,
                                  self.consistency_skip, self.ema_decay)
        return PooledOutputLossTrainer.fit(self, train_dataset, *args, **kwargs)


# ---------------------------------------------------------------------------
# Semi-supervised terms on foreign rows: a target rate and a confidence penalty
# ---------------------------------------------------------------------------

SSL_POOL = "propaganda_max_pool"
SSL_SOURCES = ("IAC", "IBM", "Propaganda")
# head@source=rate. Rough rates from reading 40 random train rows per source
# (Claude, 2026-11-03; not a proper annotation). For the IAC heads on
# Propaganda rows the rate is for rows the technique rule did not label,
# since rule-labelled rows are not foreign.
SSL_DEFAULT_RATES = (
    "propaganda_max_pool@IAC=0.30,propaganda_max_pool@IBM=0.10,"
    "emotion_fact_bi@Propaganda=0.12,nasty_nice_bi@Propaganda=0.05,"
    "attacking_respectful_bi@Propaganda=0.07,"
    "emotion_fact_bi@IBM=0.15,nasty_nice_bi@IBM=0.03,attacking_respectful_bi@IBM=0.05"
)
SSL_TRACKERS = ("ssl_rate_loss", "ssl_entropy_loss", "ssl_entropy_weight")


def parse_ssl_rates(spec: str) -> List[Tuple[str, str, float]]:
    """'head@source=rate,...' -> [(head, source, rate)]."""
    pairs = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        try:
            cell, rate = item.split("=")
            head, source = cell.split("@")
            rate = float(rate)
        except ValueError:
            raise ValueError(f"Bad rate {item!r}: expected head@source=rate") from None
        if not 0.0 < rate < 1.0:
            raise ValueError(f"Rate for {cell} must be strictly between 0 and 1, got {rate:g}")
        pairs.append((head, source, rate))
    if len({(h, s) for h, s, _ in pairs}) != len(pairs):
        raise ValueError("Each head@source pair may appear only once")
    return pairs


def ssl_pair_label(head: str, source: str) -> str:
    short = "propaganda" if head == SSL_POOL else re.sub(r"_bi$", "", head)
    return f"{short}_{source}"


def semi_supervised_terms(y, y_pred, pairs, pooled_tasks, home_tasks, rest_source,
                          mask_value=-1.0, eps=1e-6):
    """The two extra terms over foreign cells. A cell of head h is foreign for
    the pair (h, source, rate) when its row comes from ``source`` and h has no
    label there. A row's source is read from the labels: a label for any task
    in ``home_tasks[s]`` puts it in s, none of them puts it in ``rest_source``.
    The pooled propaganda output (``SSL_POOL``) is the max of the
    ``pooled_tasks`` heads, with a label wherever any technique has one.

    - Rate: the binary KL divergence from the target rate to the mean score
      over the pair's foreign cells in the batch. It constrains only the
      batch average, never a single row.
    - Entropy: the mean binary entropy of the scores on those cells, which
      is lowest when every score is near 0 or 1.

    Both are averaged over the pairs that have foreign cells in the batch, so
    they are 0 when there are none. Returns (rate, entropy, mean score per
    pair [P], foreign cells per pair [P]); a pair with no cells has mean 0."""
    def flat(d, k):
        return tf.reshape(tf.cast(d[k], tf.float32), [-1])

    def head_of(d, head):
        if head == SSL_POOL:
            return tf.reduce_max(tf.stack([flat(d, t) for t in pooled_tasks], axis=1), axis=1)
        return flat(d, head)

    home = {s: tf.reduce_any(tf.stack([tf.not_equal(flat(y, t), mask_value) for t in ts],
                                      axis=1), axis=1)
            for s, ts in home_tasks.items()}
    sources = dict(home)
    sources[rest_source] = tf.logical_not(tf.reduce_any(tf.stack(list(home.values()), axis=1),
                                                        axis=1))
    rates, ents, means, counts, active = [], [], [], [], []
    for head, source, rate in pairs:
        m = tf.cast(sources[source] & tf.equal(head_of(y, head), mask_value), tf.float32)
        p = tf.clip_by_value(head_of(y_pred, head), eps, 1.0 - eps)
        n = tf.reduce_sum(m)
        mean = tf.math.divide_no_nan(tf.reduce_sum(p * m), n)
        q = tf.clip_by_value(mean, eps, 1.0 - eps)
        kl = rate * tf.math.log(rate / q) + (1.0 - rate) * tf.math.log((1.0 - rate) / (1.0 - q))
        entropy = -(p * tf.math.log(p) + (1.0 - p) * tf.math.log(1.0 - p))
        on = tf.cast(n > 0, tf.float32)
        rates.append(kl * on)
        ents.append(tf.math.divide_no_nan(tf.reduce_sum(entropy * m), n))
        means.append(mean)
        counts.append(n)
        active.append(on)
    n_active = tf.maximum(tf.add_n(active), 1.0)
    return (tf.add_n(rates) / n_active, tf.add_n(ents) / n_active,
            tf.stack(means), tf.stack(counts))


class SemiSupervisedModel(MultiTaskModel):
    """MultiTaskModel trained on the supervised loss plus, on foreign cells
    (``semi_supervised_terms``), ``rate_weight`` x the rate term and
    ``entropy_weight`` x the entropy term. The entropy term is off until
    ``entropy_start_step`` optimizer steps have run. The reported and
    validation loss is the supervised loss alone, so early stopping compares
    with the plain pooled-loss runs. Logs both terms and, per pair, the mean
    score on its foreign cells (``ssl_mean_<head>_<source>``), on train and val.
    Call ``attach_semi_supervised`` before training."""

    def attach_semi_supervised(self, pairs, pooled_tasks, home_tasks, rest_source,
                               rate_weight, entropy_weight=0.0, entropy_start_step=0):
        names = set(self.task_names)
        sources = set(home_tasks) | {rest_source}
        missing = [t for ts in home_tasks.values() for t in ts if t not in names]
        if missing:
            raise ValueError(f"Source-defining tasks not in the model: {missing}")
        for head, source, _ in pairs:
            if head == SSL_POOL:
                if not pooled_tasks:
                    raise ValueError(f"{SSL_POOL} needs the pooled propaganda heads")
            elif head not in names:
                raise ValueError(f"Unknown head {head!r}")
            if source not in sources:
                raise ValueError(f"Unknown source {source!r}; choose from {sorted(sources)}")
        if rate_weight < 0 or entropy_weight < 0 or not (rate_weight or entropy_weight):
            raise ValueError("Weights must be >= 0 and at least one > 0")
        self.ssl_pairs = [(h, s, float(r)) for h, s, r in pairs]
        self.ssl_pooled_tasks = list(pooled_tasks)
        self.ssl_home_tasks = {k: list(v) for k, v in home_tasks.items()}
        self.ssl_rest_source = rest_source
        self.ssl_rate_weight, self.ssl_entropy_weight = float(rate_weight), float(entropy_weight)
        self.ssl_entropy_start_step = int(entropy_start_step)
        self.ssl_labels = [ssl_pair_label(h, s) for h, s, _ in self.ssl_pairs]
        # Attributes, so Keras resets them every epoch.
        for k in SSL_TRACKERS:
            setattr(self, f"{k}_tracker", tf.keras.metrics.Mean(name=k))
        self.ssl_mean_trackers = [tf.keras.metrics.Mean(name=f"ssl_mean_{lab}")
                                  for lab in self.ssl_labels]
        print(f"Semi-supervised: rate weight {self.ssl_rate_weight:g}, entropy weight "
              f"{self.ssl_entropy_weight:g} from step {self.ssl_entropy_start_step}; sources "
              f"{ {**self.ssl_home_tasks, rest_source: 'neither'} }")
        for (h, s, r), lab in zip(self.ssl_pairs, self.ssl_labels):
            print(f"  {lab}: target rate {r:g}")

    def ssl_terms(self, y, y_pred):
        return semi_supervised_terms(y, y_pred, self.ssl_pairs, self.ssl_pooled_tasks,
                                     self.ssl_home_tasks, self.ssl_rest_source,
                                     self.config.mask_value)

    def ssl_entropy_weight_now(self):
        on = tf.cast(self.optimizer.iterations, tf.float32) >= float(self.ssl_entropy_start_step)
        return tf.where(on, self.ssl_entropy_weight, 0.0)

    def _log_ssl(self, rate, entropy, means, counts, entropy_weight=None):
        out = {}
        pairs = [(self.ssl_rate_loss_tracker, rate), (self.ssl_entropy_loss_tracker, entropy)]
        if entropy_weight is not None:
            pairs.append((self.ssl_entropy_weight_tracker, entropy_weight))
        for tracker, v in pairs:
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        for i, tracker in enumerate(self.ssl_mean_trackers):
            tracker.update_state(means[i], sample_weight=counts[i])
            out[tracker.name] = tracker.result()
        return out

    def train_step(self, data):
        x, y = data[0], data[1]
        names = self.task_names
        entropy_weight = self.ssl_entropy_weight_now()
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            y_pred = {k: pred[k] for k in names}
            loss = self._mtl_loss_fn(y, y_pred)
            rate, entropy, means, counts = self.ssl_terms(y, y_pred)
            total = loss + self.ssl_rate_weight * rate + entropy_weight * entropy
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        return {'loss': loss, **self._log_ssl(rate, entropy, means, counts, entropy_weight)}

    def test_step(self, data):
        x, y = data[0], data[1]
        pred = self(x, training=False)
        self._log_ssl(*self.ssl_terms(y, {k: pred[k] for k in self.task_names}))
        return MultiTaskModel.test_step(self, data)


class SemiSupervisedTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a SemiSupervisedModel: attaches the pairs
    and weights, with the entropy start given in epochs of the train dataset."""

    def __init__(self, *args, ssl_pairs, ssl_home_tasks, ssl_rest_source: str = "IAC",
                 rate_weight: float, entropy_weight: float = 0.0,
                 entropy_start_epochs: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, SemiSupervisedModel):
            raise TypeError("SemiSupervisedTrainer needs a SemiSupervisedModel")
        self.ssl_pairs, self.ssl_home_tasks = ssl_pairs, ssl_home_tasks
        self.ssl_rest_source = ssl_rest_source
        self.rate_weight, self.entropy_weight = rate_weight, entropy_weight
        self.entropy_start_epochs = entropy_start_epochs

    def fit(self, train_dataset, *args, **kwargs):
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("The entropy start needs a train dataset of known length")
        self.model.attach_semi_supervised(
            self.ssl_pairs, self.pooled_tasks, self.ssl_home_tasks, self.ssl_rest_source,
            self.rate_weight, self.entropy_weight, round(self.entropy_start_epochs * steps))
        return super().fit(train_dataset, *args, **kwargs)


# ---------------------------------------------------------------------------
# Per-task stopping with self-distillation (plain pooled loss)
# ---------------------------------------------------------------------------
# Each Table-3 output trains on its labels until its val F1 has not improved
# for ``patience`` epochs. Then the weights of its best epoch score every train
# row once (eval mode), and from there on that output's loss, on the same
# labelled rows with the same class weights and scale, is the binary KL from
# those stored scores instead of BCE on the label. The other outputs go on as
# before. Training ends once every output has switched (or at --epochs).
# After Caruana (1997, 7.1) per-task stopping and Lu et al. (NAACL 2022),
# distilling converged tasks from their own best checkpoint.


class _DistillState:
    """Plain holder (not tracked by Keras), so the stored scores are not part
    of get_weights / set_weights and survive weight restores."""


class SelfDistillMixin:
    """The score table and switches shared by the self-distilled models:
    ``_sd.scores`` [rows, outputs] (stored best-epoch scores), ``_sd.on``
    [outputs] (1 once an output has switched). Call ``attach_self_distill``
    before training; ``SelfDistillCallback`` fills both."""

    def attach_self_distill(self, outputs: Sequence[str], n_rows: int):
        st = _DistillState()
        st.outputs = list(outputs)
        st.scores = tf.Variable(tf.fill([n_rows, len(st.outputs)], 0.5),
                                trainable=False, name="self_distill_scores")
        st.on = tf.Variable(tf.zeros([len(st.outputs)]), trainable=False,
                            name="self_distill_on")
        self._sd = st
        print(f"Self-distillation: {len(st.outputs)} outputs, score table for {n_rows:,} rows")

    def soft_targets(self, y):
        """{output: (stored score per batch row, on)} for the pooled loss."""
        st = self._sd
        rows = tf.reshape(tf.cast(y[ROW_ID_KEY], tf.int32), [-1])
        target = tf.gather(st.scores, rows)
        return {k: (target[:, j], st.on[j]) for j, k in enumerate(st.outputs)}

    def output_on(self, name):
        """Switch flag of one output (0 for outputs not in the table)."""
        st = self._sd
        return st.on[st.outputs.index(name)] if name in st.outputs else tf.constant(0.0)


class SelfDistillModel(SelfDistillMixin, MultiTaskModel):
    """MultiTaskModel on the plain pooled loss whose outputs can be switched,
    one by one, from their labels to stored scores (``SelfDistillCallback``).
    The logged train loss is the loss being optimised; val loss stays the
    plain label loss. Call ``attach_self_distill`` before training."""

    def train_step(self, data):
        x, y = data[0], data[1]
        st = self._sd
        names = self.task_names
        soft = self.soft_targets(y)
        with tf.GradientTape() as tape:
            pred = self(x, training=True)
            y_pred = {k: pred[k] for k in names}
            loss = self._mtl_loss_fn(y, y_pred, soft)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        return {'loss': loss, 'distilled_outputs': tf.reduce_sum(st.on)}


class SelfDistillFamoModel(SelfDistillMixin, LabelAwareMaskedModel):
    """famo_consf1_m5 (FAMO + filtered masked consistency) with per-task
    self-distillation. FAMO works on log losses, and a just-switched output's
    KL starts near 0, so switched outputs leave FAMO (their has_labels is set
    to 0, so they get no FAMO weight and no FAMO update) and their KL from the
    stored scores is added outside FAMO with weight ``distill_weight``, like
    the masked-word and consistency terms: the plain pooled loss restricted
    to the switched outputs (mean KL per labelled cell, same class weights).
    The consistency term is unchanged and still covers all 10 outputs."""

    distill_weight = 1.0

    def distill_loss(self, y, y_pred):
        mask = self.config.mask_value
        pooled = set(getattr(self, "_mtl_pooled_tasks", []))
        y_d = {}
        for k in self.task_names:
            on = self.output_on("propaganda_max_pool" if k in pooled else k)
            y_d[k] = tf.where(on > 0, tf.cast(y[k], tf.float32),
                              tf.fill(tf.shape(y[k]), tf.cast(mask, tf.float32)))
        on_all = {k: (t, tf.constant(1.0)) for k, (t, _) in self.soft_targets(y).items()}
        return self._mtl_loss_fn(y_d, y_pred, on_all)

    def train_step(self, data):
        x, y = data[0], data[1]
        bal, names = self._balancer, self.task_names
        variables = self.trainable_variables
        masking = self.mask_batch(x)
        cons_weight = self.consistency_weight * self.consistency_ramp()
        in_famo = 1.0 - tf.stack([self.output_on(n) for n in self._mtl_task_loss_names])
        with tf.GradientTape() as tape:
            clean = self(x, training=True)
            y_pred = {k: clean[k] for k in names}
            losses, has_labels = self._mtl_task_losses_fn(y, y_pred)
            has_labels = has_labels * in_famo
            mlm_loss, mlm_acc, label_loss, cons_loss, kept = self.label_aware_losses(
                x, y, clean, True, masking)
            distill = self.distill_loss(y, y_pred)
            objective = (bal.objective(losses, has_labels) + self.mlm_lambda * mlm_loss
                         + self.masked_label_weight * label_loss + cons_weight * cons_loss
                         + self.distill_weight * distill)
        grads = tape.gradient(objective, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        new_pred = self(x, training=True)
        new_losses, _ = self._mtl_task_losses_fn(y, {k: new_pred[k] for k in names})
        bal.update(losses, new_losses, has_labels)

        loss = self._mtl_loss_fn(y, y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
                break
        out = {'loss': loss, 'distill_loss': distill,
               'distilled_outputs': tf.reduce_sum(self._sd.on)}
        for tracker, v in ((self.mlm_loss_tracker, mlm_loss), (self.mlm_acc_tracker, mlm_acc),
                           (self.masked_label_loss_tracker, label_loss),
                           (self.consistency_loss_tracker, cons_loss),
                           (self.consistency_weight_tracker, cons_weight),
                           (self.consistency_kept_tracker, kept)):
            tracker.update_state(v)
            out[tracker.name] = tracker.result()
        return out


class SelfDistillFamoTrainer(LabelAwareMaskedTrainer):
    """LabelAwareMaskedTrainer for a SelfDistillFamoModel: also attaches the
    score table (batches x batch size >= rows)."""

    def __init__(self, *args, distill_outputs: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, SelfDistillFamoModel):
            raise TypeError("SelfDistillFamoTrainer needs a SelfDistillFamoModel")
        if self.technique_loss_weight:
            raise ValueError("self-distillation does not support --technique-loss-weight")
        self.distill_outputs = list(distill_outputs)

    def fit(self, train_dataset, *args, **kwargs):
        _, y = next(iter(train_dataset))
        if ROW_ID_KEY not in y:
            raise ValueError("self-distillation needs load_splits(train_row_ids=True)")
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("self-distillation needs a train dataset of known length")
        self.model.attach_self_distill(self.distill_outputs,
                                       steps * int(tf.shape(y[ROW_ID_KEY])[0]))
        return super().fit(train_dataset, *args, **kwargs)


class SelfDistillTrainer(PooledOutputLossTrainer):
    """PooledOutputLossTrainer for a SelfDistillModel: sizes the score table
    from the train dataset (batches x batch size >= rows) and attaches it."""

    def __init__(self, *args, distill_outputs: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, SelfDistillModel):
            raise TypeError("SelfDistillTrainer needs a SelfDistillModel")
        if self.technique_loss_weight:
            raise ValueError("self-distillation does not support --technique-loss-weight")
        self.distill_outputs = list(distill_outputs)

    def fit(self, train_dataset, *args, **kwargs):
        x, y = next(iter(train_dataset))
        if ROW_ID_KEY not in y:
            raise ValueError("self-distillation needs load_splits(train_row_ids=True)")
        steps = int(train_dataset.cardinality())
        if steps < 0:
            raise ValueError("self-distillation needs a train dataset of known length")
        self.model(x, training=False)  # create all variables
        self.model.attach_self_distill(self.distill_outputs,
                                       steps * int(tf.shape(y[ROW_ID_KEY])[0]))
        return super().fit(train_dataset, *args, **kwargs)


SELF_DISTILL_SIGNALS = ("f1", "loss")  # per-output val signal for switching


class SelfDistillCallback(tf.keras.callbacks.Callback):
    """Switches each output to self-distillation once its val ``signal`` (F1,
    higher is better, or its own val loss, lower is better; from
    ``scorer.records``, so it must come after the scorer in the callback list)
    has gone ``patience`` epochs without improving, using the scores of its
    first best epoch. Keeps the trainable weights of only the epochs that can
    still be needed. Stops training once every output has switched.
    ``switches`` lists {output, best_epoch, switched_after_epoch}."""

    def __init__(self, scorer: "EpochScorer", train_ds, patience: int = 2,
                 pooled_tasks: Sequence[str] = (), pool_strategy: str = "max",
                 signal: str = "f1"):
        super().__init__()
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if signal not in SELF_DISTILL_SIGNALS:
            raise ValueError(f"signal must be one of {SELF_DISTILL_SIGNALS}")
        self.signal = signal
        self.scorer, self.train_ds, self.patience = scorer, train_ds, patience
        self.pooled_tasks, self.pool_strategy = list(pooled_tasks), pool_strategy

    def on_train_begin(self, logs=None):
        self.snapshots: Dict[int, List[np.ndarray]] = {}
        self.switches: List[Dict[str, Any]] = []
        self._done = set()
        self._predict = tf.function(lambda x: self.model(x, training=False))

    def _weights(self):
        return [v.numpy() for v in self.model.trainable_variables]

    def _assign(self, weights):
        for v, w in zip(self.model.trainable_variables, weights):
            v.assign(w)

    def _score_train(self, outputs: Sequence[str]) -> np.ndarray:
        """Scores [rows, len(outputs)] of the current weights on every train row."""
        st = self.model._sd
        out = np.full((int(st.scores.shape[0]), len(outputs)), 0.5, dtype=np.float32)
        for x, y in self.train_ds:
            pred = self._predict(x)
            rows = np.asarray(y[ROW_ID_KEY]).reshape(-1)
            for j, k in enumerate(outputs):
                if k == "propaganda_max_pool":
                    stacked = tf.stack([tf.reshape(pred[t], [-1]) for t in self.pooled_tasks], axis=1)
                    p = pool_probabilities(stacked, self.pool_strategy, axis=1)
                else:
                    p = pred[k]
                out[rows, j] = np.asarray(p, dtype=np.float32).reshape(-1)
        return out

    def on_epoch_end(self, epoch, logs=None):
        st = self.model._sd
        n = epoch + 1
        recs = self.scorer.records
        if len(recs) != n:
            raise RuntimeError("SelfDistillCallback must come after the EpochScorer")
        self.snapshots[n] = self._weights()
        pick = np.argmin if self.signal == "loss" else np.argmax
        best = {k: 1 + int(pick([r["val"][k][self.signal] for r in recs]))
                for k in st.outputs if k not in self._done}
        ready = {k: b for k, b in best.items() if n - b >= self.patience}
        by_epoch: Dict[int, List[str]] = {}
        for k, b in ready.items():
            by_epoch.setdefault(b, []).append(k)
        if by_epoch:
            current = self.snapshots[n]
            scores, on = st.scores.numpy(), st.on.numpy()
            for b, ks in sorted(by_epoch.items()):
                self._assign(self.snapshots[b])
                new = self._score_train(ks)
                for j, k in enumerate(ks):
                    col = st.outputs.index(k)
                    scores[:, col], on[col] = new[:, j], 1.0
                    self._done.add(k)
                    self.switches.append({"output": k, "best_epoch": b, "switched_after_epoch": n})
                f1s = ", ".join(f"{100 * recs[b - 1]['val'][k]['f1']:.2f}" for k in ks)
                sig = ", ".join(f"{recs[b - 1]['val'][k][self.signal]:.4f}" for k in ks)
                print(f"Epoch {n}: switched {ks} to self-distillation from epoch {b} "
                      f"(best val {self.signal} {sig}; val F1 {f1s})", flush=True)
            self._assign(current)
            st.scores.assign(scores)
            st.on.assign(on)
        recs[-1]["distilled"] = sorted(self._done)
        needed = {n} | {b for k, b in best.items() if k not in self._done}
        self.snapshots = {e: w for e, w in self.snapshots.items() if e in needed}
        if len(self._done) == len(st.outputs):
            print(f"Epoch {n}: every output has switched, stopping", flush=True)
            self.model.stop_training = True


# ---------------------------------------------------------------------------
# Per-epoch scoring and early stopping on validation F1 / AUC
# ---------------------------------------------------------------------------

EARLY_STOP_METRICS = ("loss", "f1", "auc")


class EpochScorer(tf.keras.callbacks.Callback):
    """After every epoch, scores the ten Table-3 outputs on val (thresholds
    tuned on val, as after training) and on test (with those val thresholds),
    and logs the val means as ``val_f1`` / ``val_auc``.

    Stops once val loss, val F1 and val AUC have all gone ``patience`` epochs
    without improving, then restores the weights of the best epoch by
    ``monitor``. Every stopping rule with patience <= ``patience`` on any of
    the three can therefore be replayed from ``records``, test is never used
    to pick an epoch. ``trainer`` is set once the trainer exists. ``outputs``
    defaults to the Table-3 outputs the model has.

    ``main_output`` (Caruana 1997, Section 7.1: early stopping on the main
    task alone): the epoch is picked on that one output's val F1 / AUC
    instead of the mean, and training stops once that output's val F1 and
    AUC have both stalled ``patience`` epochs. Val loss plays no part then,
    since it sums every task. All outputs are still scored every epoch.

    ``evals_per_epoch`` = k > 1 scores k times per epoch, at equal step
    intervals (the last one at the epoch's end), so a best point between
    epochs can be kept. A record's ``epoch`` is then a fraction (0.25, 0.5,
    ...), ``patience`` counts scorings instead of epochs, and training can
    stop in the middle of an epoch. ``val_loss`` is then worked out here at
    every scoring, as the mean over the val batches of the model's loss.
    (With one scoring per epoch it stays the number Keras logs, which for
    this model is the loss of the last val batch only: see
    test_finetune_recipe.py.)
    """

    def __init__(self, val_ds, test_ds, monitor: str = "f1", patience: int = 3,
                 outputs: Optional[Sequence[str]] = None,
                 main_output: Optional[str] = None, stops: bool = True,
                 evals_per_epoch: int = 1):
        super().__init__()
        if evals_per_epoch < 1:
            raise ValueError("evals_per_epoch must be at least 1")
        self.evals_per_epoch = int(evals_per_epoch)
        if monitor not in EARLY_STOP_METRICS:
            raise ValueError(f"monitor must be one of {EARLY_STOP_METRICS}, got {monitor!r}")
        if main_output and monitor == "loss":
            raise ValueError("main_output needs monitor 'f1' or 'auc' (val loss sums every task)")
        self.val_ds, self.test_ds = val_ds, test_ds
        self.monitor, self.patience = monitor, patience
        self.outputs = list(outputs) if outputs is not None else None
        self.main_output = main_output or None
        self.stops = stops  # False: another callback ends training (self-distillation)
        self.trainer = None
        self.records: List[Dict[str, Any]] = []
        self.best_epoch = None   # the kept record's ``epoch``
        self.best_index = None   # its 1-based position in ``records``

    def _scores(self, ds, thresholds):
        res = self.trainer.evaluate(ds, thresholds=thresholds)
        merged = {**res.get("task_metrics", {}), **res.get("pool_metrics", {})}
        outputs = self.outputs or [k for k in PAPER_TABLE3_MULTITASK_F1 if k in merged]
        return {k: {"f1": float(merged[k]["f1"]), "auc": float(merged[k]["auc"])}
                for k in outputs}

    def on_train_begin(self, logs=None):
        if self.trainer is None:
            raise RuntimeError("EpochScorer.trainer was never set")
        self.records, self.best_epoch, self._best_weights = [], None, None
        self.best_index, self._epoch, self._stopped = None, 0, False
        names = list(self.model.task_names)
        labels = {k: [] for k in names}
        self._val_batch_sizes = []
        for _, y in self.val_ds:
            for k in names:
                labels[k].append(y[k].numpy())
            self._val_batch_sizes.append(len(labels[names[0]][-1]))
        self._val_labels = {k: np.concatenate(v, axis=0) for k, v in labels.items()}
        self._val_labels.update(self.trainer._derive_pooled_labels(self._val_labels))

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        """The in-between scorings of ``evals_per_epoch``; the epoch's last
        one is ``on_epoch_end``."""
        if self.evals_per_epoch == 1 or self._stopped:
            return
        steps = self.params.get("steps")
        if not steps:
            raise RuntimeError("evals_per_epoch needs a known number of steps per epoch")
        k = self.evals_per_epoch
        if batch + 1 in {round(j * steps / k) for j in range(1, k)} and batch + 1 < steps:
            self._score(self._epoch + (batch + 1) / steps, None, None)

    def on_epoch_end(self, epoch, logs=None):
        if self._stopped:  # stopped at an in-between scoring of this epoch
            return
        logs = logs if logs is not None else {}
        self._score(epoch + 1,
                    float(logs["val_loss"]) if self.evals_per_epoch == 1 else None, logs)

    def _val_loss(self, pred) -> float:
        """Mean over the val batches of the model's loss, from predictions
        already made."""
        names = list(self.model.task_names)
        losses, start = [], 0
        for size in self._val_batch_sizes:
            rows = slice(start, start + size)
            y = {k: tf.constant(self._val_labels[k][rows], tf.float32) for k in names}
            p = {k: tf.constant(np.asarray(pred[k])[rows], tf.float32) for k in names}
            losses.append(float(self.model._mtl_loss_fn(y, p)))
            start += size
        return float(np.mean(losses))

    def _score(self, epoch, val_loss, logs):
        """Score val and test now and record it as ``epoch`` (a whole number
        at an epoch's end, a fraction in between)."""
        pred = self.model.predict(self.val_ds, verbose=0)
        thresholds, _ = self.trainer._find_optimal_thresholds(self._val_labels, pred)
        val, test = self._scores(self.val_ds, thresholds), self._scores(self.test_ds, thresholds)
        for k, v in self._val_task_losses(pred).items():
            if k in val:
                val[k]["loss"] = v
        if val_loss is None:
            val_loss = self._val_loss(pred)
        rec = {"epoch": epoch, "val_loss": float(val_loss), "val": val, "test": test}
        for m in ("f1", "auc"):
            rec[f"val_{m}"] = float(np.mean([s[m] for s in val.values()]))
            rec[f"test_{m}"] = float(np.mean([s[m] for s in test.values()]))
            if logs is not None:
                logs[f"val_{m}"] = rec[f"val_{m}"]
        if self.main_output:
            if self.main_output not in val:
                raise ValueError(f"main_output {self.main_output!r} is not scored: {sorted(val)}")
            for m in ("f1", "auc"):
                rec[f"val_main_{m}"] = val[self.main_output][m]
                rec[f"test_main_{m}"] = test[self.main_output][m]
        if getattr(self.model, "teacher", None) is not None:
            rec["teacher"] = self._teacher_scores()
        self.records.append(rec)
        index = len(self.records)  # = the epoch number unless evals_per_epoch > 1
        print(f"Epoch {epoch:g}: val_loss {rec['val_loss']:.4f}  val F1 {100 * rec['val_f1']:.2f}  "
              f"val AUC {100 * rec['val_auc']:.2f}  (test F1 {100 * rec['test_f1']:.2f})",
              flush=True)
        if self.main_output:
            print(f"Epoch {epoch:g}: main task {self.main_output} val F1 "
                  f"{100 * rec['val_main_f1']:.2f}  val AUC {100 * rec['val_main_auc']:.2f}",
                  flush=True)
        if "teacher" in rec:
            print(f"Epoch {epoch:g}: EMA teacher val F1 {100 * rec['teacher']['val_f1']:.2f}  "
                  f"val AUC {100 * rec['teacher']['val_auc']:.2f}", flush=True)

        if self.best_index is None or self._better(rec, self.records[self.best_index - 1]):
            self.best_index, self.best_epoch = index, epoch
            self._best_weights = self.model.get_weights()
        stop_metrics = ("f1", "auc") if self.main_output else EARLY_STOP_METRICS
        stalled = all(index - self._best_by(m) >= self.patience for m in stop_metrics)
        if stalled and self.stops:
            what = (f"main task {self.main_output} val F1 and AUC both" if self.main_output
                    else "val loss, F1 and AUC all")
            print(f"Epoch {epoch:g}: {what} stalled for {self.patience} "
                  f"{'epochs' if self.evals_per_epoch == 1 else 'scorings'}, stopping",
                  flush=True)
            self.model.stop_training = True
            self._stopped = True

    def _teacher_scores(self):
        """The EMA teacher scored like the student (its own val thresholds),
        by loading its weights into the student for the scoring. A read-out
        only: it plays no part in picking the epoch."""
        student = [v.numpy() for v in self.model.trainable_variables]
        for v, t in zip(self.model.trainable_variables, self.model.teacher.trainable_variables):
            v.assign(t)
        try:
            pred = self.model.predict(self.val_ds, verbose=0)
            thresholds, _ = self.trainer._find_optimal_thresholds(self._val_labels, pred)
            val, test = self._scores(self.val_ds, thresholds), self._scores(self.test_ds, thresholds)
        finally:
            for v, w in zip(self.model.trainable_variables, student):
                v.assign(w)
        rec = {"val": val, "test": test}
        for m in ("f1", "auc"):
            rec[f"val_{m}"] = float(np.mean([s[m] for s in val.values()]))
            rec[f"test_{m}"] = float(np.mean([s[m] for s in test.values()]))
        return rec

    def _val_task_losses(self, pred) -> Dict[str, float]:
        """Each output's own class-weighted mean val loss (the per-task loss
        the model trains on, without nu_k: ``_mtl_task_losses_fn``), over the
        whole val split; outputs with no labelled val row are left out."""
        fn = getattr(self.model, "_mtl_task_losses_fn", None)
        if fn is None:
            return {}
        names = list(self.model.task_names)
        y = {k: tf.constant(self._val_labels[k], tf.float32) for k in names}
        p = {k: tf.constant(np.asarray(pred[k]), tf.float32) for k in names}
        losses, has = fn(y, p)
        return {n: float(v) for n, v, h in
                zip(self.model._mtl_task_loss_names, losses.numpy(), has.numpy()) if h > 0}

    def _key(self, metric):
        if metric == "loss":
            return "val_loss"
        return f"val_main_{metric}" if self.main_output else f"val_{metric}"

    def _better(self, a, b):
        key = self._key(self.monitor)
        if self.monitor == "loss":
            return a[key] < b[key]
        return a[key] > b[key]

    def _best_by(self, metric):
        """Position in ``records`` (1-based; the epoch number unless
        ``evals_per_epoch`` > 1) of the first best value of ``metric`` so far."""
        key = self._key(metric)
        vals = [r[key] for r in self.records]
        return 1 + int(np.argmin(vals) if metric == "loss" else np.argmax(vals))

    def on_train_end(self, logs=None):
        if self._best_weights is not None:
            what = f"{self.main_output} " if self.main_output else ""
            print(f"Restoring weights from epoch {self.best_epoch:g} "
                  f"(best val {what}{self.monitor})", flush=True)
            self.model.set_weights(self._best_weights)


def mask_sweep(scorer: "EpochScorer", rates: Sequence[float], repeats: int = 3,
               seed: int = 0) -> Dict[str, Any]:
    """How much masking the input changes the trained model's val answers.

    Uses the model and trainer of a finished run (best epoch restored) with
    thresholds tuned on the clean val set. For each mask rate and ``repeats``
    random maskings (BERT's rule, as in training; eval mode, so no dropout)
    it scores the masked val set with those thresholds and compares each
    output's answers with the clean ones: F1 / AUC on labelled cells, the
    share of cells whose yes/no answer flips, and the mean absolute score
    change, the last two split into labelled and unlabelled cells."""
    model, trainer = scorer.model, scorer.trainer
    # One fixed copy of val, so clean and masked answers line up row by row.
    xs, ys = [], []
    for x, y in scorer.val_ds:
        xs.append({k: v.numpy() for k, v in x.items()})
        ys.append({k: v.numpy() for k, v in y.items()})
    x_all = {k: np.concatenate([b[k] for b in xs]) for k in xs[0]}
    y_all = {k: np.concatenate([b[k] for b in ys]) for k in ys[0]}
    batch_size = len(xs[0]["input_ids"])
    vocab_size = model.encoder.config.vocab_size

    def dataset(ids):
        return tf.data.Dataset.from_tensor_slices(
            ({**x_all, "input_ids": ids}, y_all)).batch(batch_size)

    clean_ds = dataset(x_all["input_ids"])
    labels = {**y_all, **trainer._derive_pooled_labels(y_all)}
    clean_pred = model.predict(clean_ds, verbose=0)
    thresholds, _ = trainer._find_optimal_thresholds(labels, clean_pred)
    clean = scorer._scores(clean_ds, thresholds)
    outputs = list(clean)

    result = {"rates": list(rates), "repeats": repeats, "clean": clean,
              "thresholds": {k: float(thresholds[k]) for k in outputs}, "by_rate": {}}
    for r in rates:
        runs = []
        for i in range(repeats):
            tf.random.set_seed(seed + 1000 * i + int(round(1e4 * r)))
            masked, _ = mask_tokens(x_all["input_ids"], x_all["attention_mask"], vocab_size, r)
            ds = dataset(masked.numpy())
            scores = scorer._scores(ds, thresholds)
            pred = model.predict(ds, verbose=0)
            rec = {}
            for k in outputs:
                c = np.asarray(clean_pred[k]).ravel()
                m = np.asarray(pred[k]).ravel()
                lab = np.asarray(labels[k]).ravel() != trainer.mask_value
                flip = (c >= thresholds[k]) != (m >= thresholds[k])
                shift = np.abs(m - c)
                rec[k] = {"f1": scores[k]["f1"], "auc": scores[k]["auc"],
                          "flip_labelled": float(flip[lab].mean()),
                          "flip_unlabelled": float(flip[~lab].mean()) if (~lab).any() else None,
                          "shift_labelled": float(shift[lab].mean()),
                          "shift_unlabelled": float(shift[~lab].mean()) if (~lab).any() else None}
            runs.append(rec)
        result["by_rate"][f"{r:g}"] = runs

    print("\nMask sweep on val (mean over repeats): ΔF1 x100 vs clean, flips % labelled / unlabelled")
    print(f"{'output':<26}" + "".join(f"{'rate ' + format(r, 'g'):>24}" for r in rates))
    for k in outputs:
        cells = []
        for r in rates:
            runs = result["by_rate"][f"{r:g}"]
            df1 = 100 * (np.mean([x[k]["f1"] for x in runs]) - clean[k]["f1"])
            fl = 100 * np.mean([x[k]["flip_labelled"] for x in runs])
            fu = [x[k]["flip_unlabelled"] for x in runs if x[k]["flip_unlabelled"] is not None]
            cells.append(f"{df1:+.1f}  {fl:.1f}/{100 * np.mean(fu):.1f}" if fu else f"{df1:+.1f}  {fl:.1f}/–")
        print(f"{k:<26}" + "".join(f"{c:>24}" for c in cells))
    return result


# ---------------------------------------------------------------------------
# Gradient probe (negative-transfer metrics)
# ---------------------------------------------------------------------------

class GradientProbe(tf.keras.callbacks.Callback):
    """Every ``every`` training steps (and before the first), takes each task's
    gradient on the shared parameters over a fixed probe set of train batches
    and records the pairwise cosines, norms and how often two tasks' per-batch
    gradients point against each other.

    Task losses are the per-task class-weighted means of
    ``_mtl_task_losses_fn`` (pooled propaganda first), without nu_k / |T_k|,
    so cosines and conflict rates are what the weighting can't change; norms
    are of the unweighted losses. The probe set is the first train batches
    (up to ``max_batches``) that give every task ``min_labels`` labels; each
    task's probe gradient is the mean of its per-batch gradients.

    With ``lookahead_lr`` > 0 it also records task affinity (Fifty et al.,
    NeurIPS 2021): ``affinity[i][j] = 1 - L_j(after a step on task i) /
    L_j(now)``, the step taken on the shared parameters only and then undone.
    Positive means a step for task i lowers task j's loss. Training uses
    AdamW, whose step length is about lr x sqrt(#params) whatever the gradient
    size, so the lookahead step is task i's unit gradient times that length;
    ``affinity_small`` repeats it at a tenth of the length, where the result
    is close to first order (step x |g_j| x cos(i, j) / L_j)."""

    def __init__(self, train_ds, every: int, min_labels: int = 128, max_batches: int = 16,
                 lookahead_lr: float = 0.0):
        super().__init__()
        self.every, self.step, self.lookahead_lr = every, 0, lookahead_lr
        self.train_ds, self.min_labels, self.max_batches = train_ds, min_labels, max_batches
        self.records: List[Dict[str, Any]] = []

    def on_train_begin(self, logs=None):
        m = self.model
        self.task_names = list(m._mtl_task_loss_names)
        names = m.task_names
        self.batches, counts = [], None
        for x, y in self.train_ds:
            n = self._label_counts(y)
            counts = n if counts is None else counts + n
            self.batches.append((x, y))
            if counts.min() >= self.min_labels or len(self.batches) >= self.max_batches:
                break
        if not m.built:  # fit builds the model after on_train_begin
            m(self.batches[0][0], training=False)
        self.shared = shared_variables(m)
        print(f"Gradient probe: {len(self.batches)} batches, labels per task "
              f"{dict(zip(self.task_names, counts.tolist()))}, "
              f"{sum(int(np.prod(v.shape)) for v in self.shared):,} shared parameters")

        @tf.function
        def grads(x, y):
            with tf.GradientTape(persistent=True) as tape:
                y_pred = m(x, training=False)
                losses, has = m._mtl_task_losses_fn(y, {k: y_pred[k] for k in names})
                # Split per task inside the tape: ops outside it get no gradient.
                per_task = tf.unstack(losses)
            rows = []
            for loss in per_task:
                g = tape.gradient(loss, self.shared,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
                rows.append(tf.concat([tf.reshape(tf.convert_to_tensor(v), [-1]) for v in g], 0))
            del tape
            return tf.stack(rows), losses, has

        @tf.function
        def task_losses(x, y):
            y_pred = m(x, training=False)
            return m._mtl_task_losses_fn(y, {k: y_pred[k] for k in names})

        self._grads, self._task_losses = grads, task_losses
        self.epoch = 0
        self._probe()

    def _label_counts(self, y) -> np.ndarray:
        """Labelled rows per task, in ``task_names`` order (pool = any technique)."""
        def valid(k):
            return tf.reshape(tf.not_equal(tf.cast(y[k], tf.float32), -1.0), [-1])
        pooled = self.model._mtl_pooled_tasks
        out = []
        for t in self.task_names:
            if t == "propaganda_max_pool":
                v = tf.reduce_any(tf.stack([valid(k) for k in pooled], 1), 1)
            else:
                v = valid(t)
            out.append(int(tf.reduce_sum(tf.cast(v, tf.int32))))
        return np.array(out)

    def _probe(self):
        total, n_batches = 0.0, 0.0
        conflicts, pairs, loss_sum = 0.0, 0.0, 0.0
        for x, y in self.batches:
            g, losses, has = self._grads(x, y)
            has = tf.cast(has, tf.float32)
            total += g * has[:, None]
            n_batches += has
            loss_sum += losses * has
            unit = tf.math.l2_normalize(g, axis=1)
            both = has[:, None] * has[None, :]
            conflicts += tf.cast(tf.matmul(unit, unit, transpose_b=True) < 0, tf.float32) * both
            pairs += both
        mean = total / tf.maximum(n_batches, 1.0)[:, None]
        unit = tf.math.l2_normalize(mean, axis=1)
        self.records.append({
            "step": self.step, "epoch": self.epoch,
            "cosine": tf.matmul(unit, unit, transpose_b=True).numpy().tolist(),
            "norm": tf.norm(mean, axis=1).numpy().tolist(),
            "conflict_rate": (conflicts / tf.maximum(pairs, 1.0)).numpy().tolist(),
        })
        if self.lookahead_lr > 0:
            before = loss_sum / tf.maximum(n_batches, 1.0)
            self.records[-1]["affinity"] = self._affinity(unit, before, 1.0)
            self.records[-1]["affinity_small"] = self._affinity(unit, before, 0.1)
            self.records[-1]["loss"] = before.numpy().tolist()

    def _affinity(self, unit, before, scale: float) -> List[List[float]]:
        sizes = [int(np.prod(v.shape)) for v in self.shared]
        length = scale * self.lookahead_lr * float(np.sqrt(sum(sizes)))
        saved = [tf.identity(v) for v in self.shared]
        rows = []
        for i in range(len(self.task_names)):
            for v, d in zip(self.shared, tf.split(unit[i] * length, sizes)):
                v.assign_sub(tf.reshape(d, v.shape))
            after, n = 0.0, 0.0
            for x, y in self.batches:
                losses, has = self._task_losses(x, y)
                has = tf.cast(has, tf.float32)
                after += losses * has
                n += has
            rows.append((1.0 - (after / tf.maximum(n, 1.0)) / (before + 1e-12)).numpy().tolist())
            for v, orig in zip(self.shared, saved):
                v.assign(orig)
        return rows

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.every == 0:
            self._probe()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Shared train/eval harness
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Frozen stage 2: re-train each task's own block on the frozen shared part
# ---------------------------------------------------------------------------
# After stage 1 (any trainer here, with EpochScorer's best epoch restored) the
# encoder, the shared layers and the task-type ("family") blocks are frozen.
# Their outputs are computed once per text and cached, and only each task's
# own block (model.branch_layers[1][i]) and output layer keep training, on the
# real labels with stage 1's per-task loss and class weights. No balancer, no
# consistency term, no masking. With the shared part frozen the tasks share
# no trained weight, so each output can keep the epoch that is best for its
# own val F1; stage 1's block counts as epoch 0. The result is one model.

def table3_scores(trainer, labels, pred, thresholds, outputs):
    """Weighted F1 at the given cut-offs and AUC for each output, from arrays,
    computed as ``Trainer.evaluate`` does (labelled rows only; a pool's label
    is "any constituent positive")."""
    from sklearn.metrics import f1_score, roc_auc_score

    mask_value = trainer.mask_value
    scores = {}
    for k in outputs:
        y, p = np.asarray(labels[k]), np.asarray(pred[k])
        p = p[:, 0] if p.ndim > 1 else p
        valid = y != mask_value
        valid = valid[:, 0] if valid.ndim > 1 else valid
        y = y[valid]
        y = y[:, 0] if y.ndim > 1 else y
        if trainer.model.config.is_pool(k):
            y = (y > 0).astype(int)
        hard = (p > thresholds[k]).astype(int)[valid]
        scores[k] = {
            "f1": float(f1_score(y, hard, average="weighted", zero_division=0)),
            "auc": float(roc_auc_score(y, p[valid])),
        }
    return scores


class Stage2:
    """The frozen stage 2 for one trained model. ``outputs`` are the scored
    outputs (a task, or a pool standing for its constituent heads)."""

    def __init__(self, trainer, outputs: Sequence[str], batch_size: int = 256,
                 weight_decay: float = 0.01):
        model = trainer.model
        if len(model.branch_layers) != 2:
            raise ValueError("stage 2 needs a family level and a task level of blocks")
        if getattr(model, "_mtl_task_losses_fn", None) is None:
            raise ValueError("stage 2 needs the pooled-loss trainers (--pooled-propaganda-loss)")
        self.trainer, self.model = trainer, model
        self.names = list(model.task_names)
        self.outputs = list(outputs)
        self.batch_size, self.weight_decay = batch_size, weight_decay
        # family block that feeds each task block
        self.parent = [f for f, n in enumerate(model.config.task_structure[1]) for _ in range(n)]
        self.task_vars: Dict[str, List[tf.Variable]] = {}
        for k in self.outputs:
            tasks = model.config.get_pool_config(k)["tasks"] if model.config.is_pool(k) else [k]
            variables = []
            for t in tasks:
                for layer in model.branch_layers[1][self.names.index(t)]:
                    variables += layer.variables
                variables += model.output_layers[t].variables
            self.task_vars[k] = variables
        self.frozen_layers = ([model.encoder] if model.uses_encoder else []) + \
            list(model.shared_layers) + [l for branch in model.branch_layers[0] for l in branch]
        self.frozen_vars = [v for layer in self.frozen_layers for v in layer.variables]
        # reduce_retracing: batches padded to their own longest text come in many
        # widths; without it the encoder would be traced once per width.
        self._features_fn = tf.function(self._features, reduce_retracing=True)
        self._full = None  # (val_ds, val labels, test_ds, test labels): see use_full_model
        self._heads_fn = tf.function(lambda feats: self.heads(feats, training=False))

    def freeze(self):
        for layer in self.frozen_layers:
            layer.trainable = False

    def frozen_weights(self) -> List[np.ndarray]:
        return [v.numpy() for v in self.frozen_vars]

    def _features(self, x):
        h = self.model._encode_input(x, training=False)
        for layer in self.model.shared_layers:
            h = layer(h, training=False)
        feats = []
        for branch in self.model.branch_layers[0]:
            z = h
            for layer in branch:
                z = layer(z, training=False)
            feats.append(z)
        return feats

    def heads(self, feats, training=False):
        """Every task block and output layer on the cached family outputs,
        plus the pooled outputs: what the full model returns."""
        out = {}
        for i, name in enumerate(self.names):
            z = feats[self.parent[i]]
            for layer in self.model.branch_layers[1][i]:
                z = layer(z, training=training)
            out[name] = self.model.output_layers[name](z, training=training)
        out.update(self.model._compute_pooled_outputs(out))
        return out

    def cache(self, ds):
        """One pass over ``ds``: (family outputs, one array per family; labels,
        pooled labels included). Rows come in the order ``ds`` yields them."""
        feats = [[] for _ in self.model.branch_layers[0]]
        labels = {k: [] for k in self.names}
        for x, y in ds:
            for acc, f in zip(feats, self._features_fn(x)):
                acc.append(f.numpy())
            for k in self.names:
                labels[k].append(y[k].numpy())
        labels = {k: np.concatenate(v, axis=0) for k, v in labels.items()}
        labels.update(self.trainer._derive_pooled_labels(labels))
        return [np.concatenate(f, axis=0) for f in feats], labels

    def predict(self, feats, batch: int = 8192):
        parts: Dict[str, list] = {}
        for s in range(0, len(feats[0]), batch):
            out = self._heads_fn([tf.constant(f[s:s + batch]) for f in feats])
            for k, v in out.items():
                parts.setdefault(k, []).append(v.numpy())
        return {k: np.concatenate(v, axis=0) for k, v in parts.items()}

    def _labels(self, ds):
        labels = {k: [] for k in self.names}
        for _, y in ds:
            for k in self.names:
                labels[k].append(y[k].numpy())
        labels = {k: np.concatenate(v, axis=0) for k, v in labels.items()}
        labels.update(self.trainer._derive_pooled_labels(labels))
        return labels

    def use_full_model(self, val_ds, test_ds):
        """Score val and test through the whole network (``model.predict``,
        as EpochScorer does) instead of through cached features. On a GPU the
        two paths differ by ~1e-3 in a score, which is enough to move a J
        cut-off and the F1 at it; this way epoch 0 is exactly stage 1 and the
        assembled model scores exactly as each output's chosen epoch did.
        The cache is then only used to train."""
        self._full = (val_ds, self._labels(val_ds), test_ds, self._labels(test_ds))

    def score(self, val=None, test=None):
        """(val scores, test scores, cut-offs): the J cut-offs are tuned on val
        and used on both, as EpochScorer does. ``val`` / ``test`` are caches,
        ignored after ``use_full_model``."""
        if self._full is not None:
            val_ds, val_labels, test_ds, test_labels = self._full
            val_pred = self.model.predict(val_ds, verbose=0)
            test_pred = self.model.predict(test_ds, verbose=0)
        else:
            (val_feats, val_labels), (test_feats, test_labels) = val, test
            val_pred, test_pred = self.predict(val_feats), self.predict(test_feats)
        thresholds, _ = self.trainer._find_optimal_thresholds(val_labels, val_pred)
        return (table3_scores(self.trainer, val_labels, val_pred, thresholds, self.outputs),
                table3_scores(self.trainer, test_labels, test_pred, thresholds, self.outputs),
                thresholds)

    def get_blocks(self):
        return {k: [v.numpy() for v in vs] for k, vs in self.task_vars.items()}

    def set_blocks(self, blocks):
        for k, weights in blocks.items():
            for v, w in zip(self.task_vars[k], weights):
                v.assign(w)

    def fit(self, train, val=None, test=None, *, lr: float, epochs: int, seed: Optional[int] = None,
            only: Optional[Sequence[str]] = None):
        """Trains the task blocks (those of ``only``, default every output)
        from their current weights, keeps each output's own val-F1-best epoch
        (epoch 0 = the weights it started with; a later epoch must be strictly
        better) and leaves the model holding that assembly."""
        import time

        feats, labels = train
        n = len(feats[0])
        train_outputs = list(only) if only is not None else self.outputs
        variables = [v for k in train_outputs for v in self.task_vars[k]]
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=self.weight_decay)
        optimizer.build(variables)
        loss_fn = self.model._mtl_task_losses_fn
        F = [tf.constant(f) for f in feats]
        Y = {k: tf.constant(labels[k]) for k in self.names}

        @tf.function
        def step(idx):
            with tf.GradientTape() as tape:
                pred = self.heads([tf.gather(f, idx) for f in F], training=True)
                losses, has = loss_fn({k: tf.gather(Y[k], idx) for k in self.names},
                                      {k: pred[k] for k in self.names})
                loss = tf.reduce_sum(losses * has)
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            return loss

        def record(epoch, v, t, seconds):
            rec = {"epoch": epoch, "val": v, "test": t, "seconds": seconds}
            for m in ("f1", "auc"):
                rec[f"val_{m}"] = float(np.mean([s[m] for s in v.values()]))
                rec[f"test_{m}"] = float(np.mean([s[m] for s in t.values()]))
            return rec

        v, t, _ = self.score(val, test)
        records = [record(0, v, t, 0.0)]
        best_f1 = {k: v[k]["f1"] for k in self.outputs}
        best_epoch = {k: 0 for k in self.outputs}
        best_blocks = self.get_blocks()
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            start = time.time()
            order = rng.permutation(n).astype(np.int32)
            for s in range(0, n, self.batch_size):
                step(tf.constant(order[s:s + self.batch_size]))
            seconds = time.time() - start
            v, t, _ = self.score(val, test)
            records.append(record(epoch, v, t, seconds))
            for k in train_outputs:
                if v[k]["f1"] > best_f1[k]:
                    best_f1[k], best_epoch[k] = v[k]["f1"], epoch
                    best_blocks[k] = [w.numpy() for w in self.task_vars[k]]
            print(f"Stage 2 lr {lr:g} epoch {epoch}: val F1 {100 * records[-1]['val_f1']:.2f}  "
                  f"val AUC {100 * records[-1]['val_auc']:.2f}  ({seconds:.1f}s train)", flush=True)
        self.set_blocks(best_blocks)
        return {"lr": lr, "epochs": records, "chosen_epoch": best_epoch}

    def full_model_scores(self, val_ds, test_ds):
        """The model as it stands, scored by the library through the whole
        network: J cut-offs re-tuned on val, test scored once with them."""
        val_labels = self._labels(val_ds)
        pred = self.model.predict(val_ds, verbose=0)
        thresholds, _ = self.trainer._find_optimal_thresholds(val_labels, pred)
        out = {"thresholds": {k: thresholds[k] for k in self.outputs}}
        raw = {}
        for split, ds in (("val", val_ds), ("test", test_ds)):
            res = self.trainer.evaluate(ds, thresholds=thresholds)
            merged = {**res.get("task_metrics", {}), **res.get("pool_metrics", {})}
            out[split] = {k: {"f1": float(merged[k]["f1"]), "auc": float(merged[k]["auc"])}
                          for k in self.outputs}
            for m in ("f1", "auc"):
                out[f"{split}_{m}"] = float(np.mean([s[m] for s in out[split].values()]))
            raw[split] = res
        return out, raw


def run_stage2(trainer, train_ds, val_ds, test_ds, *, lrs: Sequence[str], epochs: int,
               outputs: Sequence[str], batch_size: int = 256, weight_decay: float = 0.01,
               seed: Optional[int] = None, predictions_prefix: Optional[str] = None,
               weights_path: Optional[str] = None) -> Dict[str, Any]:
    """Stage 2 at each learning rate in ``lrs`` (strings, kept as the result
    keys), all from the same stage-1 model and the same cache: the task blocks
    are put back to stage 1 after each rate. Writes one predictions file per
    rate (``<predictions_prefix>_stage2_lr<lr>_predictions.npz``) and, if
    ``weights_path`` is given, the stage-1 weights (``model.variables`` order
    before freezing, with their names)."""
    import time

    model = trainer.model
    if weights_path:
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        np.savez_compressed(
            weights_path, names=np.array([v.name for v in model.variables]),
            **{f"w{i}": v.numpy() for i, v in enumerate(model.variables)})
        print(f"Wrote {weights_path} ({len(model.variables)} variables)")
    s2 = Stage2(trainer, outputs, batch_size=batch_size, weight_decay=weight_decay)
    s2.freeze()
    frozen_before = s2.frozen_weights()
    # every variable stage 2 may not train (frozen part, and the blocks of
    # outputs not in ``outputs``, e.g. a hidden one)
    trained_ids = {id(v) for vs in s2.task_vars.values() for v in vs}
    untrained = [v for v in model.variables if id(v) not in trained_ids]
    untrained_before = [v.numpy() for v in untrained]
    start = time.time()
    train = s2.cache(train_ds)
    cache_seconds = time.time() - start
    full_val = model.predict(val_ds, verbose=0)
    cached_val = s2.predict(s2.cache(val_ds)[0])
    s2.use_full_model(val_ds, test_ds)
    cache_gap = max(float(np.max(np.abs(np.asarray(full_val[k]) - cached_val[k])))
                    for k in cached_val)
    print(f"Stage 2 cache: {len(train[0][0]):,} train rows in {cache_seconds:.0f}s; largest gap "
          f"between cached and full-model val scores {cache_gap:.2e}", flush=True)
    stage1_blocks = s2.get_blocks()
    stage1, _ = s2.full_model_scores(val_ds, test_ds)
    out: Dict[str, Any] = {
        "outputs": list(outputs), "stage2_epochs": epochs, "batch_size": batch_size,
        "weight_decay": weight_decay, "cache_seconds": cache_seconds,
        "cache_rows": {"train": len(train[0][0])},
        "cache_max_abs_gap": cache_gap, "stage1": stage1, "stage1_weights": weights_path,
        "runs": {},
    }
    for lr in lrs:
        run = s2.fit(train, lr=float(lr), epochs=epochs, seed=seed)
        run["assembled"], raw = s2.full_model_scores(val_ds, test_ds)
        # each output of the assembled model against its chosen epoch's record
        run["assembly_max_f1_gap"] = max(
            abs(run["assembled"][split][k]["f1"]
                - run["epochs"][run["chosen_epoch"][k]][split][k]["f1"])
            for split in ("val", "test") for k in outputs)
        if predictions_prefix:
            arrays = {}
            for split, res in raw.items():
                for name, prob in (res.get("probabilities_dict") or {}).items():
                    arrays[f"{split}/prob/{name}"] = np.asarray(prob, dtype=np.float32)
                for name, label in (res.get("labels_dict") or {}).items():
                    arrays[f"{split}/label/{name}"] = np.asarray(label, dtype=np.float32)
            path = f"{predictions_prefix}_stage2_lr{lr}_predictions.npz"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez_compressed(path, **arrays)
            run["predictions"] = os.path.basename(path)
            print(f"Wrote {path} ({len(arrays)} arrays)")
        a = run["assembled"]
        print(f"Stage 2 lr {lr}: assembled val F1 {100 * a['val_f1']:.2f} test F1 "
              f"{100 * a['test_f1']:.2f}  (stage 1: {100 * stage1['val_f1']:.2f} / "
              f"{100 * stage1['test_f1']:.2f}); chosen epochs {run['chosen_epoch']}", flush=True)
        out["runs"][lr] = run
        s2.set_blocks(stage1_blocks)
    out["frozen_unchanged"] = all(
        np.array_equal(a, b) for a, b in zip(frozen_before, s2.frozen_weights()))
    out["untrained_unchanged"] = all(
        np.array_equal(a, v.numpy()) for a, v in zip(untrained_before, untrained))
    print(f"Stage 2: frozen weights unchanged: {out['frozen_unchanged']}; every variable "
          f"outside the trained blocks unchanged: {out['untrained_unchanged']}", flush=True)
    return out


def run_experiment(
    config: ModelConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 4e-5,
    optimize_thresholds: bool = True,
    verbose: int = 1,
    show_structure: bool = True,
    trainer: Optional[Trainer] = None,
    task_type_mapping: Optional[Dict[str, str]] = None,
    training_config: Optional[TrainingConfig] = None,
    class_weights: Optional[List[Dict[int, float]]] = None,
    task_weights: Optional[List[float]] = None,
    test_ds: Optional[tf.data.Dataset] = None,
    trainer_factory: Optional[Any] = None,
    model_cls: type = MultiTaskModel,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> Dict[str, Any]:
    # Resolve the TrainingConfig before building the model so we can pass
    # dropout_rate into MultiTaskModel — the model is constructed before the
    # Trainer, so the trainer can't push it in later.
    if trainer is None:
        resolved_training_config = training_config or TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    else:
        resolved_training_config = trainer.config

    model = model_cls(config, dropout_rate=resolved_training_config.dropout_rate)
    # Under Keras 3 a tf_keras encoder (HuggingFace TF) is not tracked, so it
    # would silently stay frozen. Fail instead; TF_USE_LEGACY_KERAS=1 fixes it.
    if model.uses_encoder and model.encoder.trainable:
        tracked = {id(v) for v in model.trainable_variables}
        missing = [v for v in model.encoder.trainable_variables if id(v) not in tracked]
        if missing:
            raise RuntimeError(
                f"{len(missing)} trainable encoder variables are not in "
                "model.trainable_variables, so the encoder would never train. "
                "Set TF_USE_LEGACY_KERAS=1 before importing TensorFlow."
            )
    if show_structure:
        print("\nModel structure:")
        print(model.summary_tree())
        print("Task paths:")
        for info in model.get_task_info():
            print(
                f"  {info['output_idx']:>2}  {info['name']:<35} "
                f"path={info['parent_branch_path']}"
            )

    if trainer is None:
        trainer = (trainer_factory or Trainer)(model, resolved_training_config)

    _, thresholds = trainer.fit(
        train_ds,
        val_ds,
        num_tasks=len(config.task_names),
        class_weights=class_weights,
        task_weights=task_weights,
        task_type_mapping=task_type_mapping,
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        optimize_thresholds=optimize_thresholds,
        verbose=verbose,
        callbacks=callbacks,
    )
    print("Thresholds:", thresholds)

    results = trainer.evaluate(val_ds, thresholds=thresholds)
    results["history"] = {k: [float(v) for v in vs] for k, vs in trainer.history.history.items()}
    print("Overall metrics:", results.get("overall_metrics", {}))
    print("Task-specific metrics:", results.get("task_metrics", {}))
    if test_ds is not None:
        # Thresholds stay the val-tuned ones; the test split is never fit on.
        print("\n--- Held-out test split ---")
        test_results = trainer.evaluate(test_ds, thresholds=thresholds)
        print("Test overall metrics:", test_results.get("overall_metrics", {}))
        print("Test task-specific metrics:", test_results.get("task_metrics", {}))
        print("Test pool metrics:", test_results.get("pool_metrics", {}))
        results["test"] = test_results
    return results


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_propaganda(args: argparse.Namespace) -> None:
    print("\n=== Experiment: propaganda ===")
    cols = list(PROPAGANDA_TECHNIQUE_COLUMNS)
    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        multilabel_groups={"propaganda_techniques": cols},
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    config = ModelConfig(
        task_structure=[[1]],
        task_names=["propaganda_techniques"],
        task_types=["multilabel"],
        num_classes_per_task=[len(cols)],
        encoder_config=encoder_config,
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_propaganda_cols_binary(args: argparse.Namespace) -> None:
    print("\n=== Experiment: propaganda_cols_binary ===")
    task_names = list(PROPAGANDA_BINARY_TASK_NAMES)
    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        task_names,
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
    )
    config = ModelConfig(
        task_structure=[[len(task_names)]],
        task_names=task_names,
        task_types=["binary"] * len(task_names),
        num_classes_per_task=[2] * len(task_names),
        encoder_config=encoder_config,
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_discourse_dims(args: argparse.Namespace) -> None:
    print("\n=== Experiment: discourse_dims ===")
    cols = list(DISCOURSE_REGRESSION_COLUMNS)
    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        cols,
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    config = ModelConfig(
        task_structure=[[len(cols)]],
        task_names=cols,
        task_types=["regression"] * len(cols),
        num_classes_per_task=[1] * len(cols),
        encoder_config=encoder_config,
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=False,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_discourse_binary(args: argparse.Namespace) -> None:
    print("\n=== Experiment: discourse_binary ===")
    cols = list(DISCOURSE_BINARY_COLUMNS)
    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        cols,
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    config = ModelConfig(
        task_structure=[[len(cols)]],
        task_names=cols,
        task_types=["binary"] * len(cols),
        num_classes_per_task=[2] * len(cols),
        encoder_config=encoder_config,
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


IAC_SINGLE_TASK_TRAINING_CONFIG = TrainingConfig(
    optimizer_type="adamw",
    learning_rate=3e-4,
    weight_decay_rate=0.01,
    warmup_percent=5,
    batch_size=128,
    dropout_rate=0.2,
)


def run_discourse_binary_single_task(args: argparse.Namespace) -> None:
    """Train each IAC binary dimension in an isolated single-task model."""
    print("\n=== Experiment: discourse_binary_single_task ===")
    training_config = replace(
        IAC_SINGLE_TASK_TRAINING_CONFIG,
        epochs=args.epochs,
    )
    print(
        "Training config: "
        f"optimizer={training_config.optimizer_type} "
        f"lr={training_config.learning_rate} "
        f"wd={training_config.weight_decay_rate} "
        f"warmup={training_config.warmup_percent}% "
        f"batch={training_config.batch_size} "
        f"dropout={training_config.dropout_rate} "
        f"epochs={training_config.epochs}"
    )

    encoder_config = ENCODER_CONFIGS[args.encoder]
    summary: List[Dict[str, Any]] = []

    for col in DISCOURSE_BINARY_COLUMNS:
        print("\n" + "=" * 80)
        print(f"Single-task model: {col}")
        print("=" * 80)

        train_ds, val_ds, _ = load_splits(
            [col],
            encoder_config=encoder_config,
            batch_size=training_config.batch_size,
            data_dir=args.data_dir,
        )
        config = ModelConfig(
            task_structure=[[1]],
            task_names=[col],
            task_types=["binary"],
            num_classes_per_task=[2],
            encoder_config=encoder_config,
        )
        results = run_experiment(
            config,
            train_ds,
            val_ds,
            optimize_thresholds=not args.no_optimize_thresh,
            verbose=args.verbose,
            show_structure=args.show_structure,
            training_config=training_config,
        )

        task_metrics = (results.get("task_metrics", {}) or {}).get(col, {})
        overall = results.get("overall_metrics", {}) or {}
        metrics = task_metrics or overall
        summary.append({"task": col, **metrics})

    print("\n=== discourse_binary_single_task summary ===")
    for row in summary:
        print(
            f"{row['task']}: "
            f"f1={row.get('f1', 'NA')} "
            f"ap={row.get('average_precision', 'NA')} "
            f"auc={row.get('auc', 'NA')} "
            f"acc={row.get('accuracy', 'NA')}"
        )


def run_propaganda_binary(args: argparse.Namespace) -> None:
    print("\n=== Experiment: propaganda_binary ===")
    binary_cols = list(DISCOURSE_BINARY_COLUMNS)
    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        label_columns=binary_cols,
        multilabel_groups={"propaganda_techniques": list(PROPAGANDA_TECHNIQUE_COLUMNS)},
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    task_names = ["propaganda_techniques"] + binary_cols
    task_types = ["multilabel"] + ["binary"] * len(binary_cols)
    num_classes_per_task = [len(PROPAGANDA_TECHNIQUE_COLUMNS)] + [2] * len(binary_cols)
    config = ModelConfig(
        task_structure=[[len(task_names)]],
        task_names=task_names,
        task_types=task_types,
        num_classes_per_task=num_classes_per_task,
        encoder_config=encoder_config,
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def _prepare_all_task_types_data_and_config(
    args: argparse.Namespace,
    *,
    task_structure: Sequence[Sequence[int]],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, ModelConfig]:
    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        label_columns=list(DISCOURSE_REGRESSION_COLUMNS) + list(DISCOURSE_BINARY_COLUMNS),
        multilabel_groups={"propaganda_techniques": list(PROPAGANDA_TECHNIQUE_COLUMNS)},
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    task_names = (
        ["propaganda_techniques"]
        + list(DISCOURSE_REGRESSION_COLUMNS)
        + list(DISCOURSE_BINARY_COLUMNS)
    )
    task_types = (
        ["multilabel"]
        + ["regression"] * len(DISCOURSE_REGRESSION_COLUMNS)
        + ["binary"] * len(DISCOURSE_BINARY_COLUMNS)
    )
    num_classes_per_task = (
        [len(PROPAGANDA_TECHNIQUE_COLUMNS)]
        + [1] * len(DISCOURSE_REGRESSION_COLUMNS)
        + [2] * len(DISCOURSE_BINARY_COLUMNS)
    )
    config = ModelConfig(
        task_structure=[list(level) for level in task_structure],
        task_names=task_names,
        task_types=task_types,
        num_classes_per_task=num_classes_per_task,
        encoder_config=encoder_config,
    )
    return train_ds, val_ds, config


def _prepare_all_task_types_binary_prop_data_and_config(
    args: argparse.Namespace,
    *,
    task_structure: Sequence[Sequence[int]],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, ModelConfig]:
    propaganda_task_names = list(PROPAGANDA_BINARY_TASK_NAMES)
    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        label_columns=(
            propaganda_task_names
            + list(DISCOURSE_REGRESSION_COLUMNS)
            + list(DISCOURSE_BINARY_COLUMNS)
        ),
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
    )
    task_names = (
        propaganda_task_names
        + list(DISCOURSE_REGRESSION_COLUMNS)
        + list(DISCOURSE_BINARY_COLUMNS)
    )
    task_types = (
        ["binary"] * len(propaganda_task_names)
        + ["regression"] * len(DISCOURSE_REGRESSION_COLUMNS)
        + ["binary"] * len(DISCOURSE_BINARY_COLUMNS)
    )
    num_classes_per_task = (
        [2] * len(propaganda_task_names)
        + [1] * len(DISCOURSE_REGRESSION_COLUMNS)
        + [2] * len(DISCOURSE_BINARY_COLUMNS)
    )
    config = ModelConfig(
        task_structure=[list(level) for level in task_structure],
        task_names=task_names,
        task_types=task_types,
        num_classes_per_task=num_classes_per_task,
        encoder_config=encoder_config,
    )
    return train_ds, val_ds, config


def run_all_task_types(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types ===")
    train_ds, val_ds, config = _prepare_all_task_types_data_and_config(
        args,
        task_structure=[[1 + len(DISCOURSE_REGRESSION_COLUMNS) + len(DISCOURSE_BINARY_COLUMNS)]],
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_all_task_types_binary_prop(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types_binary_prop ===")
    train_ds, val_ds, config = _prepare_all_task_types_binary_prop_data_and_config(
        args,
        task_structure=[[
            len(PROPAGANDA_BINARY_TASK_NAMES)
            + len(DISCOURSE_REGRESSION_COLUMNS)
            + len(DISCOURSE_BINARY_COLUMNS)
        ]],
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_all_task_types_binary_prop_grouped(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types_binary_prop_grouped ===")
    train_ds, val_ds, config = _prepare_all_task_types_binary_prop_data_and_config(
        args,
        task_structure=[[
            3
        ], [
            len(PROPAGANDA_BINARY_TASK_NAMES),
            len(DISCOURSE_REGRESSION_COLUMNS),
            len(DISCOURSE_BINARY_COLUMNS),
        ]],
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_all_task_types_binary_prop_deep_grouped(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types_binary_prop_deep_grouped ===")
    train_ds, val_ds, config = _prepare_all_task_types_binary_prop_data_and_config(
        args,
        task_structure=[
            [2],
            [1, 2],
            [
                len(PROPAGANDA_BINARY_TASK_NAMES),
                len(DISCOURSE_REGRESSION_COLUMNS),
                len(DISCOURSE_BINARY_COLUMNS),
            ],
        ],
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_all_task_types_pair_by_dimension(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types_pair_by_dimension ===")
    paired_names = []
    paired_types = []
    paired_num_classes = []
    for reg_col, bi_col in zip(DISCOURSE_REGRESSION_COLUMNS, DISCOURSE_BINARY_COLUMNS):
        paired_names.extend([reg_col, bi_col])
        paired_types.extend(["regression", "binary"])
        paired_num_classes.extend([1, 2])

    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        label_columns=list(DISCOURSE_REGRESSION_COLUMNS) + list(DISCOURSE_BINARY_COLUMNS),
        multilabel_groups={"propaganda_techniques": list(PROPAGANDA_TECHNIQUE_COLUMNS)},
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    config = ModelConfig(
        task_structure=[[1 + len(DISCOURSE_REGRESSION_COLUMNS)], [1] + [2] * len(DISCOURSE_REGRESSION_COLUMNS)],
        task_names=["propaganda_techniques"] + paired_names,
        task_types=["multilabel"] + paired_types,
        num_classes_per_task=[len(PROPAGANDA_TECHNIQUE_COLUMNS)] + paired_num_classes,
        encoder_config=encoder_config,
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_all_task_types_grouped(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types_grouped ===")
    train_ds, val_ds, config = _prepare_all_task_types_data_and_config(
        args,
        task_structure=[[3], [1, len(DISCOURSE_REGRESSION_COLUMNS), len(DISCOURSE_BINARY_COLUMNS)]],
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
    )


def run_all_task_types_deep_grouped(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types_deep_grouped ===")
    train_ds, val_ds, config = _prepare_all_task_types_data_and_config(
        args,
        task_structure=[
            [2],
            [1, 2],
            [1, len(DISCOURSE_REGRESSION_COLUMNS), len(DISCOURSE_BINARY_COLUMNS)],
        ],
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure
    )

PAPER_TRAINING_CONFIG = TrainingConfig(
    optimizer_type="adamw",
    learning_rate=0.0003,
    weight_decay_rate=0.01,
    dropout_rate=0.4,
    batch_size=256,
    warmup_percent=5
)

# Head sizes for paper_replication (--arch). "paper" is what the paper
# describes; "legacy" is what multitask_basic/models/model.py actually builds
# (each size appears twice because it zips one dense layer per size pair).
# "wide" keeps the paper's depth at the legacy widths; "medium" sits between.
PAPER_ARCHITECTURES: Dict[str, Tuple[List[int], List[List[int]]]] = {
    "paper": ([64, 32], [[32, 16], [16, 8]]),
    "medium": ([128, 64], [[64, 32], [32, 16]]),
    "wide": ([256, 128], [[128, 64], [64, 32]]),
    "legacy": ([256, 256, 128, 128], [[128, 128, 64, 64], [64, 64, 32, 32]]),
    # Shared-depth study (modal_replicate.py::depth). Names are
    # s<shared layers>_t<task-type layers>_k<task-specific layers>[_variant].
    # A: shared depth 1-6, branches fixed at wide's 128>64 / 64>32.
    "s1_t2_k2": ([128], [[128, 64], [64, 32]]),
    "s3_t2_k2": ([256, 256, 128], [[128, 64], [64, 32]]),
    "s4_t2_k2": ([256, 256, 128, 128], [[128, 64], [64, 32]]),
    "s6_t2_k2": ([256, 256, 256, 128, 128, 128], [[128, 64], [64, 32]]),
    # B: shared width at depth 2, branches fixed.
    "s2_t2_k2_narrow": ([128, 128], [[128, 64], [64, 32]]),
    "s2_t2_k2_wider": ([512, 256], [[128, 64], [64, 32]]),
    # C: six layers in total, moved between shared and branches.
    "s3_t1_k2": ([256, 256, 128], [[128], [64, 32]]),
    "s3_t2_k1": ([256, 256, 128], [[128, 64], [32]]),
    "s4_t1_k1": ([256, 256, 128, 128], [[64], [32]]),
    "s1_t3_k2": ([256], [[128, 128, 64], [64, 32]]),
    # D: shared fixed at wide's 256>128, branch depth changed.
    "s2_t1_k1": ([256, 128], [[128], [64]]),
    "s2_t3_k3": ([256, 128], [[128, 128, 64], [64, 64, 32]]),
}


def run_paper_replication(args: argparse.Namespace) -> None:
    print("\n=== Experiment: paper_replication ===")
    # --tasks keeps a subset of the ten Table-3 outputs, for single-task and
    # leave-one-task-out runs; "propaganda_max_pool" stands for the 18
    # technique heads and their pool. Each kept task-type keeps its branch.
    keep = args.tasks.split(",") if args.tasks else list(PAPER_TABLE3_MULTITASK_F1)
    unknown = [t for t in keep if t not in PAPER_TABLE3_MULTITASK_F1]
    if unknown:
        raise SystemExit(f"--tasks: unknown outputs {unknown}; choose from "
                         f"{list(PAPER_TABLE3_MULTITASK_F1)}")
    propaganda_task_names = (list(PROPAGANDA_BINARY_TASK_NAMES)
                             if "propaganda_max_pool" in keep else [])
    wa_task_names = ["WA_bi"] if "WA_bi" in keep else []
    discourse_task_names = [t for t in DISCOURSE_BINARY_COLUMNS if t in keep]
    groups = [(g, ty) for g, ty in ((propaganda_task_names, "propaganda_binary"),
                                    (wa_task_names, "wa_bi"),
                                    (discourse_task_names, "discourse")) if g]
    all_task_names = [t for g, _ in groups for t in g]
    print(f"Outputs: {keep}")

    encoder_config = ENCODER_CONFIGS[args.encoder]
    if args.zero_fill_keep_class_weights and not args.propaganda_zero_fill:
        raise SystemExit("--zero-fill-keep-class-weights needs --propaganda-zero-fill")
    zero_fill = propaganda_task_names if args.propaganda_zero_fill else []
    # Class weights count the filled labels unless told to keep the original ones.
    weight_fill = [] if args.zero_fill_keep_class_weights else zero_fill
    # The IAC labels on Propaganda rows are rule-made from the technique tags
    # (not in the paper's text, but in its Table 1 counts); optionally drop them
    # from training. Val/test keep them, so scores stay comparable.
    # --no-rule-labels removes them from every split (train, val and test).
    if args.no_rule_labels and args.mask_news_discourse_train:
        raise SystemExit("--no-rule-labels already masks train; drop --mask-news-discourse-train")
    news_mask = (("Propaganda", discourse_task_names)
                 if (args.mask_news_discourse_train or args.no_rule_labels)
                 and discourse_task_names else None)
    # --hide-task removes one output's labels from every train row; the head
    # stays in the model and is scored, but is left out of epoch selection.
    if args.hide_task and args.hide_task not in keep:
        raise SystemExit(f"--hide-task {args.hide_task!r} is not a trained output: {keep}")
    hidden_columns = (propaganda_task_names if args.hide_task == "propaganda_max_pool"
                      else [args.hide_task] if args.hide_task else [])
    if args.hide_task and not args.early_stop_metric:
        raise SystemExit("--hide-task needs --early-stop-metric (the scorer leaves it out)")
    # One entry per trained output, so the technique heads count as their pool.
    label_share = {}
    for item in filter(None, args.label_share.split(",")):
        col, share = item.split("=")
        if col not in all_task_names or not 0 < float(share) <= 1:
            raise SystemExit(f"--label-share {item!r}: needs a trained label column "
                             "and a share in (0, 1]")
        if float(share) < 1:
            label_share[col] = float(share)
    if args.length_chunk_batches < 0 or args.evals_per_epoch < 1:
        raise SystemExit("--length-chunk-batches must be >= 0 and --evals-per-epoch >= 1")
    if args.length_chunk_batches and (args.stratify_batches
                                      or args.encoder in OPENAI_EMBEDDING_MODELS):
        raise SystemExit("--length-chunk-batches can't be combined with --stratify-batches "
                         "or stored embeddings")
    if args.evals_per_epoch > 1 and (not args.early_stop_metric or args.self_distill):
        raise SystemExit("--evals-per-epoch needs --early-stop-metric and can't be combined "
                         "with --self-distill (which switches tasks at epoch ends)")
    if not 0 < args.layer_lr_decay <= 1 or args.clipnorm < 0:
        raise SystemExit("--layer-lr-decay must be in (0, 1] and --clipnorm >= 0")
    finetune = None
    if args.layer_lr_decay != 1.0 or args.clipnorm or args.no_decay_bias_norm:
        if not args.pooled_propaganda_loss:
            raise SystemExit("--layer-lr-decay / --clipnorm / --no-decay-bias-norm need "
                             "--pooled-propaganda-loss")
        if args.layer_lr_decay != 1.0 and args.encoder in OPENAI_EMBEDDING_MODELS:
            raise SystemExit("--layer-lr-decay needs an encoder that is trained")
        finetune = dict(layer_lr_decay=args.layer_lr_decay, clipnorm=args.clipnorm,
                        decay_bias_norm=not args.no_decay_bias_norm)
    if args.repeat_train > 1 and (args.self_distill or args.stratify_batches):
        raise SystemExit("--repeat-train can't be combined with --self-distill / "
                         "--stratify-batches")
    stratify_on = ([propaganda_task_names] if propaganda_task_names else []) + \
        [[t] for t in wa_task_names + discourse_task_names]
    embedding_source = None
    if args.encoder in OPENAI_EMBEDDING_MODELS:
        # There are no tokens and no encoder in the model, so everything that
        # masks words, reads the encoder or re-tokenizes the CSVs is out.
        if (args.mlm_lambda is not None or args.masked_label_weight or args.mask_sweep
                or (args.consistency_weight and not args.consistency_teacher)
                or args.probe_tasks or args.probe_untrained):
            raise SystemExit(f"--encoder {args.encoder} trains on stored embeddings: it can't "
                             "be combined with --mlm-lambda, --masked-label-weight, "
                             "--consistency-weight on masked copies, --mask-sweep or "
                             "--probe-tasks")
        embedding_source = openai_embedding_source(args.encoder, args.embedding_cache)
    train_ds, val_ds, test_ds = load_splits(
        label_columns=all_task_names,
        encoder_config=encoder_config,
        embedding_source=embedding_source,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
        max_length=args.max_length,
        seed=args.seed,
        drop_unlabeled_train=args.drop_unlabeled_train,
        zero_fill_train=zero_fill,
        stratify_on=stratify_on if args.stratify_batches else (),
        mask_train_source=news_mask,
        mask_source_all_splits=args.no_rule_labels,
        hide_train_columns=hidden_columns,
        train_row_ids=args.self_distill,
        train_label_share=label_share,
        repeat_train=args.repeat_train,
        length_chunk_batches=args.length_chunk_batches,
    )
    task_types = ["binary"] * len(all_task_names)
    num_classes_per_task = [2] * len(all_task_names)

    task_type_mapping = {t: ty for g, ty in groups for t in g}
    class_weights = compute_full_train_binary_class_weights(
        data_dir=args.data_dir,
        task_names=all_task_names,
        label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
        zero_fill=weight_fill,
        mask_source=news_mask,
    )
    
    shared_layer_sizes, branch_layer_sizes = PAPER_ARCHITECTURES[args.arch]
    config = ModelConfig(
        task_structure=[
            [len(groups)],
            [len(g) for g, _ in groups],
        ],
        shared_layer_sizes=shared_layer_sizes,
        branch_layer_sizes=branch_layer_sizes,
        task_names=all_task_names,
        task_types=task_types,
        num_classes_per_task=num_classes_per_task,
        output_pools={
            "propaganda_max_pool": {
                "tasks": propaganda_task_names,
                "strategy": args.propaganda_pool,
                "task_type": "binary",
                "num_classes": 2,
            },
        } if propaganda_task_names else {},
        encoder_config=encoder_config,
    )
    # PAPER_TRAINING_CONFIG carries no epoch count, so without this the
    # TrainingConfig default (20) silently wins over --epochs. Early stopping is
    # the paper's: "stop training after two epochs without a decrease in loss".
    training_config = replace(
        PAPER_TRAINING_CONFIG,
        epochs=args.epochs,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
    )
    # The data pipeline is batched with --batch-size, so the config must agree.
    training_config = replace(training_config, batch_size=args.batch_size)
    # run_experiment ignores its learning_rate argument once a training_config
    # is given, so --learning-rate only takes effect through here. Only an
    # explicit flag overrides: the parser's 4e-5 default is not the paper's.
    if args.learning_rate_explicit:
        training_config = replace(training_config, learning_rate=args.learning_rate)
    if args.dropout is not None:
        training_config = replace(training_config, dropout_rate=args.dropout)
    if args.weight_decay is not None:
        training_config = replace(training_config, weight_decay_rate=args.weight_decay)
    if args.warmup_percent is not None:
        training_config = replace(training_config, warmup_percent=args.warmup_percent)
    print(f"Architecture: {args.arch} shared={shared_layer_sizes} branches={branch_layer_sizes}")
    print(f"Training config: {training_config}")
    # Without a mapping every head is its own task-type: nu_k per head and no
    # 1/|T_k| within-type factor (the library's legacy per-task weighting).
    if args.no_task_type_grouping:
        task_type_mapping = None
    # --unweighted-loss: no task weights and no class weights. Every output (the
    # pooled propaganda output, WA_bi, each IAC label) gets weight 1 and both
    # classes weight 1, so the loss is the plain mean BCE over all visible label
    # cells (no nu_k = 1/|D_k|, no 1/|T_k| split, no class balancing).
    uniform_task_weights = None
    if args.unweighted_loss:
        task_type_mapping = None
        uniform_task_weights = [1.0] * len(all_task_names)
        class_weights = [{0: 1.0, 1: 1.0} for _ in all_task_names]
    print(f"Task-type grouping: {'off' if task_type_mapping is None else 'on'}")
    task_loss_options: Dict[str, Tuple[float, float]] = {}
    for task_name in args.imbalance_tasks.split(",") if (args.logit_adjust or args.focal_gamma) else []:
        if task_name not in discourse_task_names + ["WA_bi"]:
            raise SystemExit(f"--imbalance-tasks: unknown or pooled task {task_name!r}")
        idx = all_task_names.index(task_name)
        shift = 0.0
        if args.logit_adjust:
            # w_c = N / (2 n_c), so w0 / w1 = n1 / n0, the train prior odds of class 1.
            # Logit adjustment replaces the class weights rather than adding to them.
            shift = float(np.log(class_weights[idx][0] / class_weights[idx][1]))
            class_weights[idx] = {0: 1.0, 1: 1.0}
        task_loss_options[task_name] = (shift, args.focal_gamma)
    if task_loss_options and not args.pooled_propaganda_loss:
        raise SystemExit("--logit-adjust / --focal-gamma need --pooled-propaganda-loss")

    trainer_factory = None
    teacher_mode = bool(args.consistency_teacher)
    if teacher_mode and not (args.loss_balancer == "famo" and args.mlm_lambda is None
                             and args.consistency_weight > 0 and not args.masked_label_weight):
        raise SystemExit("--consistency-teacher is built on FAMO with no masking: it needs "
                         "--loss-balancer famo and --consistency-weight, and no --mlm-lambda "
                         "or --masked-label-weight")
    if teacher_mode and args.consistency_teacher == "current" and args.consistency_student_dropout == "off":
        raise SystemExit("--consistency-teacher current with --consistency-student-dropout off "
                         "compares the model with itself: the loss is always 0")
    if not teacher_mode and (args.consistency_student_dropout != "on" or args.ema_decay != 0.99):
        raise SystemExit("--consistency-student-dropout / --ema-decay need --consistency-teacher")
    label_aware = (args.masked_label_weight > 0 or args.consistency_weight > 0) and not teacher_mode
    if label_aware and not (args.loss_balancer == "famo" and args.mlm_mode == "fixed"
                            and args.mlm_lambda is not None):
        raise SystemExit("--masked-label-weight / --consistency-weight are built on famo_fixed: "
                         "they need --loss-balancer famo, --mlm-lambda and --mlm-mode fixed")
    if args.masked_label_weight < 0 or args.consistency_weight < 0:
        raise SystemExit("--masked-label-weight / --consistency-weight must be >= 0")
    if args.consistency_keep and not args.consistency_weight:
        raise SystemExit("--consistency-keep needs --consistency-weight")
    if not 0 <= args.consistency_keep <= 0.5:
        raise SystemExit("--consistency-keep is a share per side, in [0, 0.5]")
    ssl_pairs = None
    ssl_defaults = (args.ssl_rates == SSL_DEFAULT_RATES and args.ssl_rate_weight == 0.1
                    and args.ssl_entropy_weight == 0.0 and args.ssl_entropy_start_epoch == 2.0)
    if args.semi_supervised:
        if not args.pooled_propaganda_loss:
            raise SystemExit("--semi-supervised needs --pooled-propaganda-loss")
        if (args.loss_balancer or args.mlm_lambda is not None or teacher_mode or label_aware
                or args.propaganda_zero_fill):
            raise SystemExit("--semi-supervised is built on the plain pooled loss: no "
                             "--loss-balancer, --mlm-lambda, consistency or zero-fill")
        if not propaganda_task_names or not wa_task_names:
            raise SystemExit("--semi-supervised reads each row's source from the propaganda "
                             "and WA_bi labels, so it needs both outputs")
        try:
            ssl_pairs = parse_ssl_rates(args.ssl_rates)
        except ValueError as exc:
            raise SystemExit(f"--ssl-rates: {exc}")
        bad = [f"{h}@{s}" for h, s, _ in ssl_pairs
               if s not in SSL_SOURCES or (h != SSL_POOL and h not in all_task_names)]
        if bad:
            raise SystemExit(f"--ssl-rates: unknown head or source in {bad}")
        if args.ssl_rate_weight < 0 or args.ssl_entropy_weight < 0 or not (
                args.ssl_rate_weight or args.ssl_entropy_weight):
            raise SystemExit("--ssl-rate-weight / --ssl-entropy-weight must be >= 0, one > 0")
    elif not ssl_defaults:
        raise SystemExit("--ssl-rates / --ssl-*-weight / --ssl-entropy-start-epoch need "
                         "--semi-supervised")
    if args.consistency_floor and not args.consistency_keep:
        raise SystemExit("--consistency-floor needs --consistency-keep")
    if not 0 <= args.consistency_floor < 0.5:
        raise SystemExit("--consistency-floor is a score distance from 0 or 1, in [0, 0.5)")
    if (args.propaganda_pool != "max" or args.technique_loss_weight) and not args.pooled_propaganda_loss:
        raise SystemExit("--propaganda-pool / --technique-loss-weight need --pooled-propaganda-loss")
    if args.technique_loss_weight < 0:
        raise SystemExit("--technique-loss-weight must be >= 0")
    if args.semi_supervised and args.propaganda_pool != "max":
        raise SystemExit("--semi-supervised pools propaganda with a max of its own: "
                         "it needs --propaganda-pool max")
    if args.self_distill:
        # Plain pooled loss, or famo_consf1_m5-style (FAMO + label-aware consistency).
        famo_ok = label_aware and args.loss_balancer == "famo"
        if (not args.pooled_propaganda_loss or args.early_stop_metric != "f1"
                or (args.loss_balancer and not famo_ok)
                or (args.mlm_lambda is not None and not famo_ok) or teacher_mode
                or args.semi_supervised or args.main_task
                or task_loss_options or args.technique_loss_weight):
            raise SystemExit("--self-distill is built on the plain pooled loss or on FAMO with "
                             "--consistency-weight (famo_consf1_m5): it needs "
                             "--pooled-propaganda-loss and --early-stop-metric f1, and no "
                             "other balancer, teacher consistency, --semi-supervised, "
                             "--main-task, logit adjustment / focal loss or technique loss")
    elif args.self_distill_patience != 2 or args.self_distill_signal != "f1":
        raise SystemExit("--self-distill-patience / --self-distill-signal need --self-distill")
    scored_outputs = [k for k in PAPER_TABLE3_MULTITASK_F1 if k in keep and k != args.hide_task]
    if args.pooled_propaganda_loss:
        pooled_class_weights = compute_full_train_pooled_class_weights(
            data_dir=args.data_dir,
            task_names=propaganda_task_names,
            label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
            zero_fill=bool(weight_fill),
        ) if propaganda_task_names else {0: 1.0, 1: 1.0}
        if args.unweighted_loss:
            pooled_class_weights = {0: 1.0, 1: 1.0}

        def trainer_factory(model, cfg):
            kw = dict(pooled_tasks=propaganda_task_names,
                      pooled_class_weights=pooled_class_weights,
                      task_loss_options=task_loss_options,
                      pool_strategy=args.propaganda_pool,
                      technique_loss_weight=args.technique_loss_weight,
                      technique_class_weights=[class_weights[all_task_names.index(t)]
                                               for t in propaganda_task_names],
                      finetune=finetune)
            # Propaganda rows carry rule-made IAC labels, so they are kept
            # out of the IAC heads' confident targets.
            skip = {t: propaganda_task_names for t in discourse_task_names}
            if args.self_distill and label_aware:
                return SelfDistillFamoTrainer(
                    model, cfg, mlm_lambda=args.mlm_lambda, mask_prob=args.mlm_mask_prob,
                    balancer=args.loss_balancer, mlm_mode=args.mlm_mode,
                    masked_label_weight=args.masked_label_weight,
                    consistency_weight=args.consistency_weight,
                    consistency_rampup_epochs=args.consistency_rampup_epochs,
                    consistency_keep=args.consistency_keep,
                    consistency_floor=args.consistency_floor,
                    consistency_skip=skip, distill_outputs=scored_outputs, **kw)
            if args.self_distill:
                return SelfDistillTrainer(model, cfg, distill_outputs=scored_outputs, **kw)
            if ssl_pairs is not None:
                return SemiSupervisedTrainer(
                    model, cfg, ssl_pairs=ssl_pairs,
                    ssl_home_tasks={"Propaganda": propaganda_task_names, "IBM": wa_task_names},
                    ssl_rest_source="IAC", rate_weight=args.ssl_rate_weight,
                    entropy_weight=args.ssl_entropy_weight,
                    entropy_start_epochs=args.ssl_entropy_start_epoch, **kw)
            if teacher_mode:
                return TeacherConsistencyTrainer(
                    model, cfg, teacher=args.consistency_teacher,
                    student_dropout=args.consistency_student_dropout == "on",
                    consistency_weight=args.consistency_weight,
                    consistency_rampup_epochs=args.consistency_rampup_epochs,
                    consistency_keep=args.consistency_keep,
                    consistency_floor=args.consistency_floor,
                    consistency_skip=skip, ema_decay=args.ema_decay, **kw)
            if label_aware:
                return LabelAwareMaskedTrainer(
                    model, cfg, mlm_lambda=args.mlm_lambda, mask_prob=args.mlm_mask_prob,
                    balancer=args.loss_balancer, mlm_mode=args.mlm_mode,
                    masked_label_weight=args.masked_label_weight,
                    consistency_weight=args.consistency_weight,
                    consistency_rampup_epochs=args.consistency_rampup_epochs,
                    consistency_keep=args.consistency_keep,
                    consistency_floor=args.consistency_floor,
                    consistency_skip=skip, **kw)
            if args.mlm_lambda is not None and (args.loss_balancer or args.mlm_mode == "cosine"):
                return BalancedMaskedLMTrainer(model, cfg, mlm_lambda=args.mlm_lambda,
                                               mask_prob=args.mlm_mask_prob,
                                               balancer=args.loss_balancer,
                                               mlm_mode=args.mlm_mode, **kw)
            if args.loss_balancer:
                return BalancedLossTrainer(model, cfg, balancer=args.loss_balancer, **kw)
            if args.mlm_lambda is not None:
                return MaskedLMTrainer(model, cfg, mlm_lambda=args.mlm_lambda,
                                       mask_prob=args.mlm_mask_prob, **kw)
            return PooledOutputLossTrainer(model, cfg, **kw)
    elif args.loss_balancer or args.mlm_lambda is not None:
        raise SystemExit("--loss-balancer / --mlm-lambda need --pooled-propaganda-loss")
    if args.mlm_mode != "fixed" and args.mlm_lambda is None:
        raise SystemExit("--mlm-mode needs --mlm-lambda")
    if args.mlm_mode == "task" and not args.loss_balancer:
        raise SystemExit("--mlm-mode task needs --loss-balancer")
    if args.mlm_mode == "cosine" and args.loss_balancer:
        raise SystemExit("--mlm-mode cosine can't be combined with --loss-balancer")
    if args.mlm_lambda is not None and encoder_config.encoder_integration != EncoderIntegration.HUGGINGFACE:
        raise SystemExit("--mlm-lambda needs a HuggingFace BERT encoder")
    callbacks = []
    scorer = None
    if args.early_stop_metric:
        if not args.pooled_propaganda_loss:
            raise SystemExit("--early-stop-metric needs --pooled-propaganda-loss")
        # The scorer does the stopping, so the library's val-loss rule is off.
        training_config = replace(training_config, early_stopping=False)
        scorer = EpochScorer(val_ds, test_ds, monitor=args.early_stop_metric,
                             patience=args.early_stop_patience,
                             main_output=args.main_task or None,
                             stops=not args.self_distill,
                             evals_per_epoch=args.evals_per_epoch,
                             outputs=scored_outputs
                             if args.hide_task or args.self_distill else None)
        callbacks.append(scorer)
        if args.self_distill:
            callbacks.append(SelfDistillCallback(
                scorer, train_ds, patience=args.self_distill_patience,
                pooled_tasks=propaganda_task_names, pool_strategy=args.propaganda_pool,
                signal=args.self_distill_signal))
        build_trainer = trainer_factory

        def trainer_factory(model, cfg):
            scorer.trainer = build_trainer(model, cfg)
            return scorer.trainer
    # With --hide-task, stage 2 trains and picks epochs for the scored outputs
    # only, so the hidden output's block is never trained and its val labels
    # are never read (run_stage2 records `untrained_unchanged`).
    if args.stage2 and (not args.early_stop_metric or args.self_distill):
        raise SystemExit("--stage2 needs --early-stop-metric (stage 1 = its best epoch) and "
                         "can't be combined with --self-distill")
    if args.main_task and args.early_stop_metric not in ("f1", "auc"):
        raise SystemExit("--main-task needs --early-stop-metric f1 or auc")
    if args.mask_sweep and not args.early_stop_metric:
        raise SystemExit("--mask-sweep needs --early-stop-metric (it reuses the epoch scorer)")
    if args.grad_probe_every:
        if not args.pooled_propaganda_loss:
            raise SystemExit("--grad-probe-every needs --pooled-propaganda-loss")
        callbacks.append(GradientProbe(train_ds, every=args.grad_probe_every,
                                       lookahead_lr=training_config.learning_rate))
    model_cls = MultiTaskModel
    if args.self_distill:
        model_cls = SelfDistillFamoModel if label_aware else SelfDistillModel
    elif ssl_pairs is not None:
        model_cls = SemiSupervisedModel
    elif teacher_mode:
        model_cls = TeacherConsistencyModel
    elif label_aware:
        model_cls = LabelAwareMaskedModel
    elif args.mlm_lambda is not None and (args.loss_balancer or args.mlm_mode == "cosine"):
        model_cls = BalancedMaskedLMModel
    elif args.loss_balancer:
        model_cls = BalancedMultiTaskModel
    elif args.mlm_lambda is not None:
        model_cls = MaskedLMMultiTaskModel
    if args.probe_untrained:
        # Control for the hide-one-label probes: the same probes on the model
        # before any training. 'encoder' is then pretrained BERT's pooled
        # output; 'shared' / 'group' are randomly initialised layers on top.
        import json
        from types import SimpleNamespace
        from hidden_label_probe import run_probes
        if not args.probe_tasks or args.probe_tasks == "hidden":
            raise SystemExit("--probe-untrained needs --probe-tasks all or a comma list")
        model = MultiTaskModel(config, dropout_rate=training_config.dropout_rate)
        names = list(keep) if args.probe_tasks == "all" else args.probe_tasks.split(",")
        probes = _paper_probes(
            SimpleNamespace(model=model), names, propaganda_task_names, val_ds, test_ds,
            data_dir=args.data_dir, max_length=args.max_length, batch_size=args.batch_size,
            news_mask=news_mask, seed=args.seed or 0, run_probes=run_probes)
        if args.results_json:
            os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
            with open(args.results_json, "w") as f:
                json.dump({"args": dict(vars(args)), "untrained": True, "probes": probes},
                          f, indent=2, default=float)
            print(f"Wrote {args.results_json}")
        return
    results = run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        task_type_mapping=task_type_mapping,
        class_weights=class_weights,
        task_weights=uniform_task_weights,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
        training_config=training_config,
        test_ds=test_ds,
        trainer_factory=trainer_factory,
        model_cls=model_cls,
        callbacks=callbacks,
    )
    if args.grad_probe_every:
        probe = next(c for c in callbacks if isinstance(c, GradientProbe))
        results["grad_probe"] = {"tasks": probe.task_names,
                                 "lookahead_lr": probe.lookahead_lr,
                                 "probes": probe.records}
    if scorer is not None:
        results["epoch_scores"] = scorer.records
        results["best_epoch"] = scorer.best_epoch
        results["main_task"] = scorer.main_output
    if args.self_distill:
        results["self_distill"] = {
            "patience": args.self_distill_patience,
            "signal": args.self_distill_signal,
            "switches": next(c for c in callbacks if isinstance(c, SelfDistillCallback)).switches}
    if args.stage2:
        prefix = os.path.splitext(args.results_json)[0] if args.results_json else None
        results["stage2"] = run_stage2(
            scorer.trainer, train_ds, val_ds, test_ds,
            lrs=[lr for lr in args.stage2_lrs.split(",") if lr], epochs=args.stage2_epochs,
            outputs=scored_outputs, batch_size=args.stage2_batch_size or args.batch_size,
            weight_decay=training_config.weight_decay_rate, seed=args.seed,
            predictions_prefix=prefix if args.save_predictions else None,
            weights_path=f"{prefix}_stage1_weights.npz"
            if prefix and args.stage2_save_weights else None)
    if args.mask_sweep:
        results["mask_sweep"] = mask_sweep(
            scorer, [float(r) for r in args.mask_sweep.split(",")],
            repeats=args.mask_sweep_repeats, seed=args.seed or 0)
    if args.probe_tasks:
        if scorer is None:
            raise SystemExit("--probe-tasks needs --early-stop-metric")
        from hidden_label_probe import run_probes
        probe_names = (list(keep) if args.probe_tasks == "all" else
                       [args.hide_task] if args.probe_tasks == "hidden" else
                       args.probe_tasks.split(","))
        if args.probe_tasks == "hidden" and not args.hide_task:
            raise SystemExit("--probe-tasks hidden needs --hide-task")
        results["probes"] = _paper_probes(
            scorer.trainer, probe_names, propaganda_task_names, val_ds, test_ds,
            data_dir=args.data_dir, max_length=args.max_length, batch_size=args.batch_size,
            news_mask=news_mask, seed=args.seed or 0, run_probes=run_probes)
    report_paper_comparison(results, args)


def _paper_probes(trainer, names, propaganda_task_names, val_ds, test_ds, *, data_dir,
                  max_length, batch_size, news_mask, seed, run_probes):
    """Linear probes for ``names`` (Table-3 outputs) on the trained model's
    representations. Train rows are a seeded sample of the train CSV with
    their real labels (hidden or not; the IAC rule labels stay removed when
    ``news_mask`` is set), val/test are the full splits as scored."""
    def ds_labels(ds):
        ys = [y for _, y in ds]
        out = {k: np.concatenate([np.asarray(y[k]).reshape(-1) for y in ys]) for k in ys[0]}
        if propaganda_task_names:
            out["propaganda_max_pool"] = _any_technique(
                np.stack([out[t] for t in propaganda_task_names], axis=1))
        return out

    df = pd.read_csv(f"{data_dir}/no_prop_overlap_train.csv")

    def col(c):
        v = pd.to_numeric(df[PROPAGANDA_BINARY_LABEL_MAP.get(c, c)]
                          .replace(["NA", "na", "NaN", ""], np.nan), errors="coerce").to_numpy()
        v = np.where(np.isnan(v), -1.0, v)
        if news_mask and c in news_mask[1]:
            v = np.where((df["dataset"] == news_mask[0]).to_numpy(), -1.0, v)
        return v

    train_labels = {}
    for name in names:
        if name == "propaganda_max_pool":
            train_labels[name] = _any_technique(
                np.stack([col(t) for t in propaganda_task_names], axis=1))
        else:
            train_labels[name] = col(name)
    return run_probes(trainer.model, names, texts=df["text"].fillna("").astype(str).tolist(),
                      train_labels=train_labels, val_ds=val_ds, val_labels=ds_labels(val_ds),
                      test_ds=test_ds, test_labels=ds_labels(test_ds),
                      max_length=max_length, batch_size=batch_size, seed=seed)


def _any_technique(stacked: np.ndarray) -> np.ndarray:
    """Pooled "any technique" label: max over the labelled technique cells,
    -1 when none is labelled (as the pooled loss derives it)."""
    labelled = stacked != -1
    pooled = np.where(labelled, stacked, 0).max(axis=1).astype(np.float32)
    return np.where(labelled.any(axis=1), pooled, -1.0)


# Multi-task column of the paper's Table 3 (weighted F1, small BERT), keyed by
# the model output that reports it.
PAPER_TABLE3_MULTITASK_F1: Dict[str, Tuple[str, float]] = {
    "propaganda_max_pool": ("Propaganda", 61.74),
    "disagree_agree_bi": ("Disagree/Agree", 71.38),
    "emotion_fact_bi": ("Emotion/Fact", 63.93),
    "attacking_respectful_bi": ("Attacking/Respectful", 68.07),
    "nasty_nice_bi": ("Nasty/Nice", 73.69),
    "personal_audience_bi": ("Personal/Audience", 65.69),
    "defeater_undercutter_bi": ("Defeater/Undercutter", 55.65),
    "negotiate_attack_bi": ("Negotiate/Attack", 64.81),
    "questioning_asserting_bi": ("Questioning/Asserting", 63.23),
    "WA_bi": ("Argument Quality", 79.17),
}


def report_paper_comparison(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print the ten paper outputs next to Table 3 and optionally dump JSON."""
    import json

    def ten_outputs(res: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        merged = {**res.get("task_metrics", {}), **res.get("pool_metrics", {})}
        return {name: merged.get(name, {}) for name in PAPER_TABLE3_MULTITASK_F1}

    val, test = ten_outputs(results), ten_outputs(results.get("test", {}))

    def pct(x: Any) -> str:
        return "   n/a" if x is None else f"{100 * x:6.2f}"

    print("\n=== Paper comparison: weighted F1 x100 (paper Table 3, multi-task) ===")
    print(f"{'Task':<24}{'Paper':>7}{'Val':>8}{'Test':>8}{'Test-Paper':>12}{'Test AUC':>10}")
    diffs, test_f1s = [], []
    for name, (label, paper_f1) in PAPER_TABLE3_MULTITASK_F1.items():
        v, t = val[name].get("f1"), test[name].get("f1")
        auc = test[name].get("auc")
        diff = None if t is None else 100 * t - paper_f1
        if diff is not None:
            diffs.append(diff)
            test_f1s.append(100 * t)
        print(
            f"{label:<24}{paper_f1:>7.2f}{pct(v):>8}{pct(t):>8}"
            f"{'n/a' if diff is None else format(diff, '+.2f'):>12}"
            f"{'n/a' if auc is None else format(auc, '.3f'):>10}"
        )
    if diffs:
        paper_mean = float(np.mean([f for _, f in PAPER_TABLE3_MULTITASK_F1.values()]))
        print(
            f"{'Mean over 10 outputs':<24}{paper_mean:>7.2f}{'':>8}"
            f"{float(np.mean(test_f1s)):>8.2f}{float(np.mean(diffs)):>+12.2f}"
        )

    if args.results_json:
        payload = {
            "args": {k: v for k, v in vars(args).items()},
            "paper_table3_multitask_f1": {k: v[1] for k, v in PAPER_TABLE3_MULTITASK_F1.items()},
            "thresholds": {
                k: (None if v is None else np.asarray(v, dtype=float).tolist())
                for k, v in (results.get("thresholds_used") or {}).items()
            },
            "val": {
                "overall": results.get("overall_metrics", {}),
                "task_metrics": results.get("task_metrics", {}),
                "pool_metrics": results.get("pool_metrics", {}),
            },
            "test": {
                "overall": results.get("test", {}).get("overall_metrics", {}),
                "task_metrics": results.get("test", {}).get("task_metrics", {}),
                "pool_metrics": results.get("test", {}).get("pool_metrics", {}),
            },
            "history": results.get("history", {}),
            "grad_probe": results.get("grad_probe"),
            "epoch_scores": results.get("epoch_scores"),
            "best_epoch": results.get("best_epoch"),
            "main_task": results.get("main_task"),
            "self_distill": results.get("self_distill"),
            "stage2": results.get("stage2"),
            "mask_sweep": results.get("mask_sweep"),
            "probes": results.get("probes"),
        }
        os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"Wrote {args.results_json}")

        if args.save_predictions:
            # Row i of each array is row i of the split's CSV (val/test are not
            # shuffled). Labels use -1 where missing, pools included.
            arrays = {}
            for split, res in (("val", results), ("test", results.get("test", {}))):
                for name, prob in (res.get("probabilities_dict") or {}).items():
                    arrays[f"{split}/prob/{name}"] = np.asarray(prob, dtype=np.float32)
                for name, label in (res.get("labels_dict") or {}).items():
                    arrays[f"{split}/label/{name}"] = np.asarray(label, dtype=np.float32)
            pred_path = os.path.splitext(args.results_json)[0] + "_predictions.npz"
            np.savez_compressed(pred_path, **arrays)
            print(f"Wrote {pred_path} ({len(arrays)} arrays)")


# Added for the partial replication: the 9 of the paper's 10 reported tasks that
# are available without the propaganda corpus (8 IAC discourse + argument
# quality). Grouped hierarchy mirrors paper_replication's task-type branching,
# minus the propaganda branch and its max pool.
def run_discourse_wa(args: argparse.Namespace) -> None:
    print("\n=== Experiment: discourse_wa (8 discourse + WA_bi) ===")
    discourse = list(DISCOURSE_BINARY_COLUMNS)
    all_tasks = discourse + ["WA_bi"]

    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, _ = load_splits(
        label_columns=all_tasks,
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    task_type_mapping = {
        **{t: "discourse" for t in discourse},
        "WA_bi": "wa_bi",
    }
    class_weights = compute_full_train_binary_class_weights(
        data_dir=args.data_dir, task_names=all_tasks
    )
    config = ModelConfig(
        task_structure=[[2], [len(discourse), 1]],
        shared_layer_sizes=[64, 32],
        branch_layer_sizes=[[32, 16], [16, 8]],
        task_names=all_tasks,
        task_types=["binary"] * len(all_tasks),
        num_classes_per_task=[2] * len(all_tasks),
        encoder_config=encoder_config,
    )
    run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        task_type_mapping=task_type_mapping,
        class_weights=class_weights,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
        training_config=PAPER_TRAINING_CONFIG,
    )


EXPERIMENTS = {
    "propaganda": run_propaganda,
    "propaganda_cols_binary": run_propaganda_cols_binary,
    "discourse_dims": run_discourse_dims,
    "discourse_binary": run_discourse_binary,
    "discourse_binary_single_task": run_discourse_binary_single_task,
    "propaganda_binary": run_propaganda_binary,
    "all_task_types": run_all_task_types,
    "all_task_types_binary_prop": run_all_task_types_binary_prop,
    "all_task_types_binary_prop_grouped": run_all_task_types_binary_prop_grouped,
    "all_task_types_binary_prop_deep_grouped": run_all_task_types_binary_prop_deep_grouped,
    "all_task_types_pair_by_dimension": run_all_task_types_pair_by_dimension,
    "all_task_types_grouped": run_all_task_types_grouped,
    "all_task_types_deep_grouped": run_all_task_types_deep_grouped,
    "paper_replication": run_paper_replication,
    "discourse_wa": run_discourse_wa
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no_prop_overlap experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENTS) + ["all"],
        help="Which experiment to run (or 'all' to run every experiment in sequence)",
    )
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument(
        "--no-optimize-thresh", action="store_true",
        help="Disable threshold optimization (always off for regression)",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--no-show-structure",
        action="store_false",
        dest="show_structure",
        help="Disable summary_tree() and task path printing",
    )
    parser.set_defaults(show_structure=True)
    parser.add_argument(
        "--encoder", default="distilbert", choices=list(ENCODER_CONFIGS),
        help="Encoder config to use",
    )
    parser.add_argument(
        "--max-length", type=int, default=96,
        help="Tokenizer max sequence length (paper_replication only; paper config uses 512)",
    )
    parser.add_argument(
        "--embedding-cache", default="/cache/openai_embeddings",
        help="paper_replication with --encoder openai-*: directory of the stored embeddings "
             "(filled by `modal run modal_embed.py::embed`)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument(
        "--dropout", type=float, default=None,
        help="Override PAPER_TRAINING_CONFIG dropout (paper_replication only; paper uses 0.4)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="Override PAPER_TRAINING_CONFIG weight decay (paper_replication only; paper uses 0.01)",
    )
    parser.add_argument(
        "--warmup-percent", type=float, default=None,
        help="Override PAPER_TRAINING_CONFIG warmup percent (paper_replication only; paper uses 5)",
    )
    parser.add_argument(
        "--arch", default="paper", choices=list(PAPER_ARCHITECTURES),
        help="Head layer sizes (paper_replication only); 'legacy' matches multitask_basic",
    )
    parser.add_argument(
        "--pooled-propaganda-loss", action="store_true",
        help="paper_replication only: put the loss on the max-pooled propaganda "
             "output (as the paper describes) instead of the 18 technique heads",
    )
    parser.add_argument(
        "--propaganda-pool", choices=PROPAGANDA_POOLS, default="max",
        help="paper_replication only: how the 18 technique heads are pooled into "
             "the propaganda output: max (the paper), noisy_or (1 - prod(1 - p)) or "
             "lse (sigmoid(logsumexp(logits))). Needs --pooled-propaganda-loss "
             "unless max",
    )
    parser.add_argument(
        "--technique-loss-weight", type=float, default=0.0,
        help="paper_replication only: add this weight x the mean per-technique "
             "class-weighted BCE (each head on its own label) to the pooled "
             "propaganda term. Needs --pooled-propaganda-loss",
    )
    parser.add_argument(
        "--unweighted-loss", action="store_true",
        help="paper_replication only: no task weights (every output 1: no "
             "nu_k = 1/|D_k|, no 1/|T_k| split) and no class weights (both classes 1)",
    )
    parser.add_argument(
        "--no-task-type-grouping", action="store_true",
        help="paper_replication only: weight the loss per head instead of per "
             "task-type (propaganda / WA_bi / discourse)",
    )
    parser.add_argument(
        "--logit-adjust", action="store_true",
        help="paper_replication only: train --imbalance-tasks on z + log(n1/n0) "
             "instead of class-weighted BCE (needs --pooled-propaganda-loss)",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=0.0,
        help="paper_replication only: focal loss gamma for --imbalance-tasks "
             "(0 = plain BCE; needs --pooled-propaganda-loss)",
    )
    parser.add_argument(
        "--imbalance-tasks", default="disagree_agree_bi",
        help="Comma-separated tasks that --logit-adjust / --focal-gamma apply to",
    )
    parser.add_argument(
        "--loss-balancer", default="", choices=["", "famo", "dbmtl"],
        help="paper_replication only: weight the 10 task losses with FAMO or "
             "DB-MTL instead of the fixed task-type weights (needs "
             "--pooled-propaganda-loss)",
    )
    parser.add_argument(
        "--tasks", default="",
        help="paper_replication only: comma-separated subset of the ten Table-3 "
             "outputs to train (propaganda_max_pool = the 18 technique heads); "
             "default all ten",
    )
    parser.add_argument(
        "--propaganda-zero-fill", action="store_true",
        help="paper_replication only: train the 18 propaganda heads with label 0 "
             "('no technique') on rows from the other sources instead of masking "
             "them; val/test are still scored on propaganda rows only",
    )
    parser.add_argument(
        "--zero-fill-keep-class-weights", action="store_true",
        help="With --propaganda-zero-fill: compute the propaganda class weights "
             "from the real labels only, ignoring the filled 0s",
    )
    parser.add_argument(
        "--no-rule-labels", action="store_true",
        help="paper_replication only: remove the 8 IAC labels from Propaganda (news) "
             "rows in train, val and test (they are rule-made from the technique tags); "
             "IAC labels are then scored on IAC rows only",
    )
    parser.add_argument(
        "--hide-task", default="",
        help="paper_replication only: hide this Table-3 output's labels on every train "
             "row (propaganda_max_pool = the 18 technique labels); its head is still "
             "scored but left out of epoch selection. Needs --early-stop-metric",
    )
    parser.add_argument(
        "--probe-tasks", default="",
        help="paper_replication only: after training, fit a logistic-regression probe "
             "per output on the frozen model's representations ('all', 'hidden' = the "
             "--hide-task output, or a comma list). Needs --early-stop-metric",
    )
    parser.add_argument(
        "--label-share", default="",
        help="paper_replication only: column=share[,column=share]: keep that share of "
             "the column's train labels (seeded, nested across shares) and mask the rest",
    )
    parser.add_argument(
        "--repeat-train", type=int, default=1,
        help="paper_replication only: repeat the train rows this many times, so a run on "
             "few rows takes as many steps per epoch as a full one",
    )
    parser.add_argument(
        "--probe-untrained", action="store_true",
        help="paper_replication only: skip training and run the --probe-tasks probes on "
             "the untrained model (pretrained BERT + randomly initialised layers)",
    )
    parser.add_argument(
        "--mask-news-discourse-train", action="store_true",
        help="paper_replication only: mask the 8 IAC discourse labels on Propaganda "
             "(news) rows in the train split (they are rule-made from the technique "
             "tags); class weights follow; val/test keep them",
    )
    parser.add_argument(
        "--stratify-batches", action="store_true",
        help="paper_replication only: deal train rows into batches so each batch "
             "keeps the training set's mix of sources, labelled tasks and classes",
    )
    parser.add_argument(
        "--drop-unlabeled-train", action="store_true",
        help="Drop train rows with none of the trained labels (val/test keep every row)",
    )
    parser.add_argument(
        "--grad-probe-every", type=int, default=0,
        help="paper_replication only: every N training steps, log per-task gradient "
             "cosines and norms on the shared parameters (needs --pooled-propaganda-loss)",
    )
    parser.add_argument(
        "--mlm-lambda", type=float, default=None,
        help="paper_replication only: add this weight x a masked-word loss on the "
             "encoder to the supervised loss (0 runs the same code with no effect; "
             "needs --pooled-propaganda-loss)",
    )
    parser.add_argument(
        "--mlm-mask-prob", type=float, default=0.15,
        help="Share of word tokens picked for the masked-word loss",
    )
    parser.add_argument(
        "--mlm-mode", default="fixed", choices=list(MLM_MODES),
        help="How the masked-word loss is weighted: 'fixed' (--mlm-lambda; with "
             "--loss-balancer the balancer weights only the 10 tasks), 'task' (an "
             "11th task under --loss-balancer, which sets its weight), or 'cosine' "
             "(--mlm-lambda, used only on steps where the masked-word gradient on "
             "the shared parameters has positive cosine with the supervised one)",
    )
    parser.add_argument(
        "--masked-label-weight", type=float, default=0.0,
        help="paper_replication only: add this weight x the supervised loss on the "
             "masked copy against the real labels (needs --loss-balancer famo and "
             "--mlm-lambda; 0 = off)",
    )
    parser.add_argument(
        "--consistency-weight", type=float, default=0.0,
        help="paper_replication only: add this weight x KL(full-sentence answer || "
             "masked-copy answer) on every row and the 10 outputs (needs "
             "--loss-balancer famo and --mlm-lambda; 0 = off)",
    )
    parser.add_argument(
        "--consistency-keep", type=float, default=0.0,
        help="With --consistency-weight: use only confident targets, per head the "
             "lowest and highest this share of full-sentence scores in each batch "
             "(e.g. 0.1 = 10%% each side); Propaganda rows are skipped for the IAC "
             "heads. 0 = every cell",
    )
    parser.add_argument(
        "--consistency-floor", type=float, default=0.0,
        help="With --consistency-keep: a kept score must also be <= this or >= 1 - this; "
             "both sides shrink to the smaller count, so a head with no confident cells "
             "on one side contributes none in that batch. 0 = no floor",
    )
    parser.add_argument(
        "--consistency-teacher", default="", choices=["", *CONSISTENCY_TEACHERS],
        help="paper_replication only: consistency without masking, the target from a "
             "teacher run with dropout off: 'current' (the model itself) or 'ema' (a "
             "moving average of its weights). Needs --loss-balancer famo and "
             "--consistency-weight, and no --mlm-lambda",
    )
    parser.add_argument(
        "--consistency-student-dropout", default="on", choices=["on", "off"],
        help="With --consistency-teacher: score the student pass with encoder dropout on "
             "(the supervised pass) or off (a second pass)",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.99,
        help="With --consistency-teacher ema: the moving-average decay (it trails the "
             "student by about 1 / (1 - decay) steps)",
    )
    parser.add_argument(
        "--consistency-rampup-epochs", type=float, default=1.0,
        help="Epochs over which --consistency-weight rises linearly from 0",
    )
    parser.add_argument(
        "--semi-supervised", action="store_true",
        help="paper_replication only: on foreign cells (a head with no label, on rows "
             "from a named source) add --ssl-rate-weight x KL(target rate || mean score "
             "in the batch) and --ssl-entropy-weight x mean entropy of the scores. Built "
             "on the plain pooled loss (needs --pooled-propaganda-loss)",
    )
    parser.add_argument(
        "--ssl-rates", default=SSL_DEFAULT_RATES,
        help="With --semi-supervised: comma-separated head@source=rate, source one of "
             f"{', '.join(SSL_SOURCES)}; head {SSL_POOL} is the pooled propaganda output",
    )
    parser.add_argument(
        "--ssl-rate-weight", type=float, default=0.1,
        help="With --semi-supervised: weight of the rate term (0 = off)",
    )
    parser.add_argument(
        "--ssl-entropy-weight", type=float, default=0.0,
        help="With --semi-supervised: weight of the entropy (confidence) term (0 = off)",
    )
    parser.add_argument(
        "--ssl-entropy-start-epoch", type=float, default=2.0,
        help="With --semi-supervised: the entropy term stays off for this many epochs",
    )
    parser.add_argument(
        "--mask-sweep", default="",
        help="paper_replication only: after training, score val on masked copies at "
             "these comma-separated mask rates (e.g. 0.05,0.1,0.15,0.3) against the "
             "clean answers (needs --early-stop-metric)",
    )
    parser.add_argument(
        "--mask-sweep-repeats", type=int, default=3,
        help="Random maskings per rate for --mask-sweep",
    )
    parser.add_argument(
        "--early-stop-metric", default="", choices=["", *EARLY_STOP_METRICS],
        help="paper_replication only: score val and test after every epoch, keep "
             "the epoch with the best val loss / mean F1 / mean AUC, and stop once "
             "all three have stalled for --early-stop-patience epochs (default: "
             "the paper's val-loss rule with patience 2)",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=3,
        help="Patience for --early-stop-metric",
    )
    parser.add_argument(
        "--evals-per-epoch", type=int, default=1,
        help="paper_replication only, with --early-stop-metric: score val and test "
             "this many times per epoch, at equal step intervals, and pick the best "
             "scoring instead of the best epoch; --early-stop-patience then counts "
             "scorings",
    )
    parser.add_argument(
        "--length-chunk-batches", type=int, default=0,
        help="paper_replication only: pad each batch to its own longest text instead "
             "of to --max-length. Train batches group texts of similar length within "
             "chunks of this many batches, keeping each batch's mix of sources; val "
             "and test keep the CSV order. 0 = off (pad everything to --max-length)",
    )
    parser.add_argument(
        "--layer-lr-decay", type=float, default=1.0,
        help="paper_replication only: ELECTRA-style layer-wise learning-rate decay. "
             "The task layers train at --learning-rate, each encoder layer below at "
             "this factor times the rate of the one above (top layer: factor^2), the "
             "embeddings lowest. 1 = one rate for everything",
    )
    parser.add_argument(
        "--clipnorm", type=float, default=0.0,
        help="paper_replication only: clip the global gradient norm to this value "
             "(0 = no clipping)",
    )
    parser.add_argument(
        "--no-decay-bias-norm", action="store_true",
        help="paper_replication only: no weight decay on biases and LayerNorm weights",
    )
    parser.add_argument(
        "--self-distill", action="store_true",
        help="Per-task stopping with self-distillation on the plain pooled loss: once "
             "an output's val F1 has stalled --self-distill-patience epochs, its loss "
             "switches from its labels to the scores of its own best epoch; training "
             "ends when every output has switched. Needs --early-stop-metric f1",
    )
    parser.add_argument(
        "--self-distill-patience", type=int, default=2,
        help="Epochs without a val gain before an output switches (--self-distill)",
    )
    parser.add_argument(
        "--self-distill-signal", default="f1", choices=list(SELF_DISTILL_SIGNALS),
        help="Per-output val signal for --self-distill switching: F1 (default) or the "
             "output's own class-weighted val loss",
    )
    parser.add_argument(
        "--stage2", action="store_true",
        help="paper_replication only: after training, freeze the encoder, shared layers and "
             "task-type blocks, cache their outputs and re-train each task's own block on "
             "the real labels, keeping each output's own val-F1-best epoch (stage 1's block "
             "= epoch 0). Needs --early-stop-metric",
    )
    parser.add_argument(
        "--stage2-lrs", default="1e-4,3e-5",
        help="Comma-separated stage-2 learning rates; each starts from the same stage 1",
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=10, help="Stage-2 epochs per learning rate",
    )
    parser.add_argument(
        "--stage2-batch-size", type=int, default=0,
        help="Stage-2 batch size (default 0: the same as --batch-size)",
    )
    parser.add_argument(
        "--stage2-save-weights", action="store_true",
        help="Also write the stage-1 weights next to --results-json",
    )
    parser.add_argument(
        "--main-task", default="", choices=["", *PAPER_TABLE3_MULTITASK_F1],
        help="Caruana-style main task: pick the epoch and stop on this one output's "
             "val F1/AUC instead of the mean over outputs. Every task still trains "
             "with the same loss. Needs --early-stop-metric f1 or auc",
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Also write val/test probabilities and labels next to --results-json "
             "as <name>_predictions.npz",
    )
    parser.add_argument(
        "--results-json", default=None,
        help="Write paper_replication val/test metrics to this JSON path",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.learning_rate_explicit = any(
        a == "--learning-rate" or a.startswith("--learning-rate=") for a in sys.argv[1:]
    )
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)

    if not os.path.exists(f"{args.data_dir}/no_prop_overlap_train.csv"):
        print(
            f"ERROR: data not found at {args.data_dir}/no_prop_overlap_train.csv\n"
            "Make sure --data-dir points to the directory containing the no_prop_overlap_*.csv files.",
            file=sys.stderr,
        )
        sys.exit(1)

    to_run = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in to_run:
        EXPERIMENTS[name](args)


if __name__ == "__main__":
    main()
