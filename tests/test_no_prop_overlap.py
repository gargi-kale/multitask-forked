"""
End-to-end real-world tests on the ``data/no_prop_overlap_*.csv`` datasets.

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

For all experiments: use a lightweight HuggingFace model (e.g.
``distilbert-base-uncased``). Start with a small hierarchy, train 10–20 epochs
with early stopping, and report metrics from ``trainer.evaluate()``. Intended
to run on Midway.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from transformers import AutoTokenizer

# Mixed precision gives ~1.5-2x speedup on modern GPUs (A100/V100).
# Weights/optimizer state stay fp32; activations compute in fp16. Safe for BERT.
# No-op on CPU-only machines.
tf.keras.mixed_precision.set_global_policy("mixed_float16")

from multitask.config import (
    ModelConfig, EncoderConfig, TrainingConfig,
    EncoderIntegration, EncoderInputType
)

from multitask import MultiTaskModel, Trainer

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

ALL_COLUMNS = (
    list(PROPAGANDA_TECHNIQUE_COLUMNS) 
    + list(DISCOURSE_REGRESSION_COLUMNS) 
    + list(DISCOURSE_BINARY_COLUMNS)
)


# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------

def load_splits(
    label_columns: Sequence[str] = (),
    *,
    multilabel_groups: Optional[Dict[str, Sequence[str]]] = None,
    text_column: str = "text",
    batch_size: int = 32,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 96,
    data_dir: str = "data",
    prefix: str = "no_prop_overlap_",
    na_strings: Tuple[str, ...] = ("NA", "na", "NaN", ""),
    mask_value: float = -1.0,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    multilabel_groups = multilabel_groups or {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def _to_float(s: pd.Series) -> np.ndarray:
        s = s.replace(list(na_strings), np.nan)
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
        return np.where(np.isnan(vals), mask_value, vals)
    
    def build(split: str) -> tf.data.Dataset:
        df = pd.read_csv(f"{data_dir}/{prefix}{split}.csv")
        labels: Dict[str, np.ndarray] = {}
        
        for col in label_columns:
            labels[col] = _to_float(df[col])
        
        for name, cols in multilabel_groups.items():
            labels[name] = np.stack([_to_float(df[c]) for c in cols], axis=1)
        
        enc = tokenizer(
            df[text_column].astype(str).tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int32),
            "attention_mask": enc["attention_mask"].astype(np.int32),
        }
        
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if split == "train":
            ds = ds.shuffle(min(len(df), 1000))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return build("train"), build("val"), build("test")


# ---------------------------------------------------------------------------
# Skip real-data tests when the CSVs aren't present
# ---------------------------------------------------------------------------

import os

_DATA_PRESENT = os.path.exists("data/no_prop_overlap_train.csv")
requires_data = pytest.mark.skipif(
    not _DATA_PRESENT,
    reason="no_prop_overlap CSVs not available under data/",
)

# ---------------------------------------------------------------------------
# Configs and common vars
# ---------------------------------------------------------------------------

BATCH_SIZE = 128

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
    trainable_encoder=True
)


# ---------------------------------------------------------------------------
# Shared train/eval harness
# ---------------------------------------------------------------------------


def run_experiment(
    config: ModelConfig,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs: int = 10,
    # Canonical BERT fine-tuning LR with sqrt scaling for BATCH_SIZE=128:
    # 3e-5 at bs=64 -> ~4e-5 at bs=128. 1e-3 diverges when trainable_encoder=True.
    learning_rate: float = 4e-5,
    optimize_thresholds: bool = True,
    verbose: int = 1,
) -> Dict[str, Any]:
    """Build ``MultiTaskModel`` + ``Trainer`` from ``config``, fit on
    ``train_ds`` / ``val_ds``, then evaluate. Returns the ``evaluate()`` dict.
    """
    model = MultiTaskModel(config)
    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=epochs,
            batch_size=BATCH_SIZE,
            learning_rate=learning_rate,
        ),
    )

    _, thresholds = trainer.fit(
        train_ds,
        val_ds,
        num_tasks=len(config.task_names),
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        optimize_thresholds=optimize_thresholds,
        verbose=verbose,
    )
    print("Thresholds:", thresholds)

    results = trainer.evaluate(val_ds, thresholds=thresholds)
    print("Overall metrics:", results.get("overall_metrics", {}))
    return results


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@requires_data
def test_propaganda():
    """
    Propaganda techniques only (multilabel, single 18-class head).
    """
    cols = list(PROPAGANDA_TECHNIQUE_COLUMNS)
    train_ds, val_ds, _ = load_splits(
        multilabel_groups={"propaganda_techniques": cols},
        batch_size=BATCH_SIZE,
    )

    config = ModelConfig(
        task_structure=[[1]],
        task_names=["propaganda_techniques"],
        task_types=["multilabel"],
        num_classes_per_task=[len(cols)],
        encoder_config=ecfg_distilbert_uncased,
    )

    results = run_experiment(config, train_ds, val_ds)
    assert results.get("overall_metrics"), "no metrics returned"


@requires_data
def test_discourse_dims():
    """
    Discourse dimensions only (regression). One regression task per continuous
    discourse column; ``NA`` rows masked (``mask_value=-1.0``); ``mse``,
    ``mae``, ``r2`` returned per dimension.
    """
    cols = list(DISCOURSE_REGRESSION_COLUMNS)
    train_ds, val_ds, _ = load_splits(cols, batch_size=BATCH_SIZE)

    config = ModelConfig(
        task_structure=[[len(cols)]],
        task_names=cols,
        task_types=["regression"] * len(cols),
        num_classes_per_task=[1] * len(cols),
        encoder_config=ecfg_distilbert_uncased,
    )

    # Pure regression -> no thresholds to optimize.
    results = run_experiment(config, train_ds, val_ds, optimize_thresholds=False)
    assert results.get("overall_metrics"), "no metrics returned"


@requires_data
def test_discourse_binary():
    """
    Binary discourse only. ``*_bi`` columns as binary tasks; accuracy and F1
    per dimension.
    """
    cols = list(DISCOURSE_BINARY_COLUMNS)
    train_ds, val_ds, _ = load_splits(cols, batch_size=BATCH_SIZE)

    config = ModelConfig(
        task_structure=[[len(cols)]],
        task_names=cols,
        task_types=["binary"] * len(cols),
        num_classes_per_task=[2] * len(cols),
        encoder_config=ecfg_distilbert_uncased,
    )

    results = run_experiment(config, train_ds, val_ds)
    assert results.get("overall_metrics"), "no metrics returned"


@requires_data
def test_all_task_types():
    """
    All task types together (multilabel propaganda + regression discourse +
    binary discourse + binary ``propaganda``). Check per-task metrics and
    ``results['overall_metrics']`` is populated and finite.
    """
    train_ds, val_ds, _ = load_splits(
        label_columns=list(DISCOURSE_REGRESSION_COLUMNS) + list(DISCOURSE_BINARY_COLUMNS),
        multilabel_groups={"propaganda_techniques": list(PROPAGANDA_TECHNIQUE_COLUMNS)},
        batch_size=BATCH_SIZE,
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
        task_structure=[[len(task_names)]],
        task_names=task_names,
        task_types=task_types,
        num_classes_per_task=num_classes_per_task,
        encoder_config=ecfg_distilbert_uncased,
    )

    results = run_experiment(config, train_ds, val_ds)
    assert results.get("overall_metrics"), "no metrics returned"


@requires_data
def test_output_pooling():
    """
    Output pooling on propaganda: ``output_pools={'any_propaganda': {'tasks':
    [...18...], 'strategy': 'max'}}``. Compare pool F1 against the standalone
    ``propaganda`` binary column.
    """
    pass
