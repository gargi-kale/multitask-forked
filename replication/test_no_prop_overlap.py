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
    --encoder {distilbert,tfhub-bert,small-bert}
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

ENCODER_CONFIGS = {
    "distilbert": ecfg_distilbert_uncased,
    "tfhub-bert": ecfg_tfhub_bert,
    "small-bert": ecfg_small_bert,
}

# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------

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
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    multilabel_groups = multilabel_groups or {}
    label_column_map = label_column_map or {}
    encoder_config = encoder_config or ENCODER_CONFIGS["distilbert"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

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

        for name, cols in multilabel_groups.items():
            labels[name] = np.stack([_to_float(df[c]) for c in cols], axis=1)

        texts = ["" if pd.isna(x) else str(x) for x in df[text_column]]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        if encoder_config.encoder_input_type == EncoderInputType.TFHUB_TOKENS:
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

        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if split == "train":
            # Full-length buffer: the train CSV is sorted by source corpus, so a
            # short buffer yields single-corpus batches for a whole third of an epoch.
            ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return build("train"), build("val"), build("test")


def compute_full_train_binary_class_weights(
    *,
    data_dir: str,
    task_names: Sequence[str],
    label_column_map: Optional[Dict[str, str]] = None,
    prefix: str = "no_prop_overlap_",
    mask_value: float = -1.0,
    na_strings: Tuple[str, ...] = ("NA", "na", "NaN", ""),
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
) -> Dict[int, float]:
    """Binary class weights for the max over ``task_names`` (e.g. the pooled
    propaganda label), same rule as ``compute_full_train_binary_class_weights``.
    A row is masked only when every constituent column is missing."""
    label_column_map = label_column_map or {}
    df = pd.read_csv(f"{data_dir}/{prefix}train.csv")
    cols = [label_column_map.get(t, t) for t in task_names]
    vals = df[cols].replace(list(na_strings), np.nan).apply(pd.to_numeric, errors="coerce")
    vals = vals.where(vals != mask_value)
    pooled = vals.max(axis=1, skipna=True).dropna().to_numpy().astype(int)
    counts = np.bincount(pooled, minlength=2)
    return {c: (pooled.size / (2.0 * counts[c]) if counts[c] else 0.001) for c in (0, 1)}


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
    """

    def __init__(self, *args, pooled_tasks: Sequence[str],
                 pooled_class_weights: Dict[int, float], **kwargs):
        super().__init__(*args, **kwargs)
        self.pooled_tasks = list(pooled_tasks)
        self.pooled_class_weights = pooled_class_weights

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
             float(task_weights[i]) / type_sizes[types[n]])
            for i, n in enumerate(names) if n not in pooled
        ]
        pool_factor = float(task_weights[names.index(self.pooled_tasks[0])])
        pw0 = float(self.pooled_class_weights[0])
        pw1 = float(self.pooled_class_weights[1])
        print(f"Pooled loss over {len(self.pooled_tasks)} heads: "
              f"class weights {{0: {pw0:.4f}, 1: {pw1:.4f}}}, nu_k={pool_factor:.6f}, |T_k|=1")
        for n, w0, w1, f in terms:
            print(f"  {n}: class weights {{0: {w0:.4f}, 1: {w1:.4f}}}, nu_k/|T_k|={f:.6f}")

        def term(y_true, y_pred, w0, w1, factor):
            y = tf.reshape(tf.cast(y_true, tf.float32), [-1])
            p = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
            mask = tf.cast(tf.not_equal(y, mask_value), tf.float32)
            y = tf.where(mask > 0, y, tf.zeros_like(y))
            bce = tf.reshape(tf.keras.losses.binary_crossentropy(
                tf.reshape(y, [-1, 1]), tf.reshape(p, [-1, 1])), [-1])
            w = tf.where(tf.equal(y, 0.0), w0, w1)
            return tf.reduce_sum(bce * w * mask * factor), tf.reduce_sum(mask)

        def pooled_loss(y_true_dict, y_pred_dict):
            total, n_valid = 0.0, 0.0
            for n, w0, w1, f in terms:
                s, c = term(y_true_dict[n], y_pred_dict[n], w0, w1, f)
                total += s
                n_valid += c
            # mask_value (-1) is below both classes, so the max is masked only
            # when every constituent label is masked.
            y_pool = tf.reduce_max(tf.stack(
                [tf.reshape(tf.cast(y_true_dict[t], tf.float32), [-1]) for t in self.pooled_tasks],
                axis=1), axis=1)
            p_pool = tf.reduce_max(tf.stack(
                [tf.reshape(tf.cast(y_pred_dict[t], tf.float32), [-1]) for t in self.pooled_tasks],
                axis=1), axis=1)
            s, c = term(y_pool, p_pool, pw0, pw1, pool_factor)
            return (total + s) / (n_valid + c + 1e-7)

        return pooled_loss


# ---------------------------------------------------------------------------
# Shared train/eval harness
# ---------------------------------------------------------------------------

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
    test_ds: Optional[tf.data.Dataset] = None,
    trainer_factory: Optional[Any] = None,
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

    model = MultiTaskModel(config, dropout_rate=resolved_training_config.dropout_rate)
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
        task_type_mapping=task_type_mapping,
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        optimize_thresholds=optimize_thresholds,
        verbose=verbose,
    )
    print("Thresholds:", thresholds)

    results = trainer.evaluate(val_ds, thresholds=thresholds)
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
}


def run_paper_replication(args: argparse.Namespace) -> None:
    print("\n=== Experiment: paper_replication ===")
    propaganda_task_names = list(PROPAGANDA_BINARY_TASK_NAMES)
    discourse_task_names = list(DISCOURSE_BINARY_COLUMNS)
    all_task_names = propaganda_task_names + ["WA_bi"] + discourse_task_names

    encoder_config = ENCODER_CONFIGS[args.encoder]
    train_ds, val_ds, test_ds = load_splits(
        label_columns=all_task_names,
        encoder_config=encoder_config,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
        max_length=args.max_length,
        seed=args.seed,
    )
    task_types = ["binary"] * len(all_task_names)
    num_classes_per_task = [2] * len(all_task_names)

    task_type_mapping = {
        **{task_name: "propaganda_binary" for task_name in propaganda_task_names},
        "WA_bi": "wa_bi",
        **{task_name: "discourse" for task_name in discourse_task_names},
    }
    class_weights = compute_full_train_binary_class_weights(
        data_dir=args.data_dir,
        task_names=all_task_names,
        label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
    )
    
    shared_layer_sizes, branch_layer_sizes = PAPER_ARCHITECTURES[args.arch]
    config = ModelConfig(
        task_structure=[
            [3],
            [len(propaganda_task_names), 1, len(discourse_task_names)],
        ],
        shared_layer_sizes=shared_layer_sizes,
        branch_layer_sizes=branch_layer_sizes,
        task_names=all_task_names,
        task_types=task_types,
        num_classes_per_task=num_classes_per_task,
        output_pools={
            "propaganda_max_pool": {
                "tasks": propaganda_task_names,
                "strategy": "max",
                "task_type": "binary",
                "num_classes": 2,
            },
        },
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
    trainer_factory = None
    if args.pooled_propaganda_loss:
        pooled_class_weights = compute_full_train_pooled_class_weights(
            data_dir=args.data_dir,
            task_names=propaganda_task_names,
            label_column_map=PROPAGANDA_BINARY_LABEL_MAP,
        )

        def trainer_factory(model, cfg):
            return PooledOutputLossTrainer(
                model, cfg,
                pooled_tasks=propaganda_task_names,
                pooled_class_weights=pooled_class_weights,
            )
    results = run_experiment(
        config, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        task_type_mapping=task_type_mapping,
        class_weights=class_weights,
        optimize_thresholds=not args.no_optimize_thresh,
        verbose=args.verbose,
        show_structure=args.show_structure,
        training_config=training_config,
        test_ds=test_ds,
        trainer_factory=trainer_factory,
    )
    report_paper_comparison(results, args)


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
        }
        os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        print(f"Wrote {args.results_json}")


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
