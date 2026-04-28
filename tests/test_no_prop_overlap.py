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
    discourse_dims      Discourse dimensions only (regression)
    discourse_binary    Binary discourse dimensions
    propaganda_binary   Propaganda techniques + binary discourse dimensions
    all_task_types      All task types combined
    all_task_types_pair_by_dimension
                        Pair-by-dimension hierarchy (regression+binary sibling pairs)
    all_task_types_grouped
                        All task types with grouped hierarchy by task family
    all_task_types_deep_grouped
                        All task types with deep hierarchy (all -> some -> tasks)
    output_pooling      Output pooling on propaganda (not yet implemented)
    all                 Run all experiments in sequence

Options:
    --data-dir DIR          Path to data directory (default: data)
    --epochs N              Training epochs (default: 10)
    --batch-size N          Batch size (default: 128)
    --learning-rate LR      Learning rate (default: 4e-5)
    --no-optimize-thresh    Disable threshold optimization
    --verbose N             Keras verbosity 0/1/2 (default: 1)
    --encoder {distilbert,tfhub-bert}
                            Encoder config to use (default: distilbert)
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

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

ENCODER_CONFIGS = {
    "distilbert": ecfg_distilbert_uncased,
    "tfhub-bert": ecfg_tfhub_bert,
}


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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

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

        texts = ["" if pd.isna(x) else str(x) for x in df[text_column]]
        enc = tokenizer(
            texts,
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
    show_structure: bool = False,
) -> Dict[str, Any]:
    model = MultiTaskModel(config)
    if show_structure:
        print("\nModel structure:")
        print(model.summary_tree())
        print("Task paths:")
        for info in model.get_task_info():
            print(
                f"  {info['output_idx']:>2}  {info['name']:<35} "
                f"path={info['parent_branch_path']}"
            )

    trainer = Trainer(
        model,
        TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
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

def run_propaganda(args: argparse.Namespace) -> None:
    print("\n=== Experiment: propaganda ===")
    cols = list(PROPAGANDA_TECHNIQUE_COLUMNS)
    train_ds, val_ds, _ = load_splits(
        multilabel_groups={"propaganda_techniques": cols},
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    config = ModelConfig(
        task_structure=[[1]],
        task_names=["propaganda_techniques"],
        task_types=["multilabel"],
        num_classes_per_task=[len(cols)],
        encoder_config=ENCODER_CONFIGS[args.encoder],
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
    train_ds, val_ds, _ = load_splits(cols, batch_size=args.batch_size, data_dir=args.data_dir)
    config = ModelConfig(
        task_structure=[[len(cols)]],
        task_names=cols,
        task_types=["regression"] * len(cols),
        num_classes_per_task=[1] * len(cols),
        encoder_config=ENCODER_CONFIGS[args.encoder],
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
    train_ds, val_ds, _ = load_splits(cols, batch_size=args.batch_size, data_dir=args.data_dir)
    config = ModelConfig(
        task_structure=[[len(cols)]],
        task_names=cols,
        task_types=["binary"] * len(cols),
        num_classes_per_task=[2] * len(cols),
        encoder_config=ENCODER_CONFIGS[args.encoder],
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


def run_propaganda_binary(args: argparse.Namespace) -> None:
    print("\n=== Experiment: propaganda_binary ===")
    binary_cols = list(DISCOURSE_BINARY_COLUMNS)
    train_ds, val_ds, _ = load_splits(
        label_columns=binary_cols,
        multilabel_groups={"propaganda_techniques": list(PROPAGANDA_TECHNIQUE_COLUMNS)},
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
        encoder_config=ENCODER_CONFIGS[args.encoder],
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
    train_ds, val_ds, _ = load_splits(
        label_columns=list(DISCOURSE_REGRESSION_COLUMNS) + list(DISCOURSE_BINARY_COLUMNS),
        multilabel_groups={"propaganda_techniques": list(PROPAGANDA_TECHNIQUE_COLUMNS)},
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
        encoder_config=ENCODER_CONFIGS[args.encoder],
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


def run_all_task_types_pair_by_dimension(args: argparse.Namespace) -> None:
    print("\n=== Experiment: all_task_types_pair_by_dimension ===")
    paired_names = []
    paired_types = []
    paired_num_classes = []
    for reg_col, bi_col in zip(DISCOURSE_REGRESSION_COLUMNS, DISCOURSE_BINARY_COLUMNS):
        paired_names.extend([reg_col, bi_col])
        paired_types.extend(["regression", "binary"])
        paired_num_classes.extend([1, 2])

    train_ds, val_ds, _ = load_splits(
        label_columns=list(DISCOURSE_REGRESSION_COLUMNS) + list(DISCOURSE_BINARY_COLUMNS),
        multilabel_groups={"propaganda_techniques": list(PROPAGANDA_TECHNIQUE_COLUMNS)},
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    config = ModelConfig(
        task_structure=[[1 + len(DISCOURSE_REGRESSION_COLUMNS)], [1] + [2] * len(DISCOURSE_REGRESSION_COLUMNS)],
        task_names=["propaganda_techniques"] + paired_names,
        task_types=["multilabel"] + paired_types,
        num_classes_per_task=[len(PROPAGANDA_TECHNIQUE_COLUMNS)] + paired_num_classes,
        encoder_config=ENCODER_CONFIGS[args.encoder],
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
        show_structure=args.show_structure,
    )


def run_output_pooling(args: argparse.Namespace) -> None:
    print("\n=== Experiment: output_pooling ===")
    print("Not yet implemented.")


EXPERIMENTS = {
    "propaganda": run_propaganda,
    "discourse_dims": run_discourse_dims,
    "discourse_binary": run_discourse_binary,
    "propaganda_binary": run_propaganda_binary,
    "all_task_types": run_all_task_types,
    "all_task_types_pair_by_dimension": run_all_task_types_pair_by_dimension,
    "all_task_types_grouped": run_all_task_types_grouped,
    "all_task_types_deep_grouped": run_all_task_types_deep_grouped,
    "output_pooling": run_output_pooling,
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
        "--show-structure",
        action="store_true",
        help="Print summary_tree() and task paths before training",
    )
    parser.add_argument(
        "--encoder", default="distilbert", choices=list(ENCODER_CONFIGS),
        help="Encoder config to use",
    )
    return parser


def main() -> None:
    import os
    parser = build_parser()
    args = parser.parse_args()

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
