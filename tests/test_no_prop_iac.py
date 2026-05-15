"""
IAC-only (discourse binary) grid search on the no_prop_overlap dataset.

This mirrors the paper-replication training setup from tests/test_no_prop_overlap.py
for model family + trainer wiring, but restricts tasks to the 8 discourse binary
dimensions and runs a 32-combination hyperparameter grid.

Grid dimensions:
    - optimizer: adamw (fixed)
    - learning_rate: {0.0001, 0.0003}
    - weight_decay_rate: {0.005, 0.01}
    - warmup_fraction: {0.05, 0.1}
    - batch_size: {128, 256}
    - dropout_rate: {0.2, 0.4}
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
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


DISCOURSE_REGRESSION_COLUMNS: Tuple[str, ...] = (
    "disagree_agree", "attacking_respectful", "emotion_fact", "nasty_nice",
    "personal_audience", "defeater_undercutter", "negotiate_attack",
    "questioning_asserting",
)

DISCOURSE_BINARY_COLUMNS: Tuple[str, ...] = tuple(
    f"{c}_bi" for c in DISCOURSE_REGRESSION_COLUMNS
)

ecfg_small_bert = EncoderConfig(
    encoder_identifier="google/bert_uncased_L-2_H-128_A-2",
    encoder_input_type=EncoderInputType.HUGGINGFACE_TOKENS,
    encoder_integration=EncoderIntegration.HUGGINGFACE,
    trainable_encoder=True,
    from_pt=True,
)


def load_splits(
    label_columns: Sequence[str] = (),
    *,
    multilabel_groups: Optional[Dict[str, Sequence[str]]] = None,
    text_column: str = "text",
    batch_size: int = 32,
    tokenizer_name: str = "google/bert_uncased_L-2_H-128_A-2",
    max_length: int = 96,
    data_dir: str = "data",
    prefix: str = "no_prop_overlap_",
    na_strings: Tuple[str, ...] = ("NA", "na", "NaN", ""),
    mask_value: float = -1.0,
    label_column_map: Optional[Dict[str, str]] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    multilabel_groups = multilabel_groups or {}
    label_column_map = label_column_map or {}
    encoder_config = ecfg_small_bert
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
            ds = ds.shuffle(min(len(df), 1000))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return build("train"), build("val"), build("test")


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
    trainer: Optional[Trainer] = None,
    task_type_mapping: Optional[Dict[str, str]] = None,
    training_config: Optional[TrainingConfig] = None,
    task_weights: Optional[List[float]] = None,
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
    print("\nModel structure:")
    print(model.summary_tree())
    print("Task paths:")
    for info in model.get_task_info():
        print(
            f"  {info['output_idx']:>2}  {info['name']:<35} "
            f"path={info['parent_branch_path']}"
        )

    if trainer is None:
        trainer = Trainer(model, resolved_training_config)

    _, thresholds = trainer.fit(
        train_ds,
        val_ds,
        num_tasks=len(config.task_names),
        task_weights=task_weights,
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
    return results


def _build_iac_config() -> ModelConfig:
    task_names = list(DISCOURSE_BINARY_COLUMNS)
    return ModelConfig(
        task_structure=[[len(task_names)]],
        shared_layer_sizes=[64, 32],
        branch_layer_sizes=[[32, 16]],
        task_names=task_names,
        task_types=["binary"] * len(task_names),
        num_classes_per_task=[2] * len(task_names),
        encoder_config=ecfg_small_bert,
    )


def _extract_primary_metric(results: Dict[str, Any]) -> float:
    overall = results.get("overall_metrics", {}) or {}
    for key in ("f1", "average_precision", "auc", "accuracy"):
        value = overall.get(key)
        if value is not None:
            return float(value)
    return float("-inf")


def _flatten_task_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested task metrics into CSV-friendly columns."""
    task_metrics = results.get("task_metrics", {}) or {}
    flattened: Dict[str, Any] = {}
    for task_name, metrics in task_metrics.items():
        if not isinstance(metrics, dict):
            flattened[f"task_{task_name}"] = metrics
            continue
        for metric_name, metric_value in metrics.items():
            flattened[f"task_{task_name}_{metric_name}"] = metric_value
    return flattened


def run_iac_grid_search(args: argparse.Namespace) -> None:
    print("\n=== Experiment: iac_grid_search ===")
    task_names = list(DISCOURSE_BINARY_COLUMNS)
    task_weights = [1.0] * len(task_names)

    learning_rates = [1e-4, 3e-4]
    weight_decays = [0.005, 0.01]
    warmup_fracs = [0.05, 0.1]
    batch_sizes = [128, 256]
    dropout_rates = [0.2, 0.4]

    combos = list(
        itertools.product(
            learning_rates,
            weight_decays,
            warmup_fracs,
            batch_sizes,
            dropout_rates,
        )
    )
    print(f"Running {len(combos)} hyperparameter combinations.")

    all_results: List[Dict[str, Any]] = []
    model_config = _build_iac_config()
    for idx, (lr, wd, warmup_frac, batch_size, dropout) in enumerate(combos, start=1):
        print("\n" + "=" * 80)
        print(
            f"Combo {idx:02d}/{len(combos)} | optimizer=adamw lr={lr} wd={wd} "
            f"warmup={warmup_frac} batch={batch_size} dropout={dropout}"
        )
        print("=" * 80)

        train_ds, val_ds, _ = load_splits(
            label_columns=task_names,
            batch_size=batch_size,
            data_dir=args.data_dir,
        )

        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=batch_size,
            optimizer_type="adamw",
            learning_rate=lr,
            weight_decay_rate=wd,
            dropout_rate=dropout,
            warmup_percent=warmup_frac * 100.0,
        )

        results = run_experiment(
            model_config,
            train_ds,
            val_ds,
            optimize_thresholds=not args.no_optimize_thresh,
            verbose=args.verbose,
            training_config=training_config,
            task_weights=task_weights,
        )

        overall = results.get("overall_metrics", {}) or {}
        f1 = overall.get("f1")
        ap = overall.get("average_precision")
        auc = overall.get("auc")
        acc = overall.get("accuracy")
        print(
            "Combo metrics | "
            f"f1={f1 if f1 is not None else 'NA'} "
            f"ap={ap if ap is not None else 'NA'} "
            f"auc={auc if auc is not None else 'NA'} "
            f"acc={acc if acc is not None else 'NA'}"
        )

        all_results.append(
            {
                "combo_index": idx,
                "optimizer": "adamw",
                "learning_rate": lr,
                "weight_decay_rate": wd,
                "warmup_fraction": warmup_frac,
                "batch_size": batch_size,
                "dropout_rate": dropout,
                "overall_f1": f1,
                "overall_average_precision": ap,
                "overall_auc": auc,
                "overall_accuracy": acc,
                "primary_metric": _extract_primary_metric(results),
                **_flatten_task_metrics(results),
            }
        )

    results_df = pd.DataFrame(all_results).sort_values(
        "primary_metric", ascending=False
    )
    print("\n=== Top 10 configurations by primary metric ===")
    print(
        results_df[
            [
                "combo_index",
                "learning_rate",
                "weight_decay_rate",
                "warmup_fraction",
                "batch_size",
                "dropout_rate",
                "overall_f1",
                "overall_average_precision",
                "overall_auc",
                "overall_accuracy",
                "primary_metric",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    if args.results_csv:
        out_dir = os.path.dirname(args.results_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        results_df.to_csv(args.results_csv, index=False)
        print(f"\nSaved full grid results to: {args.results_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IAC-only discourse-binary grid search experiments.",
    )
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--no-optimize-thresh",
        action="store_true",
        help="Disable threshold optimization for binary tasks.",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument(
        "--results-csv",
        default="cluster_runs/outputs/no_prop_iac_grid_results.csv",
        help="Where to save full grid metrics CSV (empty string disables write).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(f"{args.data_dir}/no_prop_overlap_train.csv"):
        print(
            f"ERROR: data not found at {args.data_dir}/no_prop_overlap_train.csv\n"
            "Make sure --data-dir points to the directory containing the no_prop_overlap_*.csv files.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_iac_grid_search(args)


if __name__ == "__main__":
    main()
