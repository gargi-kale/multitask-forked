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

# The library code that used to live here moved to multitask/train/ (and the
# methods that were not adopted to replication/experimental_methods.py).
# Re-exported so existing scripts and tests keep importing from this file.
from multitask.train.columns import (  # noqa: F401  (re-exported for old imports)
    PAPER_TABLE3_MULTITASK_F1,
    DISCOURSE_BINARY_COLUMNS,
    DISCOURSE_REGRESSION_COLUMNS,
    PROPAGANDA_BINARY_LABEL_MAP,
    PROPAGANDA_BINARY_TASK_NAMES,
    PROPAGANDA_POOLS,
    PROPAGANDA_TECHNIQUE_COLUMNS,
    _sanitize_task_name,
)
from multitask.train.encoders import (  # noqa: F401  (re-exported for old imports)
    ENCODER_CONFIGS,
    OPENAI_EMBEDDING_DIMS,
    OPENAI_EMBEDDING_MODELS,
    ecfg_distilbert_uncased,
    ecfg_electra_small,
    ecfg_small_bert,
    ecfg_tfhub_bert,
    openai_embedding_source,
    precomputed_embeddings,
    shortened_electra_small,
)
from multitask.train.data import (  # noqa: F401  (re-exported for old imports)
    ROW_ID_KEY,
    batch_strata,
    compute_full_train_binary_class_weights,
    compute_full_train_pooled_class_weights,
    dealt_batches,
    length_grouped_batches,
    load_splits,
    trim_padding,
)
from multitask.train.optim import (  # noqa: F401  (re-exported for old imports)
    LayerwiseAdamW,
    NO_DECAY_NAMES,
    build_finetune_optimizer,
    layerwise_lr_groups,
)
from multitask.train.trainer import (  # noqa: F401  (re-exported for old imports)
    PooledOutputLossTrainer,
    binary_kl,
)
from multitask.train.balancers import (  # noqa: F401  (re-exported for old imports)
    BalancedLossTrainer,
    BalancedMultiTaskModel,
    DBMTL,
    FAMO,
    shared_variables,
)
from multitask.train.mlm import (  # noqa: F401  (re-exported for old imports)
    CLS_ID,
    FIRST_WORD_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    BalancedMaskedLMModel,
    BalancedMaskedLMTrainer,
    MLMHead,
    MLM_IGNORE,
    MLM_MODES,
    MaskedLMMultiTaskModel,
    MaskedLMTrainer,
    NoMLMHead,
    attach_mlm_balancing,
    load_pretrained_mlm_head,
    mask_tokens,
)
from multitask.train.consistency import (  # noqa: F401  (re-exported for old imports)
    CONSISTENCY_TEACHERS,
    ConsistencyMixin,
    LABEL_AWARE_TRACKERS,
    LabelAwareMaskedModel,
    LabelAwareMaskedTrainer,
    TEACHER_TRACKERS,
    TeacherConsistencyModel,
    TeacherConsistencyTrainer,
    _Untracked,
    check_consistency_filter,
    confident_cells,
)
from experimental_methods import (  # noqa: F401  (re-exported for old imports)
    GradientProbe,
    SELF_DISTILL_SIGNALS,
    SSL_DEFAULT_RATES,
    SSL_POOL,
    SSL_SOURCES,
    SSL_TRACKERS,
    SelfDistillCallback,
    SelfDistillFamoModel,
    SelfDistillFamoTrainer,
    SelfDistillMixin,
    SelfDistillModel,
    SelfDistillTrainer,
    SemiSupervisedModel,
    SemiSupervisedTrainer,
    _DistillState,
    parse_ssl_rates,
    semi_supervised_terms,
    ssl_pair_label,
)
from multitask.train.scoring import (  # noqa: F401  (re-exported for old imports)
    EARLY_STOP_METRICS,
    EpochScorer,
    mask_sweep,
)
from multitask.train.stage2 import (  # noqa: F401  (re-exported for old imports)
    Stage2,
    run_experiment,
    run_stage2,
    table3_scores,
)


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

# PAPER_TRAINING_CONFIG and PAPER_ARCHITECTURES moved to multitask/train/methods.py.
from multitask.train.methods import PAPER_ARCHITECTURES, PAPER_TRAINING_CONFIG  # noqa: E402,F401


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
