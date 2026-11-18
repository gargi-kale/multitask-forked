"""Train, score and save runs: one function per run, one to plan many.

``train_one`` is the general form of ``run_paper_replication`` (in
replication/test_no_prop_overlap.py): the same data loader, model, trainers,
epoch scorer, frozen stage 2 and probes, with the task list, task groups,
merged output and scored outputs read from a DatasetSpec instead of the IAC
column names.

``plan_runs`` expands one request into the runs it needs (one per seed; one
per output for single-task and for hide-each-label), and ``run_all`` trains
each one that has no result yet.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from multitask.train.dataset import DatasetSpec, materialize, save_spec, describe
from multitask.train.methods import (
    METHODS, PAPER_TRAINING_CONFIG, RunSettings, architecture, run_name,
)

RESULT_FILE = "result.json"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_settings(spec: DatasetSpec, s: RunSettings) -> None:
    """Raise ValueError for settings that can't run together on ``spec``."""
    from multitask.train.encoders import ENCODER_CONFIGS, OPENAI_EMBEDDING_MODELS

    if s.method not in METHODS:
        raise ValueError(f"unknown method {s.method!r}; choose from {', '.join(METHODS)}")
    if s.encoder not in ENCODER_CONFIGS:
        raise ValueError(f"unknown encoder {s.encoder!r}; choose from {', '.join(ENCODER_CONFIGS)}")
    architecture(s.arch)
    if s.stop_metric not in ("f1", "auc", "loss"):
        raise ValueError("stop metric must be f1, auc or loss")
    if s.loss_balancer not in ("", "famo", "dbmtl"):
        raise ValueError("loss_balancer must be '', famo or dbmtl")
    if s.mlm_mode not in ("fixed", "task", "cosine"):
        raise ValueError("mlm_mode must be fixed, task or cosine")
    label_aware = s.consistency_weight > 0 or s.masked_label_weight > 0
    if label_aware and not (s.loss_balancer == "famo" and s.mlm_mode == "fixed"
                            and s.mlm_lambda is not None):
        raise ValueError("consistency / masked-label weights are built on FAMO with a "
                         "masked-word setting: they need loss_balancer=famo, mlm_lambda set "
                         "(0 = no masked-word loss) and mlm_mode=fixed")
    if not 0 <= s.consistency_keep <= 0.5:
        raise ValueError("consistency_keep is a share per side, in [0, 0.5]")
    if s.mlm_mode == "task" and not s.loss_balancer:
        raise ValueError("mlm_mode task needs a loss balancer")
    if s.mlm_mode == "cosine" and s.loss_balancer:
        raise ValueError("mlm_mode cosine can't be combined with a loss balancer")
    if s.mlm_lambda is not None and s.encoder in OPENAI_EMBEDDING_MODELS:
        raise ValueError(f"--encoder {s.encoder} uses stored embeddings (no words to mask): "
                         "use plain, famo, dbmtl or single_task")
    if s.encoder in OPENAI_EMBEDDING_MODELS and (s.probes or s.probe_untrained or s.hide_share):
        raise ValueError(f"--encoder {s.encoder}: probes and hidden-cell scoring need a text encoder")
    if s.encoder in OPENAI_EMBEDDING_MODELS and not s.embedding_cache:
        raise ValueError(f"--encoder {s.encoder} needs --embedding-cache (see `multitask embed`)")
    if s.encoder in OPENAI_EMBEDDING_MODELS and s.layer_lr_decay != 1.0:
        raise ValueError("--layer-lr-decay needs an encoder that is trained")
    if s.length_chunk_batches < 0 or s.evals_per_epoch < 1:
        raise ValueError("length_chunk_batches must be >= 0 and evals_per_epoch >= 1")
    if s.length_chunk_batches and s.encoder in OPENAI_EMBEDDING_MODELS:
        raise ValueError("length-grouped batches need a text encoder")
    if s.evals_per_epoch > 1 and s.stop_metric == "loss":
        raise ValueError("evals_per_epoch > 1 needs stop metric f1 or auc")
    if not 0 < s.layer_lr_decay <= 1 or s.clipnorm < 0:
        raise ValueError("layer_lr_decay must be in (0, 1] and clipnorm >= 0")
    outputs = spec.scored_outputs()
    if s.hide_label and s.hide_label not in outputs:
        raise ValueError(f"--hide-label {s.hide_label!r} is not an output: {outputs}")
    if not 0 <= s.hide_share < 1:
        raise ValueError("--hide-share must be in [0, 1)")
    if s.hide_share and (s.hide_label or s.label_budget):
        raise ValueError("--hide-share is a separate ablation from --hide-label / --label-budget")
    tasks = set(spec.task_names())
    for t, share in s.label_budget.items():
        if t not in tasks:
            raise ValueError(f"--label-budget {t!r}: needs a task column (not a merged output): "
                             f"{sorted(tasks)}")
        if not 0 < share <= 1:
            raise ValueError(f"--label-budget {t}: share must be in (0, 1]")
    if s.probes and s.probes not in ("hidden", "all"):
        bad = [o for o in s.probes.split(",") if o not in outputs]
        if bad:
            raise ValueError(f"--probes: unknown outputs {bad}")
    if s.probes == "hidden" and not s.hide_label:
        raise ValueError("--probes hidden needs --hide-label")
    if s.probe_untrained and (not s.probes or s.probes == "hidden"):
        raise ValueError("--probe-untrained needs --probes all or a list of outputs")
    if s.stage2 and s.stop_metric == "loss":
        raise ValueError("--stage2 needs stop metric f1 or auc")
    if s.repeat_train < 1:
        raise ValueError("repeat_train must be >= 1")
    if s.method != "single_task" and len(outputs) < 2:
        raise ValueError(f"method {s.method} needs at least two outputs; the dataset has "
                         f"{outputs}. Use --method single_task for one label")


def parse_hide_source(spec: DatasetSpec, value: str) -> Optional[Tuple[str, List[str]]]:
    """'SOURCE:TASKS' -> (source, [task names]); TASKS as for DatasetSpec.expand,
    comma-separated (default: every task)."""
    if not value:
        return None
    source, _, refs = value.partition(":")
    tasks = spec.expand([r for r in refs.split(",") if r]) if refs else spec.task_names()
    return source, tasks


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _labels_from_csv(path: str, spec: DatasetSpec) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Train rows and their labels (-1 = none) per task and per merged output."""
    df = pd.read_csv(path)
    out = {t: np.nan_to_num(pd.to_numeric(df[t], errors="coerce").to_numpy(np.float32), nan=-1.0)
           for t in spec.task_names()}
    for p in spec.pools:
        out[p.name] = _any_of(np.stack([out[t] for t in p.tasks], axis=1))
    return df, out


def _any_of(stacked: np.ndarray) -> np.ndarray:
    """Merged label: max over the labelled cells, -1 when none is labelled
    (as the pooled loss derives it)."""
    labelled = stacked != -1
    pooled = np.where(labelled, stacked, 0).max(axis=1).astype(np.float32)
    return np.where(labelled.any(axis=1), pooled, -1.0)


def _ds_labels(ds, spec: DatasetSpec) -> Dict[str, np.ndarray]:
    ys = [y for _, y in ds]
    out = {k: np.concatenate([np.asarray(y[k]).reshape(-1) for y in ys]) for k in ys[0]}
    for p in spec.pools:
        out[p.name] = _any_of(np.stack([out[t] for t in p.tasks], axis=1))
    return out


def _encode_texts(texts: List[str], s: RunSettings, embedding_source=None):
    import tensorflow as tf
    from transformers import AutoTokenizer

    if embedding_source is not None:
        return tf.data.Dataset.from_tensor_slices(
            np.asarray(embedding_source(texts), dtype=np.float32)).batch(s.batch_size)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)  # as load_splits
    enc = tok(texts, padding="max_length", truncation=True, max_length=s.max_length,
              return_tensors="np")
    return tf.data.Dataset.from_tensor_slices(
        {"input_ids": enc["input_ids"].astype(np.int32),
         "attention_mask": enc["attention_mask"].astype(np.int32)}).batch(s.batch_size)


def _cell_scores(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    from sklearn.metrics import f1_score, roc_auc_score

    out: Dict[str, Any] = {"n": int(y.size), "yes_share": float(y.mean()) if y.size else None}
    if y.size == 0 or len(np.unique(y)) < 2:
        return {**out, "f1": None, "auc": None}
    out["f1"] = float(f1_score(y, (p > threshold).astype(int), average="weighted", zero_division=0))
    out["auc"] = float(roc_auc_score(y, p))
    return out


def _split_scores(res: Dict[str, Any], outputs: Sequence[str]) -> Dict[str, Dict[str, float]]:
    merged = {**res.get("task_metrics", {}), **res.get("pool_metrics", {})}
    return {o: {k: float(merged[o][k]) for k in ("f1", "auc") if k in merged.get(o, {})}
            for o in outputs}


def train_one(spec: DatasetSpec, s: RunSettings, out_dir: str) -> Dict[str, Any]:
    """Train one model on the materialized ``spec`` and write
    ``out_dir/result.json`` (+ predictions). Returns the result dict."""
    import tensorflow as tf

    from multitask.config import EncoderIntegration, ModelConfig
    from multitask import MultiTaskModel
    from multitask.train.balancers import BalancedLossTrainer, BalancedMultiTaskModel
    from multitask.train.consistency import LabelAwareMaskedModel, LabelAwareMaskedTrainer
    from multitask.train.data import (
        compute_full_train_binary_class_weights, compute_full_train_pooled_class_weights,
        load_splits, random_hidden_cells,
    )
    from multitask.train.encoders import (
        ENCODER_CONFIGS, OPENAI_EMBEDDING_MODELS, openai_embedding_source,
    )
    from multitask.train.mlm import (
        BalancedMaskedLMModel, BalancedMaskedLMTrainer, MaskedLMMultiTaskModel, MaskedLMTrainer,
    )
    from multitask.train.scoring import EpochScorer
    from multitask.train.stage2 import run_experiment, run_stage2
    from multitask.train.trainer import PooledOutputLossTrainer

    check_settings(spec, s)
    tf.keras.utils.set_random_seed(s.seed)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    groups = spec.groups()
    all_tasks = spec.task_names()
    pool = spec.pool
    pooled_tasks = spec.pooled_tasks()
    outputs = spec.scored_outputs()
    hidden_columns = spec.output_tasks(s.hide_label) if s.hide_label else []
    scored = [o for o in outputs if o != s.hide_label]
    label_share = {t: v for t, v in s.label_budget.items() if v < 1}
    print(f"Outputs: {outputs}" + (f"; hidden: {s.hide_label}" if s.hide_label else ""),
          flush=True)

    encoder_config = ENCODER_CONFIGS[s.encoder]
    embedding_source = (openai_embedding_source(s.encoder, s.embedding_cache)
                        if s.encoder in OPENAI_EMBEDDING_MODELS else None)
    if s.mlm_lambda is not None and encoder_config.encoder_integration != EncoderIntegration.HUGGINGFACE:
        raise ValueError("masked-word settings need a HuggingFace encoder")
    finetune = None
    if s.layer_lr_decay != 1.0 or s.clipnorm or s.no_decay_bias_norm:
        finetune = dict(layer_lr_decay=s.layer_lr_decay, clipnorm=s.clipnorm,
                        decay_bias_norm=not s.no_decay_bias_norm)

    train_ds, val_ds, test_ds = load_splits(
        label_columns=all_tasks,
        encoder_config=encoder_config,
        embedding_source=embedding_source,
        batch_size=s.batch_size,
        data_dir=spec.data_dir,
        prefix="",
        max_length=s.max_length,
        seed=s.seed,
        drop_unlabeled_train=s.drop_unlabeled_train,
        hide_train_columns=hidden_columns,
        train_label_share=label_share,
        repeat_train=s.repeat_train,
        length_chunk_batches=s.length_chunk_batches,
        train_hide_share=s.hide_share,
    )
    class_weights = compute_full_train_binary_class_weights(
        data_dir=spec.data_dir, task_names=all_tasks, prefix="")
    task_type_mapping: Optional[Dict[str, str]] = {t: g for g, ts in groups for t in ts}

    shared_sizes, branch_sizes = architecture(s.arch)
    config = ModelConfig(
        task_structure=[[len(groups)], [len(ts) for _, ts in groups]],
        shared_layer_sizes=shared_sizes,
        branch_layer_sizes=branch_sizes,
        task_names=all_tasks,
        task_types=["binary"] * len(all_tasks),
        num_classes_per_task=[2] * len(all_tasks),
        output_pools={pool.name: {"tasks": pooled_tasks, "strategy": pool.strategy,
                                  "task_type": "binary", "num_classes": 2}} if pool else {},
        encoder_config=encoder_config,
    )
    training_config = replace(PAPER_TRAINING_CONFIG, epochs=s.epochs, early_stopping_patience=2,
                              early_stopping_min_delta=0.0)
    training_config = replace(training_config, batch_size=s.batch_size,
                              learning_rate=s.learning_rate, dropout_rate=s.dropout,
                              weight_decay_rate=s.weight_decay, warmup_percent=s.warmup_percent)
    print(f"Architecture: {s.arch} shared={shared_sizes} branches={branch_sizes}")
    print(f"Training config: {training_config}")

    if s.no_grouping:
        task_type_mapping = None
    uniform_task_weights = None
    if s.unweighted_loss:
        task_type_mapping = None
        uniform_task_weights = [1.0] * len(all_tasks)
        class_weights = [{0: 1.0, 1: 1.0} for _ in all_tasks]
    print(f"Task-type grouping: {'off' if task_type_mapping is None else 'on'}")

    pooled_class_weights = compute_full_train_pooled_class_weights(
        data_dir=spec.data_dir, task_names=pooled_tasks, prefix="") if pool else {0: 1.0, 1: 1.0}
    if s.unweighted_loss:
        pooled_class_weights = {0: 1.0, 1: 1.0}
    skip = spec.skip_map()
    label_aware = s.consistency_weight > 0 or s.masked_label_weight > 0

    def make_trainer(model, cfg):
        kw = dict(pooled_tasks=pooled_tasks, pooled_class_weights=pooled_class_weights,
                  task_loss_options={}, pool_strategy=pool.strategy if pool else "max",
                  technique_loss_weight=0.0,
                  technique_class_weights=[class_weights[all_tasks.index(t)] for t in pooled_tasks],
                  finetune=finetune)
        if label_aware:
            return LabelAwareMaskedTrainer(
                model, cfg, mlm_lambda=s.mlm_lambda, mask_prob=s.mlm_mask_prob,
                balancer=s.loss_balancer, mlm_mode=s.mlm_mode,
                masked_label_weight=s.masked_label_weight,
                consistency_weight=s.consistency_weight,
                consistency_rampup_epochs=s.consistency_rampup_epochs,
                consistency_keep=s.consistency_keep, consistency_floor=s.consistency_floor,
                consistency_skip=skip, **kw)
        if s.mlm_lambda is not None and (s.loss_balancer or s.mlm_mode == "cosine"):
            return BalancedMaskedLMTrainer(model, cfg, mlm_lambda=s.mlm_lambda,
                                           mask_prob=s.mlm_mask_prob, balancer=s.loss_balancer,
                                           mlm_mode=s.mlm_mode, **kw)
        if s.loss_balancer:
            return BalancedLossTrainer(model, cfg, balancer=s.loss_balancer, **kw)
        if s.mlm_lambda is not None:
            return MaskedLMTrainer(model, cfg, mlm_lambda=s.mlm_lambda,
                                   mask_prob=s.mlm_mask_prob, **kw)
        return PooledOutputLossTrainer(model, cfg, **kw)

    if label_aware:
        model_cls = LabelAwareMaskedModel
    elif s.mlm_lambda is not None and (s.loss_balancer or s.mlm_mode == "cosine"):
        model_cls = BalancedMaskedLMModel
    elif s.loss_balancer:
        model_cls = BalancedMultiTaskModel
    elif s.mlm_lambda is not None:
        model_cls = MaskedLMMultiTaskModel
    else:
        model_cls = MultiTaskModel

    train_csv = os.path.join(spec.data_dir, spec.files["train"])
    result: Dict[str, Any] = {
        "dataset": spec.name, "run": os.path.basename(os.path.normpath(out_dir)),
        "settings": s.to_dict(), "outputs": outputs, "scored_outputs": scored,
        "hidden_output": s.hide_label or None,
        "groups": {g: ts for g, ts in groups},
        "pool": {"name": pool.name, "tasks": pooled_tasks, "strategy": pool.strategy} if pool else None,
        "labels": {o: spec.label(o) for o in outputs},
        "reference": spec.reference,
        "class_weights": {t: {str(k): float(v) for k, v in w.items()}
                          for t, w in zip(all_tasks, class_weights)},
    }

    if s.probe_untrained:
        # Control for the hide-label probes: the same probes on the model before
        # any training ('encoder' = the pretrained encoder; 'shared' / 'group'
        # are randomly initialised layers on top).
        model = MultiTaskModel(config, dropout_rate=training_config.dropout_rate)
        names = outputs if s.probes == "all" else s.probes.split(",")
        result["probes"] = _probes(model, spec, names, train_csv, val_ds, test_ds, s)
        result["untrained"] = True
        return _write(result, out_dir, started)

    scorer = EpochScorer(val_ds, test_ds, monitor=s.stop_metric, patience=s.patience,
                         evals_per_epoch=s.evals_per_epoch, outputs=scored)
    training_config = replace(training_config, early_stopping=False)

    def trainer_factory(model, cfg):
        scorer.trainer = make_trainer(model, cfg)
        return scorer.trainer

    results = run_experiment(
        config, train_ds, val_ds,
        epochs=s.epochs, batch_size=s.batch_size, learning_rate=s.learning_rate,
        task_type_mapping=task_type_mapping, class_weights=class_weights,
        task_weights=uniform_task_weights, optimize_thresholds=True, verbose=s.verbose,
        show_structure=False, training_config=training_config, test_ds=test_ds,
        trainer_factory=trainer_factory, model_cls=model_cls, callbacks=[scorer],
    )
    trainer = scorer.trainer
    result.update({
        "best_epoch": scorer.best_epoch,
        "epoch_scores": scorer.records,
        "val": _split_scores(results, outputs),
        "test": _split_scores(results.get("test", {}), outputs),
        "thresholds": {k: (None if v is None else np.asarray(v, dtype=float).tolist())
                       for k, v in (results.get("thresholds_used") or {}).items()},
        "history": results.get("history", {}),
    })
    if s.save_predictions:
        arrays = {}
        for split, res in (("val", results), ("test", results.get("test", {}))):
            for name, prob in (res.get("probabilities_dict") or {}).items():
                arrays[f"{split}/prob/{name}"] = np.asarray(prob, dtype=np.float32)
            for name, label in (res.get("labels_dict") or {}).items():
                arrays[f"{split}/label/{name}"] = np.asarray(label, dtype=np.float32)
        np.savez_compressed(os.path.join(out_dir, "predictions.npz"), **arrays)

    if s.hide_share:
        # Score the hidden train cells (best epoch restored) with the val cut-offs.
        df, true = _labels_from_csv(train_csv, spec)
        cells = random_hidden_cells(true, all_tasks, s.hide_share, s.seed)
        pred = trainer.model.predict(_encode_texts(
            df["text"].fillna("").astype(str).tolist(), s, embedding_source), verbose=0)
        thresholds = trainer.optimal_thresholds
        hidden_scores = {}
        for t, rows in cells.items():
            p = np.asarray(pred[t]).reshape(-1)
            hidden_scores[t] = _cell_scores(true[t][rows], p[rows], float(thresholds[t]))
        result["hidden_cells"] = {"share": s.hide_share, "scores": hidden_scores}

    if s.stage2:
        result["stage2"] = run_stage2(
            trainer, train_ds, val_ds, test_ds,
            lrs=[lr for lr in s.stage2_lrs.split(",") if lr], epochs=s.stage2_epochs,
            outputs=scored, batch_size=s.stage2_batch_size or s.batch_size,
            weight_decay=s.weight_decay, seed=s.seed,
            predictions_prefix=os.path.join(out_dir, "predictions") if s.save_predictions else None)

    if s.probes:
        names = (outputs if s.probes == "all" else [s.hide_label] if s.probes == "hidden"
                 else s.probes.split(","))
        result["probes"] = _probes(trainer.model, spec, names, train_csv, val_ds, test_ds, s)
    return _write(result, out_dir, started)


def _probes(model, spec, names, train_csv, val_ds, test_ds, s: RunSettings):
    from multitask.train.probes import run_probes

    df, train_labels = _labels_from_csv(train_csv, spec)
    return run_probes(model, list(names), texts=df["text"].fillna("").astype(str).tolist(),
                      train_labels={n: train_labels[n] for n in names}, val_ds=val_ds,
                      val_labels=_ds_labels(val_ds, spec), test_ds=test_ds,
                      test_labels=_ds_labels(test_ds, spec), max_length=s.max_length,
                      batch_size=s.batch_size, seed=s.seed)


def _write(result: Dict[str, Any], out_dir: str, started: float) -> Dict[str, Any]:
    result["seconds"] = round(time.time() - started, 1)
    tmp = os.path.join(out_dir, f".{RESULT_FILE}.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, default=float)
    os.replace(tmp, os.path.join(out_dir, RESULT_FILE))
    print(f"Wrote {os.path.join(out_dir, RESULT_FILE)}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Many runs
# ---------------------------------------------------------------------------

def plan_runs(spec: DatasetSpec, s: RunSettings, seeds: Sequence[int], *,
              hide_each: bool = False) -> List[Tuple[str, DatasetSpec, RunSettings]]:
    """(run folder name, spec, settings) for every run a request needs.

    single_task: one run per output, each on that output's tasks only.
    hide_each: a reference run with every label, then one run per output with
    that output hidden (probes: every output on the reference, the hidden one
    on each hide run, unless --probes says otherwise)."""
    runs: List[Tuple[str, DatasetSpec, RunSettings]] = []
    for seed in seeds:
        base = replace(s, seed=seed)
        if s.method == "single_task":
            for o in spec.scored_outputs():
                one = spec.select(spec.output_tasks(o))
                sub = replace(base, hide_label="", probes="")
                runs.append((f"single_task_{o}_s{seed}", one, sub))
        elif hide_each:
            ref = replace(base, hide_label="", probes=s.probes or "all")
            runs.append((run_name(ref).replace(f"_s{seed}", f"_full_s{seed}"), spec, ref))
            for o in spec.scored_outputs():
                hid = replace(base, hide_label=o, probes=s.probes or "hidden")
                runs.append((run_name(hid), spec, hid))
        else:
            runs.append((run_name(base), spec, base))
    return runs


def prepare_data(specs: Sequence[DatasetSpec], s: RunSettings, out_root: str,
                 verbose: bool = True) -> DatasetSpec:
    """Materialize ``specs`` for ``s`` (keep_sources / hide_source applied)."""
    from multitask.train.dataset import join_specs

    joined = join_specs(list(specs))
    hide = parse_hide_source(joined, s.hide_source)
    return materialize(specs, os.path.join(out_root, ".prepared"), keep_sources=s.keep_sources,
                       hide_source=hide, verbose=verbose)


def run_all(specs: Sequence[DatasetSpec], s: RunSettings, out_root: str, *,
            seeds: Sequence[int], hide_each: bool = False, overwrite: bool = False,
            dry_run: bool = False) -> List[str]:
    """Train every planned run that has no result yet; returns the run folders."""
    prepared = prepare_data(specs, s, out_root, verbose=not dry_run)
    runs = plan_runs(prepared, s, seeds, hide_each=hide_each)
    for _, spec, settings in runs:
        check_settings(spec, settings)
    print(describe(prepared))
    folders = []
    for name, spec, settings in runs:
        folder = os.path.join(out_root, name)
        folders.append(folder)
        done = os.path.exists(os.path.join(folder, RESULT_FILE))
        status = "done, skipped" if done and not overwrite else "to run"
        print(f"  {name}: {status}")
    if dry_run:
        return folders
    for (name, spec, settings), folder in zip(runs, folders):
        if os.path.exists(os.path.join(folder, RESULT_FILE)) and not overwrite:
            continue
        print(f"\n=== {name} ===", flush=True)
        os.makedirs(folder, exist_ok=True)
        for i, original in enumerate(specs):
            save_spec(original, os.path.join(
                folder, "dataset.yaml" if len(specs) == 1 else f"dataset_{original.name}.yaml"),
                header="The dataset description this run used. Copy and edit it to change\n"
                       "task groups or merged outputs, then pass it with --data.")
        train_one(spec, settings, folder)
    return folders
