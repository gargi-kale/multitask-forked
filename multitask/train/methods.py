"""Named training methods, layer-size presets and the settings of one run.

A run = a dataset + a method + optional add-ons (frozen stage 2, encoder,
...) + optional ablations. ``RunSettings`` holds all of it; its defaults are
the re-tuned setup every reported result used (learning rate 1e-4, keep the
epoch with the best mean validation F1 and stop after 3 epochs without
improvement, "wide" layers, no dropout, batches of 256, inputs up to 512 word
pieces).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Tuple

from multitask.config import TrainingConfig

# The paper's training settings (lr 3e-4, AdamW, weight decay 0.01, dropout
# 0.4, batch 256, 5% warmup). RunSettings overrides lr and dropout with the
# re-tuned values.
PAPER_TRAINING_CONFIG = TrainingConfig(
    optimizer_type="adamw",
    learning_rate=0.0003,
    weight_decay_rate=0.01,
    dropout_rate=0.4,
    batch_size=256,
    warmup_percent=5
)

# Head sizes (--arch): (shared layer sizes, [group-level sizes, task-level sizes]).
# "paper" is what the paper describes; "legacy" is what multitask_basic/models/model.py
# actually builds (each size appears twice because it zips one dense layer per size
# pair). "wide" keeps the paper's depth at the legacy widths; "medium" sits between.
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


def architecture(spec: str) -> Tuple[List[int], List[List[int]]]:
    """A preset name, or custom sizes "SHARED/GROUP/TASK" with comma-separated
    widths, e.g. "256,128/128,64/64,32" (= "wide")."""
    if spec in PAPER_ARCHITECTURES:
        return PAPER_ARCHITECTURES[spec]
    m = re.fullmatch(r"([\d,]+)/([\d,]+)/([\d,]+)", spec.replace(" ", ""))
    if not m:
        raise ValueError(f"--arch {spec!r}: a preset ({', '.join(PAPER_ARCHITECTURES)}) or "
                         "SHARED/GROUP/TASK widths like 256,128/128,64/64,32")
    shared, group, task = ([int(x) for x in g.split(",") if x] for g in m.groups())
    return shared, [group, task]


# Method name -> RunSettings fields it sets. Every method uses the pooled
# loss trainer (a merged output, if the dataset has one, trains on its pooled
# score) and the epoch scorer.
METHODS: Dict[str, Dict[str, Any]] = {
    # fixed task weights: each task group weighted by 1 / its labelled rows
    "plain": {},
    # FAMO (Liu et al. 2023) re-weights the tasks every step
    "famo": {"loss_balancer": "famo"},
    # DB-MTL (Lin et al. 2023): log-loss scaling + gradient-magnitude balancing
    "dbmtl": {"loss_balancer": "dbmtl"},
    # FAMO + 0.03 x masked-word loss on the encoder
    "famo_mlm": {"loss_balancer": "famo", "mlm_lambda": 0.03},
    # FAMO + consistency: a copy of each text with 5% of its words masked must
    # give the same scores as the full text, on the 10% most confident cells
    # per side per head (weight 1, no masked-word loss)
    "famo_consf1_m5": {"loss_balancer": "famo", "mlm_lambda": 0.0, "mlm_mask_prob": 0.05,
                       "consistency_weight": 1.0, "consistency_keep": 0.1},
    # one plain model per output, trained on the rows that output labels
    "single_task": {"drop_unlabeled_train": True},
}
DEFAULT_METHOD = "famo_consf1_m5"

METHOD_HELP = {
    "plain": "fixed task weights",
    "famo": "FAMO adaptive task weights",
    "dbmtl": "DB-MTL loss and gradient balancing",
    "famo_mlm": "FAMO + 0.03 x masked-word loss",
    "famo_consf1_m5": "FAMO + consistency on 5%-masked copies (default)",
    "single_task": "one model per output (the single-task baseline)",
}


@dataclass
class RunSettings:
    method: str = DEFAULT_METHOD
    # model and optimisation
    encoder: str = "small-bert"
    max_length: int = 512
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_percent: float = 5.0
    dropout: float = 0.0
    arch: str = "wide"
    stop_metric: str = "f1"
    patience: int = 3
    evals_per_epoch: int = 1
    seed: int = 42
    length_chunk_batches: int = 0
    layer_lr_decay: float = 1.0
    clipnorm: float = 0.0
    no_decay_bias_norm: bool = False
    embedding_cache: str = ""
    # frozen stage 2
    stage2: bool = False
    stage2_lrs: str = "1e-4,3e-5"
    stage2_epochs: int = 10
    stage2_batch_size: int = 0
    # method knobs (set by METHODS; can be overridden with --set)
    loss_balancer: str = ""
    mlm_lambda: Optional[float] = None
    mlm_mask_prob: float = 0.15
    mlm_mode: str = "fixed"
    consistency_weight: float = 0.0
    consistency_keep: float = 0.0
    consistency_floor: float = 0.0
    consistency_rampup_epochs: float = 1.0
    masked_label_weight: float = 0.0
    drop_unlabeled_train: bool = False
    # ablations
    hide_label: str = ""          # hide this output's labels on every train row
    hide_share: float = 0.0       # hide this share of every label's train cells, at random
    label_budget: Dict[str, float] = field(default_factory=dict)  # {task: share of train labels kept}
    hide_source: str = ""         # "SOURCE:TASKS": remove those labels on that source's rows
    keep_sources: List[str] = field(default_factory=list)
    unweighted_loss: bool = False
    no_grouping: bool = False
    probes: str = ""              # "", "hidden", "all" or a comma list of outputs
    probe_untrained: bool = False
    repeat_train: int = 1
    # output
    save_predictions: bool = True
    verbose: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def settings_for(method: str, **overrides: Any) -> RunSettings:
    """RunSettings for ``method`` with ``overrides`` on top."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    s = replace(RunSettings(method=method), **METHODS[method])
    return replace(s, **overrides) if overrides else s


def parse_set(items: List[str]) -> Dict[str, Any]:
    """--set key=value pairs, typed by RunSettings' field types."""
    types = {f.name: f.type for f in fields(RunSettings)}
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set {item!r}: expected key=value")
        key, raw = (x.strip() for x in item.split("=", 1))
        key = key.replace("-", "_")
        if key not in types:
            raise ValueError(f"--set {key!r}: unknown setting; known: {', '.join(sorted(types))}")
        t = str(types[key])
        if "bool" in t:
            if raw.lower() not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
                raise ValueError(f"--set {key}: expected true/false, got {raw!r}")
            val: Any = raw.lower() in ("1", "true", "yes", "on")
        elif "Optional[float]" in t:
            val = None if raw.lower() in ("none", "") else float(raw)
        elif t in ("float", "<class 'float'>"):
            val = float(raw)
        elif t in ("int", "<class 'int'>"):
            val = int(raw)
        elif "Dict" in t or "List" in t:
            raise ValueError(f"--set {key}: use its own command-line option")
        else:
            val = raw
        out[key] = val
    return out


def run_name(s: RunSettings) -> str:
    """Folder name of a run: method, then each non-default add-on or ablation."""
    base = RunSettings()
    parts = [s.method]
    if s.encoder != base.encoder:
        parts.append(s.encoder)
    if s.arch != base.arch:
        parts.append("arch-" + re.sub(r"[^A-Za-z0-9]+", "-", s.arch))
    if s.hide_label:
        parts.append(f"hide-{s.hide_label}")
    if s.hide_share:
        parts.append(f"hide{100 * s.hide_share:g}pct")
    for t, share in sorted(s.label_budget.items()):
        parts.append(f"budget-{t}-{100 * share:g}pct")
    if s.hide_source:
        parts.append("nosrc-" + s.hide_source.split(":")[0])
    if s.keep_sources:
        parts.append("only-" + "-".join(s.keep_sources))
    if s.unweighted_loss:
        parts.append("unweighted")
    if s.no_grouping:
        parts.append("nogroups")
    if s.probe_untrained:
        parts.append("untrained-probe")
    if s.stage2:
        parts.append("stage2")
    parts.append(f"s{s.seed}")
    return "_".join(re.sub(r"[^A-Za-z0-9._\-]+", "-", p) for p in parts)
