"""Fine-tuning optimizer: AdamW with layer-wise learning-rate decay.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import tensorflow as tf


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
