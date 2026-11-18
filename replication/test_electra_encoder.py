"""Checks for the ELECTRA-small encoder option (--encoder electra-small) in
test_no_prop_overlap.py: the model reads the [CLS] position (ELECTRA has no
pooler), the encoder is fine-tuned, and the famo_consf1_m5 setup (FAMO +
consistency on a masked copy, masked-word weight 0) runs without the
masked-word head that ELECTRA's discriminator checkpoint lacks.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_electra_encoder.py

Downloads google/electra-small-discriminator, so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, LabelAwareMaskedModel, LabelAwareMaskedTrainer, NoMLMHead,
    attach_mlm_balancing,
)
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

M = -1.0
NAMES = ["p1", "p2", "wa", "d1", "d2"]
NUM_LOSSES = 4  # pooled propaganda, wa, d1, d2


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    n, length = 32, 24
    ids = rng.integers(999, 30522, size=(n, length)).astype(np.int32)
    ids[:, 0], ids[:, -1] = 101, 102
    x = {"input_ids": tf.constant(ids), "attention_mask": tf.ones_like(tf.constant(ids))}
    y = {k: np.full((n, 1), M, np.float32) for k in NAMES}
    for k, rows in (("p1", range(16)), ("p2", range(16)), ("wa", range(16, 24)),
                    ("d1", range(24, 28)), ("d2", range(24, 28))):
        y[k][list(rows), 0] = rng.integers(0, 2, size=len(rows))
    return x, {k: tf.constant(v) for k, v in y.items()}


def make_trainer(mlm_lambda=0.0, encoder="electra-small"):
    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS[encoder],
    )
    model = LabelAwareMaskedModel(cfg, dropout_rate=0.0)
    trainer = LabelAwareMaskedTrainer(
        model, TrainingConfig(), mlm_lambda=mlm_lambda, mask_prob=0.05, balancer="famo",
        mlm_mode="fixed", masked_label_weight=0.0, consistency_weight=1.0,
        consistency_keep=0.1, pooled_tasks=["p1", "p2"],
        pooled_class_weights={0: 1.0, 1: 1.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 1.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    return trainer


@pytest.fixture(scope="module")
def model(batch):
    """famo_consf1_m5 as the launcher sets it up, on ELECTRA-small."""
    trainer = make_trainer()
    m = trainer.model
    trainer.prepare(tf.data.Dataset.from_tensors(batch))
    attach_mlm_balancing(m, "famo", "fixed", NUM_LOSSES)
    m.attach_label_aware(0.0, 1.0, 0, 0.1)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=m._mtl_loss_fn)
    return m


def test_reads_the_cls_position(batch, model):
    x, _ = batch
    enc = model.encoder(x, training=False)
    assert getattr(enc, "pooler_output", None) is None
    assert enc.last_hidden_state.shape == (32, 24, 256)
    np.testing.assert_allclose(model._encode_input(x, training=False).numpy(),
                               enc.last_hidden_state[:, 0, :].numpy(), atol=1e-6)


def test_checkpoint_is_the_pretrained_one(model):
    cfg = model.encoder.config
    assert (cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size) == (
        "electra", 12, 256, 30522)
    # Freshly initialised weights would have a word-embedding sd of 0.02.
    emb = model.encoder.get_input_embeddings().weight.numpy()
    assert emb.shape == (30522, 128) and emb.std() > 0.03


def test_no_head_is_attached_and_a_weighted_masked_word_loss_is_refused(batch, model):
    assert isinstance(model.mlm_head, NoMLMHead) and model.mlm_head.vocab_size == 30522
    with pytest.raises(ValueError, match="needs a BERT checkpoint"):
        make_trainer(mlm_lambda=0.03).prepare(tf.data.Dataset.from_tensors(batch))


def test_shortened_electra_is_the_first_two_pretrained_layers(batch, model):
    """electra-small-L2 must be ELECTRA-small's embeddings and first 2 layers
    with their pretrained weights, not a freshly initialised 2-layer model:
    its output equals the full encoder's hidden state after layer 2."""
    x, _ = batch
    short_model = make_trainer(encoder="electra-small-L2").model
    short = short_model.encoder
    assert short.config.num_hidden_layers == 2 and len(short.electra.encoder.layer) == 2
    after_layer = model.encoder(x, output_hidden_states=True, training=False).hidden_states
    assert len(after_layer) == 13  # embeddings + 12 layers
    out = short(x, training=False).last_hidden_state.numpy()
    np.testing.assert_allclose(out, after_layer[2].numpy(), atol=1e-4)
    assert not np.allclose(out, after_layer[12].numpy(), atol=1e-2)
    np.testing.assert_allclose(short_model._encode_input(x, training=False).numpy(),
                               out[:, 0, :], atol=1e-6)
    n = sum(int(np.prod(v.shape)) for v in short.trainable_variables)
    full = sum(int(np.prod(v.shape)) for v in model.encoder.trainable_variables)
    assert 5.5e6 < n < 5.7e6 and 13.4e6 < full < 13.6e6, (n, full)


def test_frozen_stage2_runs_on_electra():
    """The frozen stage 2 (re-train each task's own block on a frozen model)
    on an ELECTRA model: it runs to the end, scores every output, and leaves
    the encoder's weights bit-identical."""
    from test_no_prop_overlap import run_stage2
    from test_stage2 import OUTPUTS, make
    m, trainer, train, val, test = make(encoder="electra-small")
    assert m.encoder.config.model_type == "electra"
    before = [v.numpy() for v in m.encoder.variables]
    out = run_stage2(trainer, train, val, test, lrs=["0.05"], epochs=2, outputs=OUTPUTS,
                     batch_size=16, seed=0)
    assert out["frozen_unchanged"] is True
    for a, v in zip(before, m.encoder.variables):
        np.testing.assert_array_equal(a, v.numpy())
    run = out["runs"]["0.05"]
    assert set(run["chosen_epoch"]) == set(OUTPUTS)
    for node in (out["stage1"], run["assembled"]):
        assert np.isfinite(node["val_f1"]) and np.isfinite(node["test_f1"])
    assert run["assembly_max_f1_gap"] < 1e-9


def test_fit_fine_tunes_the_encoder_and_logs(batch, model):
    enc_vars = model.encoder.trainable_variables
    tracked = {id(v) for v in model.trainable_variables}
    assert len(enc_vars) > 100 and all(id(v) in tracked for v in enc_vars)
    before = [v.numpy().copy() for v in enc_vars]
    ds = tf.data.Dataset.from_tensors(batch).repeat(3)
    hist = model.fit(ds, epochs=2, validation_data=tf.data.Dataset.from_tensors(batch),
                     verbose=0).history
    for k in ("loss", "mlm_loss", "consistency_loss", "consistency_kept", "val_loss",
              "val_consistency_loss"):
        assert k in hist and len(hist[k]) == 2 and all(np.isfinite(hist[k])), (k, hist)
    assert hist["mlm_loss"] == [0.0, 0.0]            # no head: logged as 0
    assert all(v > 0 for v in hist["consistency_loss"])  # the masked copy does differ
    moved = sum(not np.array_equal(b, v.numpy()) for b, v in zip(before, enc_vars))
    assert moved > 0.9 * len(enc_vars), f"only {moved} of {len(enc_vars)} encoder weights moved"
