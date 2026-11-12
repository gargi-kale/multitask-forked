"""Checks for the masked-word (MLM) auxiliary loss in test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_masked_lm_loss.py

Downloads google/bert_uncased_L-2_H-128_A-2 and the distilbert-base-uncased
tokenizer (the one load_splits uses), so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    CLS_ID, ENCODER_CONFIGS, FIRST_WORD_ID, MASK_ID, MLM_IGNORE, PAD_ID, SEP_ID,
    MaskedLMMultiTaskModel, PooledOutputLossTrainer, load_pretrained_mlm_head, mask_tokens,
)
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

M = -1.0
NAME = "google/bert_uncased_L-2_H-128_A-2"
SENTENCES = [
    "The capital of France is Paris, and it is a very large city.",
    "She drank a cup of hot coffee before going to work in the morning.",
    "The government announced a new tax on imported cars last week.",
    "I think you are completely wrong about the cause of the war.",
    "Children should go to school every day to learn how to read and write.",
    "He played the guitar while his sister sang a song for the family.",
    "The president said the country would never give up its freedom.",
    "We need more evidence before we can agree with your argument.",
    "The sun rises in the east and sets in the west.",
    "Please close the door when you leave the room.",
    "The doctor told him to stop smoking and to eat more vegetables.",
    "They watched a movie at the cinema on Saturday night.",
]


def _ids(rows):
    return tf.constant(rows, tf.int32)


def test_masking_rule():
    tf.random.set_seed(0)
    b, length = 256, 128
    rng = np.random.default_rng(0)
    lens = rng.integers(10, length, size=b)
    ids = np.zeros((b, length), np.int32)
    for i, n in enumerate(lens):
        ids[i, :n] = rng.integers(FIRST_WORD_ID, 30522, size=n)
        ids[i, 0], ids[i, n - 1] = CLS_ID, SEP_ID
    am = (np.arange(length)[None, :] < lens[:, None]).astype(np.int32)
    masked, answers = mask_tokens(_ids(ids), _ids(am), 30522)
    masked, answers = masked.numpy(), answers.numpy()

    picked = answers != MLM_IGNORE
    candidate = (am == 1) & (ids != CLS_ID) & (ids != SEP_ID) & (ids != PAD_ID)
    assert not np.any(picked & ~candidate)          # never padding / [CLS] / [SEP]
    np.testing.assert_array_equal(masked[~picked], ids[~picked])
    np.testing.assert_array_equal(answers[picked], ids[picked])
    assert abs(picked.sum() / candidate.sum() - 0.15) < 0.01
    to_mask = (masked == MASK_ID)[picked].mean()
    unchanged = (masked == ids)[picked].mean()
    replaced = 1 - to_mask - unchanged
    assert abs(to_mask - 0.8) < 0.02 and abs(unchanged - 0.1) < 0.02 and abs(replaced - 0.1) < 0.02
    assert np.all(masked[picked & (masked != MASK_ID)] >= FIRST_WORD_ID)


@pytest.fixture(scope="module")
def batch():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)
    enc = tok(SENTENCES, padding="max_length", truncation=True, max_length=32, return_tensors="np")
    x = {"input_ids": tf.constant(enc["input_ids"], tf.int32),
         "attention_mask": tf.constant(enc["attention_mask"], tf.int32)}
    n = len(SENTENCES)
    y = {"a": tf.constant(np.tile([1.0, 0.0], n // 2).reshape(-1, 1), tf.float32),
         "b": tf.constant(np.tile([0.0, 1.0, M], n // 3).reshape(-1, 1), tf.float32)}
    return x, y


@pytest.fixture(scope="module")
def head():
    return load_pretrained_mlm_head(NAME, from_pt=True)


def make_model(batch, head, mlm_lambda):
    x, _ = batch
    cfg = ModelConfig(
        task_structure=[[2]], task_names=["a", "b"], task_types=["binary", "binary"],
        num_classes_per_task=[2, 2], shared_layer_sizes=[16], default_layer_sizes=[8],
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )
    model = MaskedLMMultiTaskModel(cfg, dropout_rate=0.0)
    model(x, training=False)
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=[],
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        2, [{0: 0.7, 1: 1.6}, {0: 1.2, 1: 0.8}], [1.0, 1.5], task_type_mapping={"a": "t", "b": "u"})
    model.attach_mlm_head(head, mlm_lambda)
    return model


def test_head_matches_pretrained_masked_lm(batch, head):
    from transformers import TFBertForMaskedLM
    x, _ = batch
    src = TFBertForMaskedLM.from_pretrained(NAME, from_pt=True, use_safetensors=False)
    ref = src(x, training=False).logits.numpy()
    model = make_model(batch, head, 0.0)
    # The head takes [positions, hidden], as in training (picked positions only).
    hidden = tf.reshape(model.encoder(x, training=False).last_hidden_state, [-1, 128])
    ours = head(hidden, model.encoder.get_input_embeddings().weight).numpy()
    np.testing.assert_allclose(ours, ref.reshape(ours.shape), atol=1e-3)


def test_pretrained_head_guesses_far_above_chance(batch, head):
    tf.random.set_seed(1)
    model = make_model(batch, head, 0.0)
    accs = [float(model.mlm_loss(batch[0], training=False)[1]) for _ in range(20)]
    # Chance is ~1/30522; untrained or untied weights give ~0.
    assert np.mean(accs) > 0.2, accs


def test_lambda_zero_gradients_match_supervised_loss(batch, head):
    x, y = batch
    model = make_model(batch, head, 0.0)
    variables = model.trainable_variables
    with tf.GradientTape() as tape:
        total, sup, _, _ = model.train_objective(x, y, training=False)
    g_new = tape.gradient(total, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    with tf.GradientTape() as tape:
        y_pred = model(x, training=False)
        ref = model._mtl_loss_fn(y, {k: y_pred[k] for k in model.task_names})
    g_ref = tape.gradient(ref, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    np.testing.assert_allclose(float(total), float(ref), rtol=1e-6)
    for v, a, b in zip(variables, g_new, g_ref):
        np.testing.assert_allclose(tf.convert_to_tensor(a).numpy(), tf.convert_to_tensor(b).numpy(),
                                   atol=1e-7, err_msg=v.name)


def test_masked_word_loss_never_reaches_task_heads(batch, head):
    model = make_model(batch, head, 1.0)
    enc_ids = {id(v) for v in model.encoder.trainable_variables}
    mlm_ids = {id(v) for v in model.mlm_head.trainable_variables}
    task_vars = [v for v in model.trainable_variables if id(v) not in enc_ids | mlm_ids]
    enc_vars = [v for v in model.trainable_variables if id(v) in enc_ids]
    assert task_vars and enc_vars
    with tf.GradientTape(persistent=True) as tape:
        loss, _ = model.mlm_loss(batch[0], training=True)
    for g in tape.gradient(loss, task_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO):
        assert not np.any(tf.convert_to_tensor(g).numpy())
    enc_grads = tape.gradient(loss, enc_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    assert any(np.any(tf.convert_to_tensor(g).numpy()) for g in enc_grads)


def test_fit_logs_both_losses_and_lowers_mlm_loss(batch, head):
    tf.random.set_seed(2)
    x, y = batch
    model = make_model(batch, head, 1.0)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    ds = tf.data.Dataset.from_tensors((x, y)).repeat(8)
    hist = model.fit(ds, epochs=5, validation_data=tf.data.Dataset.from_tensors((x, y)),
                     verbose=0).history
    for key in ("loss", "mlm_loss", "mlm_acc", "val_loss", "val_mlm_loss", "val_mlm_acc"):
        assert key in hist and all(np.isfinite(hist[key])), (key, hist)
    assert all(v > 0 for v in hist["val_mlm_loss"])  # the tracker is updated, not stale
    assert hist["mlm_loss"][-1] < hist["mlm_loss"][0]
