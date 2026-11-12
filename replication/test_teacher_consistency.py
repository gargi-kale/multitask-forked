"""Checks for TeacherConsistencyModel (FAMO plus consistency with a dropout-off
teacher: the current weights or an EMA copy, student dropout on or off) in
test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_teacher_consistency.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, FAMO, BalancedMultiTaskModel, MultiTaskModel, PooledOutputLossTrainer,
    TeacherConsistencyModel,
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


def config():
    return ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS["small-bert"],
    )


def make_model(batch, teacher="ema", student_dropout=True, cons_w=1.0, decay=0.99,
               keep=0.0, encoder_dropout=True, cls=TeacherConsistencyModel, attach=True):
    """A built model with the encoder's own dropout left on (as in training)
    unless ``encoder_dropout`` is False, so steps can be compared exactly."""
    model = cls(config(), dropout_rate=0.0)
    model(batch[0], training=False)
    if not encoder_dropout:
        for layer in model.encoder.submodules:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.rate = 0.0
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                      pooled_class_weights={0: 1.0, 1: 1.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 1.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    model._balancer = FAMO(NUM_LOSSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=model._mtl_loss_fn)
    if attach:
        ema = None
        if teacher == "ema":
            ema = MultiTaskModel(config(), dropout_rate=0.0)
            ema(batch[0], training=False)
        model.attach_teacher(ema, student_dropout, cons_w, 0, keep, ema_decay=decay)
    return model


def weights(variables):
    return [v.numpy().copy() for v in variables]


def test_teacher_starts_as_a_copy_and_stays_out_of_the_student(batch):
    model = make_model(batch)
    teacher = model.teacher
    for t, s in zip(teacher.trainable_variables, model.trainable_variables):
        np.testing.assert_array_equal(t.numpy(), s.numpy())
    student_ids = {id(v) for v in model.trainable_variables + model.weights}
    assert not student_ids & {id(v) for v in teacher.weights}
    x, _ = batch
    a, b = model(x, training=False), teacher(x, training=False)
    for k in NAMES:
        np.testing.assert_allclose(a[k].numpy(), b[k].numpy(), atol=1e-6)


def test_ema_update_moves_teacher_by_the_decay(batch):
    model = make_model(batch, decay=0.99)
    model.train_step(batch)  # first step: decay min(0.99, 1 - 1/2) = 0.5
    t0, s0 = weights(model.teacher.trainable_variables), weights(model.trainable_variables)
    for _ in range(3):
        model.train_step(batch)
    # after step n the decay is min(0.99, 1 - 1/(n + 1))
    assert int(model.optimizer.iterations) == 4
    t_before = weights(model.teacher.trainable_variables)
    s_before = weights(model.trainable_variables)
    model.train_step(batch)
    s_after = weights(model.trainable_variables)
    decay = min(0.99, 1 - 1 / 6)
    for t, b, a in zip(model.teacher.trainable_variables, t_before, s_after):
        np.testing.assert_allclose(t.numpy(), decay * b + (1 - decay) * a, rtol=1e-5, atol=1e-6)
    assert any(not np.allclose(a, b) for a, b in zip(t0, t_before))  # the teacher does move
    assert any(not np.allclose(a, b) for a, b in zip(s_before, s_after))


def test_teacher_gets_no_gradient_and_is_not_trained_by_the_optimizer(batch):
    model = make_model(batch, decay=0.0)  # decay 0: teacher = student after each step
    model.train_step(batch)
    for t, s in zip(model.teacher.trainable_variables, model.trainable_variables):
        np.testing.assert_allclose(t.numpy(), s.numpy(), rtol=1e-6)
    opt_vars = model.optimizer.variables
    opt_ids = {id(v) for v in (opt_vars() if callable(opt_vars) else opt_vars)}
    assert not opt_ids & {id(v) for v in model.teacher.weights}


def test_current_teacher_without_student_dropout_is_refused(batch):
    with pytest.raises(ValueError, match="needs student dropout"):
        make_model(batch, teacher="current", student_dropout=False)


def test_current_teacher_matches_dropout_off_pass_so_the_loss_is_zero(batch):
    model = make_model(batch, teacher="current")
    x, y = batch
    target, student = model.teacher_outputs(x), model(x, training=False)
    loss, kept = model.consistency_loss(target, student, y)
    assert float(loss) < 1e-9 and float(kept) == 1.0
    assert float(model.teacher_gap(target, student)) < 1e-7


def test_dropout_off_pass_is_fixed_and_dropout_on_pass_is_not(batch):
    model = make_model(batch, teacher="current")
    x, _ = batch
    a, b = model(x, training=False), model(x, training=False)
    np.testing.assert_array_equal(a["d1"].numpy(), b["d1"].numpy())
    tf.random.set_seed(3)
    c, d = model(x, training=True), model(x, training=True)
    assert not np.allclose(c["d1"].numpy(), d["d1"].numpy())


@pytest.mark.parametrize("teacher,student_dropout", [("current", True), ("ema", True), ("ema", False)])
def test_weight_zero_matches_famo(batch, teacher, student_dropout):
    model = make_model(batch, teacher, student_dropout, cons_w=0.0, encoder_dropout=False)
    plain = make_model(batch, cls=BalancedMultiTaskModel, encoder_dropout=False, attach=False)
    for v, w in zip(plain.trainable_variables, weights(model.trainable_variables)):
        v.assign(w)
    model.train_step(batch)
    plain.train_step(batch)
    for v, a, b in zip(model.trainable_variables, weights(model.trainable_variables),
                       weights(plain.trainable_variables)):
        np.testing.assert_allclose(a, b, atol=1e-6, err_msg=v.name)


def test_training_logs_and_teacher_gap(batch):
    model = make_model(batch, keep=0.1)
    x, y = batch
    hist = model.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(16), epochs=2,
                     validation_data=tf.data.Dataset.from_tensor_slices((x, y)).batch(16),
                     verbose=0).history
    for k in ["loss", "consistency_loss", "consistency_weight", "consistency_kept",
              "teacher_gap", "ema_decay", "val_loss", "val_consistency_loss", "val_teacher_gap"]:
        assert k in hist and len(hist[k]) == 2 and all(np.isfinite(hist[k])), (k, hist)
    assert all(0 < v < 0.5 for v in hist["consistency_kept"])
    assert hist["teacher_gap"][1] > 0  # the EMA teacher lags the student
    # steps 1-2 then 3-4: decay min(0.99, 1 - 1/(n + 1))
    np.testing.assert_allclose(hist["ema_decay"], [(1 / 2 + 2 / 3) / 2, (3 / 4 + 4 / 5) / 2],
                               rtol=1e-5)
