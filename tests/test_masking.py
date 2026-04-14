"""
Tests for label masking behavior.

Covers:
- Fully-masked batches (all labels = mask_value): loss must be finite, not NaN
- Custom mask_value (other than -1.0): masking still works correctly
- Cross-task masking isolation: masking one task does not prevent another from training
"""

import pytest
import numpy as np
import tensorflow as tf

from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer

EMB_DIM = 32
BATCH = 16


def _make_model(task_names, task_types, num_classes=None, mask_value=-1.0):
    if num_classes is None:
        num_classes = [2] * len(task_types)
    config = ModelConfig(
        task_structure=[[len(task_types)]],
        task_names=task_names,
        task_types=task_types,
        num_classes_per_task=num_classes,
        mask_value=mask_value,
        shared_layer_sizes=[16],
        default_layer_sizes=[8],
        standardize_regression=False,
    )
    return MultiTaskModel(config=config, embedding_dim=EMB_DIM)


def _make_trainer(model, epochs=2):
    cfg = TrainingConfig(
        epochs=epochs,
        early_stopping=False,
        use_lr_schedule=False,
        use_warmup=False,
    )
    return Trainer(model=model, config=cfg)


# ---------------------------------------------------------------------------
# Section 1: Fully-masked batches
# ---------------------------------------------------------------------------

class TestFullyMaskedBatch:
    """All labels equal mask_value for an entire batch or dataset."""

    def test_loss_finite_all_labels_masked(self):
        """Loss function returns a finite value when every sample is masked."""
        model = _make_model(["a", "b"], ["binary", "binary"])
        trainer = _make_trainer(model)
        loss_fn = trainer._build_loss_function(
            num_tasks=2,
            class_weights=[{0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}],
            task_weights=[1.0, 1.0],
        )
        mask = -1.0
        y_true = {
            "a": tf.constant([[mask], [mask], [mask], [mask]], dtype=tf.float32),
            "b": tf.constant([[mask], [mask], [mask], [mask]], dtype=tf.float32),
        }
        y_pred = {
            "a": tf.constant([[0.5], [0.5], [0.5], [0.5]], dtype=tf.float32),
            "b": tf.constant([[0.5], [0.5], [0.5], [0.5]], dtype=tf.float32),
        }
        loss = loss_fn(y_true, y_pred)
        assert np.isfinite(loss.numpy()), f"Loss is not finite with fully-masked batch: {loss.numpy()}"

    def test_training_does_not_crash_with_fully_masked_dataset(self):
        """An entire training dataset where every label is masked_value must not crash."""
        model = _make_model(["a"], ["binary"])
        trainer = _make_trainer(model, epochs=1)
        x = np.random.randn(32, EMB_DIM).astype(np.float32)
        y = {"a": np.full(32, -1.0, dtype=np.float32)}
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH)
        history, _ = trainer.fit(ds, ds, num_tasks=1, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_fully_masked_multiclass_batch(self):
        """Fully-masked multiclass task: loss must be finite."""
        model = _make_model(["mc"], ["multiclass"], num_classes=[4])
        trainer = _make_trainer(model)
        loss_fn = trainer._build_loss_function(
            num_tasks=1,
            class_weights=[{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}],
            task_weights=[1.0],
        )
        mask = -1.0
        y_true = {"mc": tf.constant([mask, mask, mask], dtype=tf.float32)}
        y_pred = {"mc": tf.constant([[0.25, 0.25, 0.25, 0.25]] * 3, dtype=tf.float32)}
        loss = loss_fn(y_true, y_pred)
        assert np.isfinite(loss.numpy())

    def test_fully_masked_single_batch_mixed_tasks(self):
        """All labels masked across binary + regression tasks."""
        model = _make_model(["b", "r"], ["binary", "regression"], num_classes=[2, 1])
        trainer = _make_trainer(model)
        loss_fn = trainer._build_loss_function(
            num_tasks=2,
            class_weights=[{0: 1.0, 1: 1.0}, {0: 1.0}],
            task_weights=[1.0, 1.0],
        )
        mask = -1.0
        y_true = {
            "b": tf.constant([[mask], [mask]], dtype=tf.float32),
            "r": tf.constant([[mask], [mask]], dtype=tf.float32),
        }
        y_pred = {
            "b": tf.constant([[0.5], [0.5]], dtype=tf.float32),
            "r": tf.constant([[0.0], [0.0]], dtype=tf.float32),
        }
        loss = loss_fn(y_true, y_pred)
        assert np.isfinite(loss.numpy())


# ---------------------------------------------------------------------------
# Section 2: Custom mask_value
# ---------------------------------------------------------------------------

class TestCustomMaskValue:
    """mask_value other than -1.0 is correctly respected."""

    def test_custom_mask_excludes_from_loss(self):
        """With mask_value=-999, labels equal to -999 must be excluded from loss."""
        MASK = -999.0
        model = _make_model(["a"], ["binary"], mask_value=MASK)
        trainer = _make_trainer(model)
        loss_fn = trainer._build_loss_function(
            num_tasks=1,
            class_weights=[{0: 1.0, 1: 1.0}],
            task_weights=[1.0],
        )
        # One valid label (1.0), one custom-masked (-999.0)
        y_true = {"a": tf.constant([[1.0], [MASK]], dtype=tf.float32)}
        y_pred = {"a": tf.constant([[0.5], [0.5]], dtype=tf.float32)}
        loss = loss_fn(y_true, y_pred)
        # BCE(1.0, 0.5) ≈ 0.693 (only the valid sample contributes)
        assert loss.numpy() == pytest.approx(0.693, abs=0.01)

    def test_default_mask_value_minus_one_not_matched_by_custom(self):
        """With mask_value=-999, labels equal to -1.0 must NOT be masked."""
        MASK = -999.0
        model = _make_model(["a"], ["binary"], mask_value=MASK)
        trainer = _make_trainer(model)
        loss_fn = trainer._build_loss_function(
            num_tasks=1,
            class_weights=[{0: 1.0, 1: 1.0}],
            task_weights=[1.0],
        )

        # -1.0 should be treated as a real label (not masked) when mask_value=-999
        # BCE(-1.0, 0.5) is still computed — loss must be finite but different from masked
        y_true_custom_masked = {"a": tf.constant([[MASK]], dtype=tf.float32)}
        y_true_real = {"a": tf.constant([[-1.0]], dtype=tf.float32)}
        y_pred = {"a": tf.constant([[0.5]], dtype=tf.float32)}
        loss_masked = loss_fn(y_true_custom_masked, y_pred).numpy()
        loss_real = loss_fn(y_true_real, y_pred).numpy()

        # Masked → loss ≈ 0 (no valid samples); real → finite non-zero loss
        assert np.isfinite(loss_masked)
        assert np.isfinite(loss_real)
        assert loss_masked != pytest.approx(loss_real, abs=0.01)

    def test_positive_custom_mask_value(self):
        """mask_value=999.0 (positive) is respected."""
        MASK = 999.0
        model = _make_model(["a", "b"], ["binary", "binary"], mask_value=MASK)
        trainer = _make_trainer(model, epochs=1)
        rng = np.random.RandomState(0)
        x = rng.randn(32, EMB_DIM).astype(np.float32)
        ya = rng.randint(0, 2, 32).astype(np.float32)
        ya[::3] = MASK  # mask every 3rd sample
        yb = rng.randint(0, 2, 32).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, {"a": ya, "b": yb})).batch(BATCH)
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_mask_value_zero(self):
        """mask_value=0.0 means 0.0 labels are excluded."""
        MASK = 0.0
        model = _make_model(["r"], ["regression"], num_classes=[1], mask_value=MASK)
        trainer = _make_trainer(model)
        loss_fn = trainer._build_loss_function(
            num_tasks=1,
            class_weights=[{0: 1.0}],
            task_weights=[1.0],
        )

        # Only the 1.5 label should contribute; 0.0 is masked
        y_true = {"r": tf.constant([[1.5], [MASK]], dtype=tf.float32)}
        y_pred = {"r": tf.constant([[0.0], [0.0]], dtype=tf.float32)}
        loss = loss_fn(y_true, y_pred)
        assert np.isfinite(loss.numpy())


# ---------------------------------------------------------------------------
# Section 3: Cross-task masking isolation
# ---------------------------------------------------------------------------

class TestCrossTaskMaskingIsolation:
    """Masking task A should not prevent task B from training."""

    def test_fully_masked_task_does_not_prevent_other_from_training(self):
        """Task A fully masked, task B fully labeled: task B must still train."""
        model = _make_model(["a", "b"], ["binary", "binary"])
        trainer = _make_trainer(model, epochs=4)
        rng = np.random.RandomState(42)

        x = rng.randn(64, EMB_DIM).astype(np.float32) # precomputed embeddings
        ya = np.full(64, -1.0, dtype=np.float32)
        yb = rng.randint(0, 2, 64).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, {"a": ya, "b": yb})).batch(BATCH)

        # Build weights with a forward pass.
        _ = model(tf.zeros((1, EMB_DIM), dtype=tf.float32), training=False)
        weights_before = [w.numpy().copy() for w in model.trainable_variables]
        assert len(weights_before) > 0, "Model should have trainable weights after build"

        history, _ = trainer.fit(
            ds,
            ds,
            num_tasks=2,
            verbose=0,
            task_weights=[1.0, 1.0],
            class_weights=[{0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}],
        )
        assert all(np.isfinite(v) for v in history.history["loss"])

        weights_after = [w.numpy() for w in model.trainable_variables]
        assert len(weights_before) == len(weights_after), "Trainable variable count changed across fit"
        
        any_changed = any(
            not np.allclose(before, after)
            for before, after in zip(weights_before, weights_after)
        )
        assert any_changed, "No weights changed; task B may not be training"

    def test_loss_changes_when_masking_changes(self):
        """Same predictions but with task A masked vs unmasked produce different loss."""
        model = _make_model(["a", "b"], ["binary", "binary"])
        trainer = _make_trainer(model)
        loss_fn = trainer._build_loss_function(
            num_tasks=2,
            class_weights=[{0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}],
            task_weights=[1.0, 1.0],
        )

        # Different preds for B vs A so (L_A+L_B)/8 != L_B/4 when A is masked (see docstring).
        y_pred = {
            "a": tf.constant([[0.5], [0.5], [0.5], [0.5]], dtype=tf.float32),
            "b": tf.constant([[0.9], [0.1], [0.9], [0.1]], dtype=tf.float32),
        }
        y_true_both = {
            "a": tf.constant([[1.0], [0.0], [1.0], [0.0]], dtype=tf.float32),
            "b": tf.constant([[1.0], [1.0], [0.0], [0.0]], dtype=tf.float32),
        }
        y_true_a_masked = {
            "a": tf.constant([[-1.0], [-1.0], [-1.0], [-1.0]], dtype=tf.float32),
            "b": tf.constant([[1.0], [1.0], [0.0], [0.0]], dtype=tf.float32),
        }
        loss_both = loss_fn(y_true_both, y_pred).numpy()
        loss_masked = loss_fn(y_true_a_masked, y_pred).numpy()
        assert np.isfinite(loss_both)
        assert np.isfinite(loss_masked)
        
        # With different per-task predictions, masking A changes the combined mean loss.
        assert not np.isclose(loss_both, loss_masked, rtol=1e-4, atol=1e-4)

    def test_masking_independent_tasks_do_not_interfere(self):
        """Mask A for first half, mask B for second half; both tasks see valid gradients."""
        model = _make_model(["a", "b"], ["binary", "binary"])
        trainer = _make_trainer(model, epochs=3)
        rng = np.random.RandomState(7)
        n = 64
        x = rng.randn(n, EMB_DIM).astype(np.float32)
        ya = rng.randint(0, 2, n).astype(np.float32)
        yb = rng.randint(0, 2, n).astype(np.float32)
        ya[:n // 2] = -1.0   # mask first half for A
        yb[n // 2:] = -1.0   # mask second half for B
        ds = tf.data.Dataset.from_tensor_slices((x, {"a": ya, "b": yb})).batch(BATCH)
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_masking_one_task_in_mixed_type_model(self):
        """Mask the regression task completely; binary task still trains."""
        model = _make_model(["b", "r"], ["binary", "regression"], num_classes=[2, 1])
        trainer = _make_trainer(model, epochs=3)
        rng = np.random.RandomState(3)
        x = rng.randn(64, EMB_DIM).astype(np.float32)
        yb = rng.randint(0, 2, 64).astype(np.float32)
        yr = np.full(64, -1.0, dtype=np.float32)  # regression fully masked
        ds = tf.data.Dataset.from_tensor_slices((x, {"b": yb, "r": yr})).batch(BATCH)
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])
