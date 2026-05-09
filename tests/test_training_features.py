"""
Tests for training configuration features not covered by existing tests.

Covers:
- LR schedules: exponential_decay, constant (and unknown schedule error)
- Early stopping: stops before max epochs with tight patience
- Dropout: behavior in training vs inference mode
- Checkpointing via checkpoint_dir argument to fit()
- Verbose modes (0, 1, 2): must not crash
- trainer.save_checkpoint() / Trainer.load_checkpoint()
- Prediction consistency before and after save/load
- model.validate_dataset()
"""

import os
import tempfile
import pytest
import numpy as np
import tensorflow as tf

from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer
from multitask.models.persistence import load_model

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EMB_DIM = 32
N = 64
BATCH = 16


def _binary_model():
    config = ModelConfig(
        task_structure=[[2]],
        task_names=["a", "b"],
        task_types=["binary", "binary"],
        shared_layer_sizes=[32],
        default_layer_sizes=[16],
    )
    return MultiTaskModel(config=config, embedding_dim=EMB_DIM)


def _make_dataset(n=N, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, EMB_DIM).astype(np.float32)
    y = {
        "a": rng.randint(0, 2, n).astype(np.float32),
        "b": rng.randint(0, 2, n).astype(np.float32),
    }
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH)


def _fit(model, ds, epochs=2, **kwargs):
    cfg = TrainingConfig(
        epochs=epochs,
        early_stopping=False,
        use_lr_schedule=False,
        use_warmup=False,
    )
    trainer = Trainer(model=model, config=cfg)
    return trainer, *trainer.fit(ds, ds, num_tasks=2, verbose=0, **kwargs)


# ---------------------------------------------------------------------------
# LR schedule tests
# ---------------------------------------------------------------------------

class TestLRSchedules:
    def test_exponential_decay_builds_schedule_object(self):
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=2,
            use_lr_schedule=True,
            lr_schedule_type="exponential_decay",
            use_warmup=False,
            early_stopping=False,
        )
        trainer = Trainer(model=model, config=cfg)
        opt = trainer._build_optimizer(steps_per_epoch=4)
        lr_obj = getattr(opt, "_learning_rate", opt.learning_rate)
        assert isinstance(lr_obj, tf.keras.optimizers.schedules.LearningRateSchedule)

    def test_exponential_decay_training_runs(self):
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=2,
            use_lr_schedule=True,
            lr_schedule_type="exponential_decay",
            use_warmup=False,
            early_stopping=False,
        )
        trainer = Trainer(model=model, config=cfg)
        ds = _make_dataset()
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_constant_schedule_returns_float_lr(self):
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=2,
            use_lr_schedule=True,
            lr_schedule_type="constant",
            use_warmup=False,
            early_stopping=False,
        )
        trainer = Trainer(model=model, config=cfg)
        opt = trainer._build_optimizer(steps_per_epoch=4)
        lr_obj = getattr(opt, "_learning_rate", opt.learning_rate)
        # constant returns the bare float, not a schedule object
        assert isinstance(lr_obj, (float, int, np.floating))

    def test_constant_schedule_training_runs(self):
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=2,
            use_lr_schedule=True,
            lr_schedule_type="constant",
            use_warmup=False,
            early_stopping=False,
        )
        trainer = Trainer(model=model, config=cfg)
        ds = _make_dataset()
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_unknown_schedule_raises_value_error(self):
        model = _binary_model()
        cfg = TrainingConfig(epochs=2, early_stopping=False)
        cfg.use_lr_schedule = True
        cfg.lr_schedule_type = "hyperbolic_decay"
        cfg.use_warmup = False
        trainer = Trainer(model=model, config=cfg)
        with pytest.raises(ValueError, match="Unknown lr_schedule_type"):
            trainer._build_lr_schedule(steps_per_epoch=4)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stops_before_max_epochs_with_tight_patience(self):
        """patience=1 on noisy random data should stop well before 30 epochs."""
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=30,
            early_stopping=True,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            use_lr_schedule=False,
            use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        ds = _make_dataset()
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert len(history.history["loss"]) < 30

    def test_disabled_early_stopping_runs_all_epochs(self):
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=3,
            early_stopping=False,
            use_lr_schedule=False,
            use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        ds = _make_dataset()
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert len(history.history["loss"]) == 3

    def test_early_stopping_returns_finite_loss(self):
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=20,
            early_stopping=True,
            early_stopping_patience=2,
            use_lr_schedule=False,
            use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        ds = _make_dataset()
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])


# ---------------------------------------------------------------------------
# Dropout behavior
# ---------------------------------------------------------------------------

class TestDropoutBehavior:
    def test_high_dropout_model_builds_and_trains(self):
        config = ModelConfig(
            task_structure=[[2]],
            task_names=["a", "b"],
            task_types=["binary", "binary"],
            shared_layer_sizes=[32],
            default_layer_sizes=[16],
        )
        model = MultiTaskModel(config=config, embedding_dim=EMB_DIM, dropout_rate=0.5)
        cfg = TrainingConfig(
            epochs=2, early_stopping=False,
            use_lr_schedule=False, use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        ds = _make_dataset()
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_high_dropout_stochastic_in_training_mode(self):
        """With 90% dropout, two forward passes in training mode should differ."""
        config = ModelConfig(
            task_structure=[[1]],
            task_names=["a"],
            task_types=["binary"],
            shared_layer_sizes=[64],
            default_layer_sizes=[32],
        )
        model = MultiTaskModel(config=config, embedding_dim=EMB_DIM, dropout_rate=0.9)
        x = tf.random.normal((32, EMB_DIM), seed=0)
        out1 = model(x, training=True)["a"].numpy()
        out2 = model(x, training=True)["a"].numpy()
        # With 90% dropout two stochastic passes must differ
        assert not np.allclose(out1, out2), (
            "Expected stochastic difference with 90% dropout in training mode"
        )

    def test_inference_mode_is_deterministic(self):
        """Two forward passes with training=False must produce identical outputs."""
        config = ModelConfig(
            task_structure=[[1]],
            task_names=["a"],
            task_types=["binary"],
            shared_layer_sizes=[32],
            default_layer_sizes=[16],
        )
        model = MultiTaskModel(config=config, embedding_dim=EMB_DIM, dropout_rate=0.5)
        x = tf.random.normal((8, EMB_DIM), seed=1)
        out1 = model(x, training=False)["a"].numpy()
        out2 = model(x, training=False)["a"].numpy()
        np.testing.assert_array_equal(out1, out2)

    def test_zero_dropout_valid(self):
        config = ModelConfig(
            task_structure=[[1]],
            task_names=["a"],
            task_types=["binary"],
        )
        model = MultiTaskModel(config=config, embedding_dim=EMB_DIM, dropout_rate=0.0)
        x = tf.random.normal((4, EMB_DIM))
        out = model(x, training=False)
        assert out["a"].shape == (4, 1)


# ---------------------------------------------------------------------------
# Checkpointing via checkpoint_dir
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_checkpoint_directory_created(self):
        model = _binary_model()
        ds = _make_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ckpt")
            trainer, history, _ = _fit(model, ds, epochs=2, checkpoint_dir=ckpt)
            assert os.path.isdir(ckpt), "Checkpoint directory not created"
            assert len(os.listdir(ckpt)) > 0, "No files written to checkpoint directory"

    def test_checkpoint_contains_config_file(self):
        model = _binary_model()
        ds = _make_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ckpt")
            _fit(model, ds, epochs=2, checkpoint_dir=ckpt)
            assert os.path.exists(os.path.join(ckpt, "config.json"))

    def test_checkpoint_loadable_via_load_model(self):
        """A checkpoint saved by fit() can be loaded with load_model()."""
        model = _binary_model()
        ds = _make_dataset()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ckpt")
            _fit(model, ds, epochs=2, checkpoint_dir=ckpt)
            loaded_model, _ = load_model(ckpt)
            assert loaded_model is not None
            # Loaded model produces same-shaped output
            x = tf.random.normal((4, EMB_DIM))
            out = loaded_model(x, training=False)
            assert "a" in out and "b" in out


# ---------------------------------------------------------------------------
# Verbose modes
# ---------------------------------------------------------------------------

class TestVerboseModes:
    def _run(self, verbose):
        model = _binary_model()
        cfg = TrainingConfig(
            epochs=1, early_stopping=False,
            use_lr_schedule=False, use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        ds = _make_dataset()
        history, _ = trainer.fit(ds, ds, num_tasks=2, verbose=verbose)
        assert history is not None

    def test_verbose_0_silent(self):
        self._run(0)

    def test_verbose_1_progress_bar(self):
        self._run(1)

    def test_verbose_2_one_line_per_epoch(self):
        self._run(2)


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint (Trainer-level)
# ---------------------------------------------------------------------------

class TestSaveLoadCheckpoint:
    def test_save_writes_training_config_json(self):
        model = _binary_model()
        ds = _make_dataset()
        cfg = TrainingConfig(
            epochs=2, learning_rate=5e-4,
            early_stopping=False, use_lr_schedule=False, use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        _, thresholds = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt")
            trainer.save_checkpoint(path, thresholds=thresholds)
            assert os.path.exists(os.path.join(path, "training_config.json"))

    def test_load_checkpoint_returns_trainer_and_thresholds(self):
        model = _binary_model()
        ds = _make_dataset()
        cfg = TrainingConfig(
            epochs=2, early_stopping=False,
            use_lr_schedule=False, use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        _, thresholds = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt")
            trainer.save_checkpoint(path, thresholds=thresholds)
            loaded_trainer, loaded_thresholds = Trainer.load_checkpoint(path)
        assert loaded_trainer is not None
        assert loaded_trainer.model is not None
        assert loaded_thresholds is not None
        assert set(loaded_thresholds.keys()) == set(thresholds.keys())

    def test_load_checkpoint_restores_learning_rate(self):
        """TrainingConfig.learning_rate must survive the save/load round-trip."""
        model = _binary_model()
        ds = _make_dataset()
        lr = 7e-4
        cfg = TrainingConfig(
            epochs=2, learning_rate=lr,
            early_stopping=False, use_lr_schedule=False, use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        trainer.fit(ds, ds, num_tasks=2, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt")
            trainer.save_checkpoint(path)
            loaded_trainer, _ = Trainer.load_checkpoint(path)
        assert loaded_trainer.config.learning_rate == pytest.approx(lr)

    def test_save_checkpoint_without_prior_fit_uses_default_thresholds(self):
        """save_checkpoint before fit() should not crash (no thresholds available)."""
        model = _binary_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt")
            # Build the model with a dummy forward pass so weights exist
            x = tf.random.normal((4, EMB_DIM))
            model(x, training=False)
            cfg = TrainingConfig(epochs=1, early_stopping=False)
            trainer = Trainer(model=model, config=cfg)
            trainer.save_checkpoint(path)  # should not raise


# ---------------------------------------------------------------------------
# Prediction consistency before and after save/load
# ---------------------------------------------------------------------------

class TestPredictionConsistency:
    def test_model_save_load_identical_predictions(self):
        """model.save() then load_model() must produce bit-identical predictions."""
        model = _binary_model()
        ds = _make_dataset()
        trainer, history, thresholds = _fit(model, ds, epochs=2)
        x_test = tf.random.normal((8, EMB_DIM), seed=99)
        preds_before = {k: v.numpy() for k, v in model(x_test, training=False).items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model")
            model.save(path, thresholds=thresholds)
            loaded_model, loaded_thresh = load_model(path)
        preds_after = {k: v.numpy() for k, v in loaded_model(x_test, training=False).items()}
        for task in preds_before:
            np.testing.assert_allclose(
                preds_before[task], preds_after[task], atol=1e-5,
                err_msg=f"Predictions differ for task '{task}' after save/load",
            )

    def test_checkpoint_save_load_identical_predictions(self):
        """trainer.save_checkpoint() then load_checkpoint() must preserve predictions."""
        model = _binary_model()
        ds = _make_dataset()
        cfg = TrainingConfig(
            epochs=2, early_stopping=False,
            use_lr_schedule=False, use_warmup=False,
        )
        trainer = Trainer(model=model, config=cfg)
        _, thresholds = trainer.fit(ds, ds, num_tasks=2, verbose=0)
        x_test = tf.random.normal((8, EMB_DIM), seed=77)
        preds_before = {k: v.numpy() for k, v in model(x_test, training=False).items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt")
            trainer.save_checkpoint(path, thresholds=thresholds)
            loaded_trainer, _ = Trainer.load_checkpoint(path)
        preds_after = {
            k: v.numpy()
            for k, v in loaded_trainer.model(x_test, training=False).items()
        }
        for task in preds_before:
            np.testing.assert_allclose(
                preds_before[task], preds_after[task], atol=1e-5,
                err_msg=f"Predictions differ for task '{task}' after checkpoint save/load",
            )

    def test_thresholds_round_trip(self):
        """Thresholds saved with model.save() are correctly restored by load_model()."""
        model = _binary_model()
        ds = _make_dataset()
        trainer, _, thresholds = _fit(model, ds, epochs=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model")
            model.save(path, thresholds=thresholds)
            _, loaded_thresh = load_model(path)
        assert loaded_thresh is not None
        for task, val in thresholds.items():
            if val is None:
                assert loaded_thresh[task] is None
            else:
                assert loaded_thresh[task] == pytest.approx(val, abs=1e-6)


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

class TestDatasetValidation:
    def _good_dataset(self):
        x = np.random.randn(16, EMB_DIM).astype(np.float32)
        y = {
            "a": np.random.randint(0, 2, 16).astype(np.float32),
            "b": np.random.randint(0, 2, 16).astype(np.float32),
        }
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(8)

    def test_valid_dataset_returns_true(self):
        model = _binary_model()
        ds = self._good_dataset()
        result = model.validate_dataset(ds, verbose=False)
        assert result is True

    def test_wrong_embedding_dim_raises(self):
        model = _binary_model()
        x = np.random.randn(16, EMB_DIM + 10).astype(np.float32)
        y = {
            "a": np.random.randint(0, 2, 16).astype(np.float32),
            "b": np.random.randint(0, 2, 16).astype(np.float32),
        }
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
        with pytest.raises(ValueError, match="[Ee]mbedding"):
            model.validate_dataset(ds, verbose=False)

    def test_wrong_task_name_in_labels_raises(self):
        model = _binary_model()
        x = np.random.randn(16, EMB_DIM).astype(np.float32)
        y = {
            "wrong_name": np.random.randint(0, 2, 16).astype(np.float32),
            "b": np.random.randint(0, 2, 16).astype(np.float32),
        }
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
        with pytest.raises(ValueError, match="[Tt]ask name"):
            model.validate_dataset(ds, verbose=False)

    def test_invalid_binary_label_values_raises(self):
        """Binary task labels with value 5 (not 0/1/-1) must be rejected."""
        model = _binary_model()
        x = np.random.randn(16, EMB_DIM).astype(np.float32)
        ya = np.random.randint(0, 2, 16).astype(np.float32)
        ya[0] = 5.0  # invalid
        y = {"a": ya, "b": np.random.randint(0, 2, 16).astype(np.float32)}
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            model.validate_dataset(ds, verbose=False)

    def test_string_input_to_embedding_model_raises(self):
        """Models without an encoder expect floating point embedding matrices, not raw strings."""
        model = _binary_model()
        texts = tf.constant(["hello", "world"], dtype=tf.string)
        y = {
            "a": np.array([0.0, 1.0], dtype=np.float32),
            "b": np.array([1.0, 0.0], dtype=np.float32),
        }
        ds = tf.data.Dataset.from_tensor_slices((texts, y)).batch(2)
        with pytest.raises(ValueError, match="2D"):
            model.validate_dataset(ds, verbose=False)

    def test_verbose_false_suppresses_output(self, capsys):
        model = _binary_model()
        model.validate_dataset(self._good_dataset(), verbose=False)
        captured = capsys.readouterr()
        assert "Validating" not in captured.out

    def test_verbose_true_prints_output(self, capsys):
        model = _binary_model()
        model.validate_dataset(self._good_dataset(), verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
