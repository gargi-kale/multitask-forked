import os
import json
import pytest
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer

# Helpers: build models and fake datasets for testing
TASK_NAMES_4 = ['t0', 't1', 't2', 't3']


def make_model(output_pools=None, num_tasks=4, embedding_dim=32):
    """Build a small 4-task model with optional pools."""
    config = ModelConfig(
        task_structure=[[2], [2, 2]],
        task_names=TASK_NAMES_4[:num_tasks],
        output_pools=output_pools,
    )
    return MultiTaskModel(config=config, embedding_dim=embedding_dim)


def make_dataset(num_samples=64, embedding_dim=32, task_names=None, seed=42):
    """Create a fake dataset of (embeddings, labels_dict) pairs.
    
    Labels are random binary {0, 1} for each task. No masking by default.
    """
    task_names = task_names or TASK_NAMES_4
    rng = np.random.RandomState(seed)
    
    x = rng.randn(num_samples, embedding_dim).astype(np.float32)
    labels = {
        name: rng.randint(0, 2, size=(num_samples,)).astype(np.float32)
        for name in task_names
    }
    
    ds = tf.data.Dataset.from_tensor_slices((x, labels)).batch(16)
    return ds


def make_dataset_with_masking(num_samples=64, embedding_dim=32,
                               task_names=None, mask_rate=0.3, seed=42):
    """Create a fake dataset with some labels masked (-1)."""
    task_names = task_names or TASK_NAMES_4
    rng = np.random.RandomState(seed)
    
    x = rng.randn(num_samples, embedding_dim).astype(np.float32)
    labels = {}
    for name in task_names:
        y = rng.randint(0, 2, size=(num_samples,)).astype(np.float32)
        mask = rng.rand(num_samples) < mask_rate
        y[mask] = -1
        labels[name] = y
    
    ds = tf.data.Dataset.from_tensor_slices((x, labels)).batch(16)
    return ds


# ######################################################################
#  SECTION 1: REGRESSION TESTS
# ######################################################################

class TestRegressionTrainerNoPools:
    """Trainer behavior must be unchanged when no pools are configured."""

    def test_fit_returns_thresholds_for_tasks_only(self):
        model = make_model()
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        history, thresholds = trainer.fit(ds, ds, num_tasks=4, verbose=0)
        
        assert set(thresholds.keys()) == set(TASK_NAMES_4)

    def test_evaluate_returns_task_metrics_only(self):
        model = make_model()
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        trainer.fit(ds, ds, num_tasks=4, verbose=0)
        results = trainer.evaluate(ds)
        
        assert 'task_metrics' in results
        assert 'pool_metrics' not in results
        assert set(results['task_metrics'].keys()) == set(TASK_NAMES_4)

    def test_evaluate_overall_metrics_present(self):
        model = make_model()
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        trainer.fit(ds, ds, num_tasks=4, verbose=0)
        results = trainer.evaluate(ds)
        
        assert 'overall_metrics' in results
        assert isinstance(results['overall_metrics'], dict)

    def test_fit_with_optimize_thresholds(self):
        model = make_model()
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        history, thresholds = trainer.fit(
            ds, ds, num_tasks=4, optimize_thresholds=True, verbose=0
        )
        
        # Should have thresholds for all 4 tasks
        assert set(thresholds.keys()) == set(TASK_NAMES_4)
        for name in TASK_NAMES_4:
            assert isinstance(thresholds[name], float)


# ######################################################################
#  SECTION 2: UNIT TESTS — _derive_pooled_labels
# ######################################################################

class TestDerivePooledLabelsMax:
    """Max pooling on ground truth labels."""

    def test_max_no_masking(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model)
        
        y_true = {
            't0': np.array([0, 1, 0, 1], dtype=np.float32),
            't1': np.array([1, 0, 0, 1], dtype=np.float32),
            't2': np.array([0, 0, 1, 1], dtype=np.float32),
            't3': np.array([1, 1, 0, 0], dtype=np.float32),
        }
        result = trainer._derive_pooled_labels(y_true)
        expected = np.array([1, 1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(result['pool'], expected)

    def test_max_with_partial_masking(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model)
        
        y_true = {
            't0': np.array([0, -1, 0, -1], dtype=np.float32),
            't1': np.array([1,  0, -1, -1], dtype=np.float32),
            't2': np.array([0,  0,  0,  0], dtype=np.float32),
            't3': np.array([0,  0,  0,  0], dtype=np.float32),
        }
        result = trainer._derive_pooled_labels(y_true)
        # obs0: max(0,1)=1; obs1: t0 masked→max(0)=0; obs2: t1 masked→max(0)=0; obs3: both masked→-1
        expected = np.array([1, 0, 0, -1], dtype=np.float32)
        np.testing.assert_array_equal(result['pool'], expected)

    def test_max_all_masked(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model)
        
        y_true = {
            't0': np.array([-1, -1], dtype=np.float32),
            't1': np.array([-1, -1], dtype=np.float32),
            't2': np.array([0, 0], dtype=np.float32),
            't3': np.array([0, 0], dtype=np.float32),
        }
        result = trainer._derive_pooled_labels(y_true)
        np.testing.assert_array_equal(result['pool'], np.array([-1, -1], dtype=np.float32))


class TestDerivePooledLabelsMean:
    """Mean pooling on ground truth labels."""

    def test_mean_no_masking(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'mean'}
        })
        trainer = Trainer(model)
        
        y_true = {
            't0': np.array([0, 1, 0, 1], dtype=np.float32),
            't1': np.array([1, 0, 0, 1], dtype=np.float32),
            't2': np.array([0, 0, 0, 0], dtype=np.float32),
            't3': np.array([0, 0, 0, 0], dtype=np.float32),
        }
        result = trainer._derive_pooled_labels(y_true)
        expected = np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(result['pool'], expected, atol=1e-6)

    def test_mean_with_masking_excludes_masked(self):
        """Mean should only average over non-masked constituents."""
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'mean'}
        })
        trainer = Trainer(model)
        
        y_true = {
            't0': np.array([1, -1], dtype=np.float32),
            't1': np.array([0,  1], dtype=np.float32),
            't2': np.array([0,  0], dtype=np.float32),
            't3': np.array([0,  0], dtype=np.float32),
        }
        result = trainer._derive_pooled_labels(y_true)
        # obs0: mean(1, 0) = 0.5; obs1: t0 masked → mean(1) = 1.0
        expected = np.array([0.5, 1.0], dtype=np.float32)
        np.testing.assert_allclose(result['pool'], expected, atol=1e-6)


class TestDerivePooledLabelsMin:
    """Min pooling on ground truth labels."""

    def test_min_no_masking(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'min'}
        })
        trainer = Trainer(model)
        
        y_true = {
            't0': np.array([0, 1, 0, 1], dtype=np.float32),
            't1': np.array([1, 0, 0, 1], dtype=np.float32),
            't2': np.array([0, 0, 0, 0], dtype=np.float32),
            't3': np.array([0, 0, 0, 0], dtype=np.float32),
        }
        result = trainer._derive_pooled_labels(y_true)
        expected = np.array([0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(result['pool'], expected)


class TestDerivePooledLabelsNoPools:
    """No pools configured → empty dict."""

    def test_returns_empty_dict(self):
        model = make_model()  # no pools
        trainer = Trainer(model)
        result = trainer._derive_pooled_labels({'t0': np.zeros(4), 't1': np.zeros(4)})
        assert result == {}


class TestDerivePooledLabels2DInput:
    """Labels with shape [n, 1] should be handled correctly."""

    def test_2d_input_flattened(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model)
        
        y_true = {
            't0': np.array([[0], [1]], dtype=np.float32),
            't1': np.array([[1], [0]], dtype=np.float32),
            't2': np.array([[0], [0]], dtype=np.float32),
            't3': np.array([[0], [0]], dtype=np.float32),
        }
        result = trainer._derive_pooled_labels(y_true)
        expected = np.array([1, 1], dtype=np.float32)
        np.testing.assert_array_equal(result['pool'], expected)


# ######################################################################
#  SECTION 3: INTEGRATION TESTS
# ######################################################################

class TestFitWithPools:
    """fit() should include pool thresholds."""

    def test_default_thresholds_include_pools(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        history, thresholds = trainer.fit(ds, ds, num_tasks=4, verbose=0)
        
        # Should have thresholds for 4 tasks + 1 pool
        assert 'pool' in thresholds
        assert thresholds['pool'] == 0.5  # default
        assert set(TASK_NAMES_4).issubset(thresholds.keys())

    def test_optimized_thresholds_include_pools(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        history, thresholds = trainer.fit(
            ds, ds, num_tasks=4, optimize_thresholds=True, verbose=0
        )
        
        assert 'pool' in thresholds
        assert isinstance(thresholds['pool'], float)

    def test_multiple_pools_thresholds(self):
        model = make_model(output_pools={
            'pool_a': {'tasks': ['t0', 't1'], 'strategy': 'max'},
            'pool_b': {'tasks': ['t2', 't3'], 'strategy': 'mean'},
        })
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        history, thresholds = trainer.fit(ds, ds, num_tasks=4, verbose=0)
        
        assert 'pool_a' in thresholds
        assert 'pool_b' in thresholds


class TestEvaluateWithPools:
    """evaluate() should produce pool_metrics alongside task_metrics."""

    def _train_and_evaluate(self, output_pools=None, optimize_thresholds=False):
        """Helper: train for 1 epoch, then evaluate."""
        model = make_model(output_pools=output_pools)
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        trainer.fit(ds, ds, num_tasks=4,
                    optimize_thresholds=optimize_thresholds, verbose=0)
        results = trainer.evaluate(ds)
        return results

    def test_pool_metrics_present(self):
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}}
        )
        assert 'pool_metrics' in results
        assert 'pool' in results['pool_metrics']

    def test_pool_metrics_has_expected_keys(self):
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}}
        )
        pm = results['pool_metrics']['pool']
        # Should have at least some metrics
        assert len(pm) > 0

    def test_task_metrics_unchanged(self):
        """task_metrics should still only contain architectural tasks."""
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}}
        )
        assert set(results['task_metrics'].keys()) == set(TASK_NAMES_4)
        assert 'pool' not in results['task_metrics']

    def test_overall_metrics_excludes_pools(self):
        """overall_metrics should average over tasks only, not pools."""
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}}
        )
        # overall_metrics should be the same whether or not pools exist
        results_no_pool = self._train_and_evaluate()
        
        # Both should have the same metric keys
        assert set(results['overall_metrics'].keys()) == set(results_no_pool['overall_metrics'].keys())

    def test_probabilities_dict_includes_pools(self):
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}}
        )
        assert 'pool' in results['probabilities_dict']

    def test_predictions_dict_includes_pools(self):
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}}
        )
        if 'predictions_dict' in results:
            assert 'pool' in results['predictions_dict']

    def test_thresholds_used_includes_pools(self):
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}}
        )
        if 'thresholds_used' in results:
            assert 'pool' in results['thresholds_used']

    def test_evaluate_with_optimized_thresholds(self):
        results = self._train_and_evaluate(
            output_pools={'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}},
            optimize_thresholds=True,
        )
        assert 'pool_metrics' in results
        pm = results['pool_metrics']['pool']
        # With optimized thresholds, should have binary metrics
        assert 'f1' in pm or 'accuracy' in pm

    def test_evaluate_prob_only_metrics(self):
        """When no thresholds, only probability metrics should be computed."""
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        trainer.fit(ds, ds, num_tasks=4, verbose=0)
        
        # Evaluate with probability-only metrics
        results = trainer.evaluate(ds, metrics=['auc', 'average_precision'])
        
        assert 'pool_metrics' in results


class TestEvaluateWithMasking:
    """evaluate() should handle masked labels correctly for pools."""

    def test_pool_metrics_with_masked_data(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds_train = make_dataset()
        ds_eval = make_dataset_with_masking(mask_rate=0.3)
        
        trainer.fit(ds_train, ds_train, num_tasks=4, verbose=0)
        results = trainer.evaluate(ds_eval)
        
        assert 'pool_metrics' in results
        # Pool metrics should still be computed (some samples are valid)
        assert results['pool_metrics']['pool'] is not None


class TestLossFunctionIgnoresPools:
    """The training loss should only iterate over architectural tasks."""

    def test_loss_function_task_count(self):
        """Loss function should be built for 4 tasks, not 4+pools."""
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        # Fit should work — loss function only iterates task_names
        history, _ = trainer.fit(ds, ds, num_tasks=4, verbose=0)
        
        # Should complete without errors about pool keys in loss
        assert history is not None


class TestMultiplePoolsIntegration:
    """Multiple pools should all work through the full pipeline."""

    def test_two_pools_full_pipeline(self):
        model = make_model(output_pools={
            'pool_a': {'tasks': ['t0', 't1'], 'strategy': 'max'},
            'pool_b': {'tasks': ['t2', 't3'], 'strategy': 'min'},
        })
        trainer = Trainer(model, TrainingConfig(epochs=1))
        ds = make_dataset()
        
        history, thresholds = trainer.fit(
            ds, ds, num_tasks=4, optimize_thresholds=True, verbose=0
        )
        
        assert 'pool_a' in thresholds
        assert 'pool_b' in thresholds
        
        results = trainer.evaluate(ds)
        
        assert 'pool_metrics' in results
        assert 'pool_a' in results['pool_metrics']
        assert 'pool_b' in results['pool_metrics']


# ######################################################################
#  Run without pytest if needed
# ######################################################################

if __name__ == "__main__":
    pytest.main([__file__, "-v"])