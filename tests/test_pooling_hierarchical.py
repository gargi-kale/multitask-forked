# hierarchical.py
import os
import json
import pytest
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from multitask.config import ModelConfig
from multitask.models.hierarchical import MultiTaskModel


# ======================================================================
# Helpers
# ======================================================================
TASK_NAMES_10 = [f"t{i}" for i in range(10)]

def make_model(output_pools=None, embedding_dim=64, **kwargs):
    """Build a model using pre-computed embeddings (no encoder needed)."""
    config = ModelConfig(
        task_structure=[[3], [2, 5, 3]],
        task_names=TASK_NAMES_10,
        output_pools=output_pools,
        **kwargs,
    )
    return MultiTaskModel(config=config, embedding_dim=embedding_dim)

def random_input(batch_size=4, dim=64):
    """Generate random embedding input."""
    return tf.random.normal([batch_size, dim])


# ######################################################################
#  SECTION 1: REGRESSION TESTS
# ######################################################################

class TestRegressionForwardPass:
    """Model without pools must behave exactly as before."""

    def test_output_is_dict(self):
        model = make_model()
        out = model(random_input(), training=False)
        assert isinstance(out, dict)

    def test_output_has_all_task_names(self):
        model = make_model()
        out = model(random_input(), training=False)
        assert set(out.keys()) == set(TASK_NAMES_10)

    def test_output_count_no_pools(self):
        model = make_model()
        out = model(random_input(), training=False)
        assert len(out) == 10

    def test_output_shapes_binary(self):
        model = make_model()
        out = model(random_input(batch_size=8), training=False)
        for name in TASK_NAMES_10:
            assert out[name].shape == (8, 1), f"{name} shape mismatch"

    def test_output_values_are_probabilities(self):
        model = make_model()
        out = model(random_input(), training=False)
        for name in TASK_NAMES_10:
            vals = out[name].numpy()
            assert np.all(vals >= 0) and np.all(vals <= 1), f"{name} not in [0,1]"

    def test_training_mode_runs(self):
        model = make_model()
        out = model(random_input(), training=True)
        assert len(out) == 10


class TestRegressionSummaryTree:
    """summary_tree should still work without pools."""

    def test_summary_tree_no_pools(self):
        model = make_model()
        tree = model.summary_tree()
        assert "Total Outputs: 10" in tree
        assert "Output Pools" not in tree


class TestRegressionGetTaskInfo:
    """get_task_info should still return 10 entries without pools."""

    def test_task_info_count(self):
        model = make_model()
        info = model.get_task_info()
        assert len(info) == 10

    def test_task_info_names(self):
        model = make_model()
        info = model.get_task_info()
        names = [entry['name'] for entry in info]
        assert names == TASK_NAMES_10


class TestRegressionFlattenUnflatten:
    """flatten/unflatten round-trip without pools."""

    def test_flatten_unflatten_round_trip(self):
        model = make_model()
        out = model(random_input(), training=False)
        flat, indices = model.flatten_outputs(out)
        restored = model.unflatten_outputs(flat)
        for name in TASK_NAMES_10:
            np.testing.assert_allclose(
                out[name].numpy(), restored[name].numpy(), atol=1e-6
            )


# ######################################################################
#  SECTION 2: UNIT TESTS — _compute_pooled_outputs
# ######################################################################

class TestPooledOutputsMax:
    """Max pooling takes element-wise max across constituent tasks."""

    def test_max_pool_two_tasks(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        out = model(random_input(), training=False)
        expected = np.maximum(out['t0'].numpy(), out['t1'].numpy())
        np.testing.assert_allclose(out['pool'].numpy(), expected, atol=1e-6)

    def test_max_pool_many_tasks(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1', 't2', 't3', 't4'], 'strategy': 'max'}
        })
        out = model(random_input(), training=False)
        constituent = np.stack([out[f't{i}'].numpy() for i in range(5)], axis=1)
        expected = constituent.max(axis=1)
        np.testing.assert_allclose(out['pool'].numpy(), expected, atol=1e-6)


class TestPooledOutputsMean:
    """Mean pooling averages across constituent tasks."""

    def test_mean_pool_two_tasks(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'mean'}
        })
        out = model(random_input(), training=False)
        expected = (out['t0'].numpy() + out['t1'].numpy()) / 2.0
        np.testing.assert_allclose(out['pool'].numpy(), expected, atol=1e-6)

    def test_mean_pool_three_tasks(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t2', 't3', 't4'], 'strategy': 'mean'}
        })
        out = model(random_input(), training=False)
        expected = (out['t2'].numpy() + out['t3'].numpy() + out['t4'].numpy()) / 3.0
        np.testing.assert_allclose(out['pool'].numpy(), expected, atol=1e-6)


class TestPooledOutputsMin:
    """Min pooling takes element-wise min across constituent tasks."""

    def test_min_pool_two_tasks(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'min'}
        })
        out = model(random_input(), training=False)
        expected = np.minimum(out['t0'].numpy(), out['t1'].numpy())
        np.testing.assert_allclose(out['pool'].numpy(), expected, atol=1e-6)


class TestPooledOutputsShape:
    """Pooled outputs should have the same shape as constituent tasks."""

    def test_pool_shape_matches_constituents(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1', 't2'], 'strategy': 'max'}
        })
        out = model(random_input(batch_size=16), training=False)
        assert out['pool'].shape == out['t0'].shape == (16, 1)

    def test_pool_values_are_probabilities(self):
        """Pooled output of sigmoid outputs should still be in [0, 1]."""
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        out = model(random_input(batch_size=32), training=False)
        vals = out['pool'].numpy()
        assert np.all(vals >= 0) and np.all(vals <= 1)


class TestMultiplePools:
    """Multiple pools operating on disjoint task subsets."""

    def test_two_pools_both_present(self):
        model = make_model(output_pools={
            'pool_a': {'tasks': ['t0', 't1'], 'strategy': 'max'},
            'pool_b': {'tasks': ['t8', 't9'], 'strategy': 'mean'},
        })
        out = model(random_input(), training=False)
        assert len(out) == 12  # 10 tasks + 2 pools
        assert 'pool_a' in out
        assert 'pool_b' in out

    def test_two_pools_independent(self):
        """Each pool should only depend on its own constituent tasks."""
        model = make_model(output_pools={
            'pool_a': {'tasks': ['t0', 't1'], 'strategy': 'max'},
            'pool_b': {'tasks': ['t8', 't9'], 'strategy': 'min'},
        })
        out = model(random_input(), training=False)
        
        expected_a = np.maximum(out['t0'].numpy(), out['t1'].numpy())
        expected_b = np.minimum(out['t8'].numpy(), out['t9'].numpy())
        np.testing.assert_allclose(out['pool_a'].numpy(), expected_a, atol=1e-6)
        np.testing.assert_allclose(out['pool_b'].numpy(), expected_b, atol=1e-6)


class TestPoolsTrainingVsInference:
    """Pools should be computed in both training and inference modes."""

    def test_pools_present_during_training(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        out = model(random_input(), training=True)
        assert 'pool' in out

    def test_pools_present_during_inference(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        out = model(random_input(), training=False)
        assert 'pool' in out


# ######################################################################
#  SECTION 3: INTEGRATION TESTS
# ######################################################################

class TestFlattenUnflattenWithPools:
    """flatten/unflatten should handle pooled outputs."""

    def test_flatten_excludes_pool_keys(self):
        """flatten_outputs concatenates only architectural tasks; pools are
        re-derived in unflatten_outputs, so including them would duplicate
        information and break the column alignment of the round-trip.
        """
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        out = model(random_input(), training=False)
        flat, indices = model.flatten_outputs(out)
        assert 'pool' not in indices
        assert set(indices.keys()) == set(TASK_NAMES_10)
        # Width matches sum of task output dims (10 binary tasks → 10 columns).
        assert flat.shape[-1] == 10

    def test_unflatten_recomputes_pools(self):
        """unflatten_outputs should recompute pooled values from task outputs."""
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        out = model(random_input(), training=False)
        
        # Flatten ONLY the task outputs (exclude pool), then unflatten
        task_only = {k: v for k, v in out.items() if k in TASK_NAMES_10}
        flat, _ = model.flatten_outputs(task_only)
        restored = model.unflatten_outputs(flat)
        
        # Pool should be recomputed
        assert 'pool' in restored
        np.testing.assert_allclose(
            out['pool'].numpy(), restored['pool'].numpy(), atol=1e-6
        )

    def test_full_round_trip_with_pools(self):
        """Round-trip: flatten task outputs → unflatten → pools recomputed."""
        model = make_model(output_pools={
            'pool_a': {'tasks': ['t0', 't1'], 'strategy': 'max'},
            'pool_b': {'tasks': ['t5', 't6', 't7'], 'strategy': 'mean'},
        })
        out = model(random_input(), training=False)
        
        # flatten/unflatten round-trip uses task-only outputs because
        # unflatten_outputs reads columns sequentially by task_names order.
        # Pool columns would shift the alignment if included.
        task_only = {k: v for k, v in out.items() if k in TASK_NAMES_10}
        flat, indices = model.flatten_outputs(task_only)
        restored = model.unflatten_outputs(flat)
        
        # All tasks should match exactly
        for name in TASK_NAMES_10:
            np.testing.assert_allclose(
                out[name].numpy(), restored[name].numpy(), atol=1e-6
            )
        # Pools should be recomputed and match
        np.testing.assert_allclose(
            out['pool_a'].numpy(), restored['pool_a'].numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            out['pool_b'].numpy(), restored['pool_b'].numpy(), atol=1e-6
        )


class TestSummaryTreeWithPools:
    """summary_tree should display pool information."""

    def test_summary_shows_pools(self):
        model = make_model(output_pools={
            'propaganda': {'tasks': ['t0', 't1', 't2'], 'strategy': 'max'},
        })
        tree = model.summary_tree()
        assert "Output Pools (1):" in tree
        assert "propaganda" in tree
        assert "max" in tree

    def test_summary_shows_multiple_pools(self):
        model = make_model(output_pools={
            'pool_a': {'tasks': ['t0', 't1'], 'strategy': 'max'},
            'pool_b': {'tasks': ['t8', 't9'], 'strategy': 'mean'},
        })
        tree = model.summary_tree()
        assert "Output Pools (2):" in tree
        assert "pool_a" in tree
        assert "pool_b" in tree


class TestGetPoolInfo:
    """get_pool_info should return structured pool metadata."""

    def test_no_pools_returns_empty(self):
        model = make_model()
        assert model.get_pool_info() == []

    def test_single_pool_info(self):
        model = make_model(output_pools={
            'agg': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        info = model.get_pool_info()
        assert len(info) == 1
        assert info[0]['name'] == 'agg'
        assert info[0]['strategy'] == 'max'
        assert info[0]['task_type'] == 'binary'
        assert info[0]['num_classes'] == 2
        assert info[0]['constituent_tasks'] == ['t0', 't1']

    def test_multiple_pools_info(self):
        model = make_model(output_pools={
            'pool_a': {'tasks': ['t0', 't1'], 'strategy': 'max'},
            'pool_b': {'tasks': ['t8', 't9'], 'strategy': 'min',
                       'task_type': 'multilabel', 'num_classes': 3},
        })
        info = model.get_pool_info()
        assert len(info) == 2
        names = [p['name'] for p in info]
        assert 'pool_a' in names
        assert 'pool_b' in names
        pool_b = [p for p in info if p['name'] == 'pool_b'][0]
        assert pool_b['task_type'] == 'multilabel'
        assert pool_b['num_classes'] == 3


class TestGetTaskInfoUnchanged:
    """get_task_info should NOT include pools — only architectural tasks."""

    def test_task_info_excludes_pools(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        info = model.get_task_info()
        names = [entry['name'] for entry in info]
        assert len(names) == 10
        assert 'pool' not in names


class TestSaveIncludesPools:
    """save() should persist output_pools in config.json."""

    def test_saved_config_has_output_pools(self, tmp_path):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        save_dir = str(tmp_path / 'model')
        
        # Build model by running forward pass (needed before save_weights)
        _ = model(random_input(), training=False)
        model.save(save_dir)
        
        with open(os.path.join(save_dir, 'config.json')) as f:
            config_dict = json.load(f)
        
        assert 'output_pools' in config_dict
        assert 'pool' in config_dict['output_pools']
        assert config_dict['output_pools']['pool']['strategy'] == 'max'
        assert config_dict['output_pools']['pool']['tasks'] == ['t0', 't1']

    def test_saved_config_no_pools_has_null(self, tmp_path):
        model = make_model()
        save_dir = str(tmp_path / 'model')
        _ = model(random_input(), training=False)
        model.save(save_dir)
        
        with open(os.path.join(save_dir, 'config.json')) as f:
            config_dict = json.load(f)
        
        assert config_dict['output_pools'] is None


class TestGetConfigIncludesPools:
    """get_config() should include output_pools for Keras serialization."""

    def test_get_config_with_pools(self):
        model = make_model(output_pools={
            'pool': {'tasks': ['t0', 't1'], 'strategy': 'max'}
        })
        cfg = model.get_config()
        assert 'output_pools' in cfg['config']
        assert cfg['config']['output_pools']['pool']['strategy'] == 'max'

    def test_get_config_no_pools(self):
        model = make_model()
        cfg = model.get_config()
        assert cfg['config']['output_pools'] is None


class TestDifferentTaskStructures:
    """Pools should work with various task structures."""

    def test_simple_two_branch_with_pool(self):
        config = ModelConfig(
            task_structure=[[2]],
            task_names=['a', 'b'],
            output_pools={'both': {'tasks': ['a', 'b'], 'strategy': 'mean'}},
        )
        model = MultiTaskModel(config=config, embedding_dim=32)
        out = model(tf.random.normal([4, 32]), training=False)
        assert 'both' in out
        expected = (out['a'].numpy() + out['b'].numpy()) / 2.0
        np.testing.assert_allclose(out['both'].numpy(), expected, atol=1e-6)

    def test_three_level_hierarchy_with_pool(self):
        names = [f'x{i}' for i in range(9)]
        config = ModelConfig(
            task_structure=[[2], [3, 4], [1, 2, 1, 1, 1, 1, 2]],
            task_names=names,
            output_pools={'left': {'tasks': ['x0', 'x1', 'x2'], 'strategy': 'max'}},
        )
        model = MultiTaskModel(config=config, embedding_dim=32)
        out = model(tf.random.normal([4, 32]), training=False)
        assert 'left' in out
        assert len(out) == 10  # 9 tasks + 1 pool


class TestPoolsWithMulticlass:
    """Pools should work when constituent tasks are multiclass."""

    def test_multiclass_pool_shape(self):
        config = ModelConfig(
            task_structure=[[2]],
            task_names=['a', 'b'],
            task_types=['multiclass', 'multiclass'],
            num_classes_per_task=[5, 5],
            output_pools={
                'both': {
                    'tasks': ['a', 'b'],
                    'strategy': 'max',
                    'task_type': 'multiclass',
                    'num_classes': 5,
                }
            },
        )
        model = MultiTaskModel(config=config, embedding_dim=32)
        out = model(tf.random.normal([4, 32]), training=False)
        # Multiclass outputs are [batch, num_classes]
        assert out['a'].shape == (4, 5)
        assert out['b'].shape == (4, 5)
        # Pooled output should have same shape
        assert out['both'].shape == (4, 5)


# ######################################################################
#  Run without pytest if needed
# ######################################################################

if __name__ == "__main__":
    pytest.main([__file__, "-v"])