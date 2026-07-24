import json
try:
    import pytest
except ImportError:
    # Minimal pytest compatibility shim for running without pytest.
    # Provides pytest.raises as a context manager and a no-op fixture decorator.
    import contextlib

    class _PytestShim:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc_type, match=None):
            """Minimal reimplementation of pytest.raises."""
            try:
                yield
            except exc_type as e:
                if match:
                    import re
                    if not re.search(match, str(e)):
                        raise AssertionError(
                            f"Exception message '{e}' did not match pattern '{match}'"
                        ) from e
            else:
                raise AssertionError(f"Expected {exc_type.__name__} but no exception was raised")

        @staticmethod
        def fixture(func):
            """No-op decorator that mimics pytest.fixture.
            The decorated method is left callable so the fallback runner
            can call it to produce the fixture value."""
            return func

    pytest = _PytestShim()

from dataclasses import asdict
from multitask.config import ModelConfig, TrainingConfig
TASK_NAMES_10 = [f"task_{i}" for i in range(10)]


# ######################################################################
#  SECTION 1: REGRESSION TESTS
#  These test pre-existing behavior that must NOT change.
# ######################################################################

class TestRegressionModelConfigDefaults:
    """Ensure default ModelConfig construction still works exactly as before."""

    def test_default_construction(self):
        """ModelConfig() with no arguments should work and produce the same
        defaults that existed before the output_pools feature was added."""
        cfg = ModelConfig()
        assert cfg.task_structure == [[3], [2, 5, 3]]
        assert cfg.total_outputs == 10
        assert cfg.num_branch_levels == 2
        assert cfg.default_layer_sizes == [128, 64]
        assert cfg.shared_layer_sizes == [256, 128]
        assert cfg.trainable_encoder is True
        assert cfg.pooling_strategy is None
        assert cfg.task_names is None
        assert cfg.task_types is None
        assert cfg.branch_layer_sizes is None

    def test_default_num_classes_per_task(self):
        """num_classes_per_task should default to [2]*total_outputs (all binary)."""
        cfg = ModelConfig()
        assert cfg.num_classes_per_task == [2] * 10

    def test_default_task_types_none(self):
        """task_types should remain None when not provided (defaults to binary
        via get_task_type)."""
        cfg = ModelConfig()
        assert cfg.task_types is None
        for i in range(cfg.total_outputs):
            assert cfg.get_task_type(i) == 'binary'

    def test_default_output_dim(self):
        """Binary tasks should have output dim of 1."""
        cfg = ModelConfig()
        for i in range(cfg.total_outputs):
            assert cfg.get_output_dim_for_task(i) == 1

    def test_get_layer_sizes_default(self):
        """Without branch_layer_sizes, all levels should use default_layer_sizes."""
        cfg = ModelConfig()
        for level in range(cfg.num_branch_levels):
            assert cfg.get_layer_sizes(level) == [128, 64]

    def test_get_layer_sizes_custom(self):
        """With branch_layer_sizes, each level should use its own sizes."""
        cfg = ModelConfig(branch_layer_sizes=[[64, 32], [32, 16]])
        assert cfg.get_layer_sizes(0) == [64, 32]
        assert cfg.get_layer_sizes(1) == [32, 16]


class TestRegressionModelConfigValidation:
    """Existing validation rules must still fire correctly."""

    def test_empty_task_structure(self):
        with pytest.raises(ValueError, match="at least one branching level"):
            ModelConfig(task_structure=[])

    def test_first_level_multiple_elements(self):
        with pytest.raises(ValueError, match="exactly one element"):
            ModelConfig(task_structure=[[2, 3]])

    def test_inconsistent_levels(self):
        """Level 1 should have sum(level_0) elements."""
        with pytest.raises(ValueError, match="should have 3"):
            ModelConfig(task_structure=[[3], [2, 5]])  # needs 3 elements

    def test_negative_branch_count(self):
        with pytest.raises(ValueError, match="positive integers"):
            ModelConfig(task_structure=[[3], [-1, 5, 3]])

    def test_wrong_task_names_length(self):
        with pytest.raises(ValueError, match="task_names length"):
            ModelConfig(task_names=["a", "b"])  # needs 10

    def test_wrong_task_types_length(self):
        with pytest.raises(ValueError, match="task_types length"):
            ModelConfig(task_types=["binary"])  # needs 10

    def test_invalid_task_type(self):
        with pytest.raises(ValueError, match="Invalid task_type"):
            ModelConfig(task_types=["binary"] * 9 + ["not_a_valid_type"])

    def test_binary_task_wrong_num_classes(self):
        with pytest.raises(ValueError, match="Binary tasks should have num_classes=2"):
            ModelConfig(
                task_types=["binary"] * 10,
                num_classes_per_task=[2] * 9 + [5],
            )

    def test_multiclass_task_too_few_classes(self):
        with pytest.raises(ValueError, match="Multiclass tasks should have num_classes >= 3"):
            ModelConfig(
                task_types=["binary"] * 9 + ["multiclass"],
                num_classes_per_task=[2] * 9 + [2],
            )

    def test_invalid_pooling_strategy_legacy(self):
        """The deprecated pooling_strategy field should still validate."""
        with pytest.raises(ValueError, match="pooling_strategy must be one of"):
            ModelConfig(pooling_strategy="invalid")

    def test_branch_layer_sizes_wrong_length(self):
        with pytest.raises(ValueError, match="branch_layer_sizes length"):
            ModelConfig(branch_layer_sizes=[[64, 32]])  # needs 2 lists for 2 levels


class TestRegressionModelConfigProperties:
    """Pre-existing properties and methods must return the same values."""

    def test_total_outputs_simple(self):
        cfg = ModelConfig(task_structure=[[2]])
        assert cfg.total_outputs == 2

    def test_total_outputs_two_level(self):
        cfg = ModelConfig(task_structure=[[3], [2, 5, 3]])
        assert cfg.total_outputs == 10

    def test_total_outputs_three_level(self):
        cfg = ModelConfig(task_structure=[[2], [3, 4], [1, 2, 1, 1, 1, 1, 2]])
        assert cfg.total_outputs == 1 + 2 + 1 + 1 + 1 + 1 + 2  # 9

    def test_num_branch_levels(self):
        cfg = ModelConfig(task_structure=[[2], [3, 4], [1, 2, 1, 1, 1, 1, 2]])
        assert cfg.num_branch_levels == 3

    def test_get_branches_at_level(self):
        cfg = ModelConfig(task_structure=[[3], [2, 5, 3]])
        assert cfg.get_branches_at_level(0) == 3
        assert cfg.get_branches_at_level(1) == 10

    def test_get_task_type_with_types(self):
        types = ["binary"] * 9 + ["multiclass"]
        cfg = ModelConfig(task_types=types, num_classes_per_task=[2] * 9 + [5])
        assert cfg.get_task_type(0) == "binary"
        assert cfg.get_task_type(9) == "multiclass"

    def test_get_num_classes_per_task_method(self):
        cfg = ModelConfig(num_classes_per_task=[2] * 9 + [5])
        assert cfg.get_num_classes_per_task(0) == 2
        assert cfg.get_num_classes_per_task(9) == 5

    def test_get_output_dim_multiclass(self):
        cfg = ModelConfig(
            task_types=["binary"] * 9 + ["multiclass"],
            num_classes_per_task=[2] * 9 + [5],
        )
        assert cfg.get_output_dim_for_task(0) == 1   # binary → 1
        assert cfg.get_output_dim_for_task(9) == 5   # multiclass → K


class TestRegressionTrainingConfig:
    """TrainingConfig must be completely unaffected by our changes."""

    def test_default_construction(self):
        tc = TrainingConfig()
        assert tc.epochs == 20
        assert tc.batch_size == 32
        assert tc.learning_rate == 3e-4
        assert tc.optimizer_type == "adam"

    def test_validation(self):
        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-1)
        with pytest.raises(ValueError):
            TrainingConfig(dropout_rate=1.5)


# ######################################################################
#  SECTION 2: UNIT TESTS
#  Test every new method, property, and validation path for output_pools.
# ######################################################################

class TestPoolValidationRequired:
    """Test that required fields and basic structure are enforced."""

    def test_pools_must_be_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            ModelConfig(task_names=TASK_NAMES_10, output_pools="not_a_dict")

    def test_pool_config_must_be_dict(self):
        with pytest.raises(ValueError, match="config must be a dict"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={"pool_a": "not_a_dict"},
            )

    def test_missing_tasks_field(self):
        with pytest.raises(ValueError, match="missing required field 'tasks'"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={"pool_a": {"strategy": "max"}},
            )

    def test_missing_strategy_field(self):
        with pytest.raises(ValueError, match="missing required field 'strategy'"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={"pool_a": {"tasks": ["task_0", "task_1"]}},
            )

    def test_task_names_required_when_pools_used(self):
        """output_pools without task_names should be rejected because we
        can't validate task references."""
        with pytest.raises(ValueError, match="task_names must be provided"):
            ModelConfig(output_pools={"pool_a": {"tasks": ["a", "b"], "strategy": "max"}})

    def test_empty_pool_name(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={"": {"tasks": ["task_0", "task_1"], "strategy": "max"}},
            )


class TestPoolValidationTaskReferences:
    """Test validation of task name references within pools."""

    def test_nonexistent_task_reference(self):
        with pytest.raises(ValueError, match="not in task_names"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {"tasks": ["task_0", "nonexistent"], "strategy": "max"}
                },
            )

    def test_single_task_pool_rejected(self):
        """Pooling a single task is pointless and should be rejected."""
        with pytest.raises(ValueError, match="at least 2 task names"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={"pool_a": {"tasks": ["task_0"], "strategy": "max"}},
            )

    def test_empty_tasks_list_rejected(self):
        with pytest.raises(ValueError, match="at least 2 task names"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={"pool_a": {"tasks": [], "strategy": "max"}},
            )

    def test_duplicate_task_across_pools(self):
        """A task cannot appear in more than one pool."""
        with pytest.raises(ValueError, match="already assigned to pool"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "max"},
                    "pool_b": {"tasks": ["task_1", "task_2"], "strategy": "mean"},
                },
            )


class TestPoolValidationNameCollisions:
    """Pool names must not collide with architectural task names."""

    def test_pool_name_equals_task_name(self):
        with pytest.raises(ValueError, match="conflicts with an existing task name"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "task_0": {"tasks": ["task_1", "task_2"], "strategy": "max"}
                },
            )

    def test_pool_names_unique_from_all_tasks(self):
        """Even the last task name should be caught."""
        with pytest.raises(ValueError, match="conflicts with an existing task name"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "task_9": {"tasks": ["task_0", "task_1"], "strategy": "max"}
                },
            )


class TestPoolValidationStrategy:
    """Test that only valid pooling strategies are accepted."""

    def test_valid_strategies(self):
        for strategy in ("max", "mean", "min"):
            cfg = ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {"tasks": ["task_0", "task_1"], "strategy": strategy}
                },
            )
            assert cfg.get_pool_config("pool_a")["strategy"] == strategy

    def test_invalid_strategy_sum(self):
        with pytest.raises(ValueError, match="not supported"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "sum"}
                },
            )

    def test_invalid_strategy_concat(self):
        with pytest.raises(ValueError, match="not supported"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "concat"}
                },
            )

    def test_invalid_strategy_arbitrary(self):
        with pytest.raises(ValueError, match="not supported"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "product"}
                },
            )


class TestPoolValidationTypeAndClasses:
    """Test validation of optional task_type and num_classes fields."""

    def test_default_task_type_is_binary(self):
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={"pool_a": {"tasks": ["task_0", "task_1"], "strategy": "max"}},
        )
        assert cfg.get_pool_task_type("pool_a") == "binary"

    def test_default_num_classes_is_2(self):
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={"pool_a": {"tasks": ["task_0", "task_1"], "strategy": "max"}},
        )
        assert cfg.get_pool_num_classes("pool_a") == 2

    def test_explicit_task_type(self):
        for tt in ("binary", "multiclass", "multilabel"):
            nc = 2 if tt == "binary" else 5
            cfg = ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {
                        "tasks": ["task_0", "task_1"],
                        "strategy": "max",
                        "task_type": tt,
                        "num_classes": nc,
                    }
                },
            )
            assert cfg.get_pool_task_type("pool_a") == tt

    def test_invalid_task_type(self):
        with pytest.raises(ValueError, match="not valid"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {
                        "tasks": ["task_0", "task_1"],
                        "strategy": "max",
                        "task_type": "regression",
                    }
                },
            )

    def test_binary_wrong_num_classes(self):
        with pytest.raises(ValueError, match="Binary pools should have num_classes=2"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {
                        "tasks": ["task_0", "task_1"],
                        "strategy": "max",
                        "task_type": "binary",
                        "num_classes": 3,
                    }
                },
            )

    def test_multiclass_too_few_classes(self):
        with pytest.raises(ValueError, match="Multiclass pools should have num_classes >= 3"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {
                        "tasks": ["task_0", "task_1"],
                        "strategy": "max",
                        "task_type": "multiclass",
                        "num_classes": 2,
                    }
                },
            )

    def test_num_classes_below_2(self):
        with pytest.raises(ValueError, match="num_classes must be an integer >= 2"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {
                        "tasks": ["task_0", "task_1"],
                        "strategy": "max",
                        "num_classes": 1,
                    }
                },
            )

    def test_num_classes_not_int(self):
        with pytest.raises(ValueError, match="num_classes must be an integer >= 2"):
            ModelConfig(
                task_names=TASK_NAMES_10,
                output_pools={
                    "pool_a": {
                        "tasks": ["task_0", "task_1"],
                        "strategy": "max",
                        "num_classes": 2.5,
                    }
                },
            )


class TestPoolProperties:
    """Test the new properties: pool_names, all_output_names, has_pools."""

    def test_no_pools_pool_names_empty(self):
        cfg = ModelConfig(task_names=TASK_NAMES_10)
        assert cfg.pool_names == []

    def test_no_pools_has_pools_false(self):
        cfg = ModelConfig(task_names=TASK_NAMES_10)
        assert cfg.has_pools is False

    def test_no_pools_all_output_names_equals_task_names(self):
        cfg = ModelConfig(task_names=TASK_NAMES_10)
        assert cfg.all_output_names == TASK_NAMES_10

    def test_single_pool_names(self):
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={"agg": {"tasks": ["task_0", "task_1"], "strategy": "max"}},
        )
        assert cfg.pool_names == ["agg"]

    def test_multiple_pools_sorted(self):
        """pool_names should be sorted alphabetically."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "zebra_pool": {"tasks": ["task_0", "task_1"], "strategy": "max"},
                "alpha_pool": {"tasks": ["task_2", "task_3"], "strategy": "mean"},
            },
        )
        assert cfg.pool_names == ["alpha_pool", "zebra_pool"]

    def test_has_pools_true(self):
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={"agg": {"tasks": ["task_0", "task_1"], "strategy": "max"}},
        )
        assert cfg.has_pools is True

    def test_all_output_names_order(self):
        """Task names first (original order), then pool names (sorted)."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "z_pool": {"tasks": ["task_0", "task_1"], "strategy": "max"},
                "a_pool": {"tasks": ["task_2", "task_3"], "strategy": "mean"},
            },
        )
        expected = TASK_NAMES_10 + ["a_pool", "z_pool"]
        assert cfg.all_output_names == expected

    def test_all_output_names_no_task_names(self):
        """When task_names is None, auto-generated names should be used."""
        cfg = ModelConfig()  # no task_names, no pools
        expected = [f"task_{i}" for i in range(10)]
        assert cfg.all_output_names == expected


class TestPoolHelperMethods:
    """Test get_pool_config, get_pool_task_type, get_pool_num_classes, is_pool."""

    @pytest.fixture
    def cfg_with_pools(self):
        return ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pool_max": {
                    "tasks": ["task_0", "task_1", "task_2"],
                    "strategy": "max",
                },
                "pool_mean": {
                    "tasks": ["task_3", "task_4"],
                    "strategy": "mean",
                    "task_type": "multilabel",
                    "num_classes": 4,
                },
            },
        )

    def test_get_pool_config_defaults_filled(self, cfg_with_pools):
        """get_pool_config should fill in defaults for optional fields."""
        pc = cfg_with_pools.get_pool_config("pool_max")
        assert pc == {
            "tasks": ["task_0", "task_1", "task_2"],
            "strategy": "max",
            "task_type": "binary",
            "num_classes": 2,
        }

    def test_get_pool_config_explicit_fields(self, cfg_with_pools):
        pc = cfg_with_pools.get_pool_config("pool_mean")
        assert pc == {
            "tasks": ["task_3", "task_4"],
            "strategy": "mean",
            "task_type": "multilabel",
            "num_classes": 4,
        }

    def test_get_pool_config_returns_copy(self, cfg_with_pools):
        """Returned dict should be a new dict, not a reference to internal state."""
        pc1 = cfg_with_pools.get_pool_config("pool_max")
        pc2 = cfg_with_pools.get_pool_config("pool_max")
        assert pc1 == pc2
        assert pc1 is not pc2  # different objects

    def test_get_pool_config_nonexistent_raises_keyerror(self, cfg_with_pools):
        with pytest.raises(KeyError, match="No pool named"):
            cfg_with_pools.get_pool_config("nonexistent")

    def test_get_pool_config_no_pools_raises_keyerror(self):
        cfg = ModelConfig(task_names=TASK_NAMES_10)
        with pytest.raises(KeyError, match="No pool named"):
            cfg.get_pool_config("anything")

    def test_get_pool_task_type(self, cfg_with_pools):
        assert cfg_with_pools.get_pool_task_type("pool_max") == "binary"
        assert cfg_with_pools.get_pool_task_type("pool_mean") == "multilabel"

    def test_get_pool_num_classes(self, cfg_with_pools):
        assert cfg_with_pools.get_pool_num_classes("pool_max") == 2
        assert cfg_with_pools.get_pool_num_classes("pool_mean") == 4

    def test_is_pool_true_for_pools(self, cfg_with_pools):
        assert cfg_with_pools.is_pool("pool_max") is True
        assert cfg_with_pools.is_pool("pool_mean") is True

    def test_is_pool_false_for_tasks(self, cfg_with_pools):
        for name in TASK_NAMES_10:
            assert cfg_with_pools.is_pool(name) is False

    def test_is_pool_false_for_unknown(self, cfg_with_pools):
        assert cfg_with_pools.is_pool("unknown_name") is False

    def test_is_pool_no_pools(self):
        cfg = ModelConfig(task_names=TASK_NAMES_10)
        assert cfg.is_pool("anything") is False


# ######################################################################
#  SECTION 3: INTEGRATION TESTS
#  Test that output_pools interacts correctly with other ModelConfig
#  features and with external operations (serialization, etc.)
# ######################################################################

class TestPoolsWithExistingFeatures:
    """Pools combined with task_types, num_classes_per_task, etc."""

    def test_pools_with_mixed_task_types(self):
        """output_pools should work when tasks have different types."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            task_types=["binary"] * 9 + ["multiclass"],
            num_classes_per_task=[2] * 9 + [5],
            output_pools={
                "binary_pool": {
                    "tasks": ["task_0", "task_1"],
                    "strategy": "max",
                }
            },
        )
        # Existing task-level methods should still work
        assert cfg.get_task_type(0) == "binary"
        assert cfg.get_task_type(9) == "multiclass"
        assert cfg.get_num_classes_per_task(9) == 5
        # Pool should be accessible
        assert cfg.get_pool_task_type("binary_pool") == "binary"

    def test_pools_dont_affect_total_outputs(self):
        """total_outputs counts architectural tasks only, not pools."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "max"},
                "pool_b": {"tasks": ["task_2", "task_3", "task_4"], "strategy": "mean"},
            },
        )
        assert cfg.total_outputs == 10  # still 10, not 12

    def test_pools_dont_affect_task_names(self):
        """task_names should still be exactly what the user provided."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "max"},
            },
        )
        assert cfg.task_names == TASK_NAMES_10

    def test_pools_dont_affect_num_classes_per_task_list(self):
        """num_classes_per_task list should still be length total_outputs."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "max"},
            },
        )
        assert len(cfg.num_classes_per_task) == 10

    def test_pools_with_legacy_pooling_strategy(self):
        """output_pools and the deprecated pooling_strategy can coexist."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            pooling_strategy="max",  # legacy field
            output_pools={
                "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "mean"},
            },
        )
        assert cfg.pooling_strategy == "max"  # untouched
        assert cfg.get_pool_config("pool_a")["strategy"] == "mean"


class TestPoolSerialization:
    """Test that configs with pools survive JSON round-trips (save/load)."""

    def test_json_round_trip_no_pools(self):
        """Configs without pools should serialize and reconstruct identically."""
        cfg = ModelConfig(task_names=TASK_NAMES_10)
        json_str = json.dumps(asdict(cfg))
        loaded = ModelConfig(**json.loads(json_str))
        assert loaded.task_names == cfg.task_names
        assert loaded.output_pools is None
        assert loaded.has_pools is False

    def test_json_round_trip_with_pools(self):
        """Configs with pools should serialize and reconstruct identically."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pool_a": {
                    "tasks": ["task_0", "task_1"],
                    "strategy": "max",
                    "task_type": "binary",
                    "num_classes": 2,
                },
                "pool_b": {
                    "tasks": ["task_5", "task_6", "task_7"],
                    "strategy": "mean",
                },
            },
        )
        json_str = json.dumps(asdict(cfg))
        loaded = ModelConfig(**json.loads(json_str))

        assert loaded.pool_names == cfg.pool_names
        assert loaded.has_pools is True
        assert loaded.get_pool_config("pool_a") == cfg.get_pool_config("pool_a")
        assert loaded.get_pool_config("pool_b") == cfg.get_pool_config("pool_b")
        assert loaded.all_output_names == cfg.all_output_names

    def test_json_round_trip_with_pools_and_task_types(self):
        """Full config with pools, task_types, and num_classes should survive."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            task_types=["binary"] * 9 + ["multiclass"],
            num_classes_per_task=[2] * 9 + [4],
            output_pools={
                "pool_a": {
                    "tasks": ["task_0", "task_1"],
                    "strategy": "min",
                    "task_type": "multilabel",
                    "num_classes": 3,
                },
            },
        )
        json_str = json.dumps(asdict(cfg))
        loaded = ModelConfig(**json.loads(json_str))

        assert loaded.get_task_type(9) == "multiclass"
        assert loaded.get_pool_config("pool_a")["task_type"] == "multilabel"
        assert loaded.get_pool_config("pool_a")["num_classes"] == 3

    def test_json_round_trip_preserves_task_order_in_pool(self):
        """The order of tasks within a pool should survive serialization."""
        task_list = ["task_3", "task_1", "task_5"]
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pool_a": {"tasks": task_list, "strategy": "max"},
            },
        )
        json_str = json.dumps(asdict(cfg))
        loaded = ModelConfig(**json.loads(json_str))
        assert loaded.get_pool_config("pool_a")["tasks"] == task_list


class TestPoolEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_tasks_in_one_pool(self):
        """It should be valid to pool every task into one output."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "everything": {"tasks": TASK_NAMES_10, "strategy": "max"},
            },
        )
        assert cfg.pool_names == ["everything"]
        assert len(cfg.all_output_names) == 11  # 10 tasks + 1 pool

    def test_exactly_two_tasks_in_pool(self):
        """Minimum valid pool size is 2."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pair": {"tasks": ["task_0", "task_1"], "strategy": "min"},
            },
        )
        assert cfg.get_pool_config("pair")["tasks"] == ["task_0", "task_1"]

    def test_many_pools_non_overlapping(self):
        """Multiple pools covering disjoint subsets of tasks."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "group_a": {"tasks": ["task_0", "task_1"], "strategy": "max"},
                "group_b": {"tasks": ["task_2", "task_3"], "strategy": "mean"},
                "group_c": {"tasks": ["task_4", "task_5"], "strategy": "min"},
                "group_d": {"tasks": ["task_6", "task_7", "task_8", "task_9"], "strategy": "max"},
            },
        )
        assert len(cfg.pool_names) == 4
        assert len(cfg.all_output_names) == 14  # 10 tasks + 4 pools

    def test_tasks_not_in_any_pool(self):
        """Tasks that aren't in any pool should still work normally."""
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "partial": {"tasks": ["task_0", "task_1"], "strategy": "max"},
            },
        )
        # task_5 is not in any pool
        assert cfg.is_pool("task_5") is False
        assert cfg.get_task_type(5) == "binary"

    def test_different_task_structure_with_pools(self):
        """Pools should work with non-default task structures."""
        names = ["a", "b", "c", "d"]
        cfg = ModelConfig(
            task_structure=[[2], [2, 2]],
            task_names=names,
            output_pools={
                "left_pool": {"tasks": ["a", "b"], "strategy": "max"},
                "right_pool": {"tasks": ["c", "d"], "strategy": "mean"},
            },
        )
        assert cfg.total_outputs == 4
        assert cfg.pool_names == ["left_pool", "right_pool"]
        assert cfg.all_output_names == ["a", "b", "c", "d", "left_pool", "right_pool"]

    def test_three_level_hierarchy_with_pools(self):
        """Pools should work with deep hierarchies."""
        names = [f"t{i}" for i in range(9)]
        cfg = ModelConfig(
            task_structure=[[2], [3, 4], [1, 2, 1, 1, 1, 1, 2]],
            task_names=names,
            output_pools={
                "left_branch": {"tasks": ["t0", "t1", "t2"], "strategy": "max"},
            },
        )
        assert cfg.total_outputs == 9
        assert cfg.has_pools is True


class TestPoolInteractionWithIsPool:
    """Verify is_pool works correctly for classifying all output names."""

    def test_classify_all_output_names(self):
        cfg = ModelConfig(
            task_names=TASK_NAMES_10,
            output_pools={
                "pool_a": {"tasks": ["task_0", "task_1"], "strategy": "max"},
                "pool_b": {"tasks": ["task_8", "task_9"], "strategy": "mean"},
            },
        )
        # Iterate over all outputs and classify each
        tasks = []
        pools = []
        for name in cfg.all_output_names:
            if cfg.is_pool(name):
                pools.append(name)
            else:
                tasks.append(name)

        assert tasks == TASK_NAMES_10
        assert sorted(pools) == ["pool_a", "pool_b"]


# ######################################################################
#  Run with plain python if pytest isn't available
# ######################################################################

if __name__ == "__main__":
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Fallback: run all test methods manually using introspection.
        # Discovers every class starting with "Test" and every method
        # starting with "test_", handles fixtures named 'cfg_with_pools',
        # counts passes and failures, and prints a summary.
        import inspect
        import traceback

        passed = 0
        failed = 0
        errors = []

        # Collect all test classes defined in this module
        test_classes = [
            obj for name, obj in globals().items()
            if inspect.isclass(obj) and name.startswith("Test")
        ]

        for cls in test_classes:
            # Instantiate the test class
            instance = cls()

            # Check if the class has a fixture method we need to call
            fixture_method = getattr(instance, "cfg_with_pools", None)
            if callable(fixture_method):
                fixture_value = fixture_method()
            else:
                fixture_value = None

            # Collect all test methods
            methods = [
                name for name in dir(instance)
                if name.startswith("test_") and callable(getattr(instance, name))
            ]

            for method_name in sorted(methods):
                full_name = f"{cls.__name__}::{method_name}"
                method = getattr(instance, method_name)

                try:
                    # If method accepts an argument (fixture), pass it
                    sig = inspect.signature(method)
                    if len(sig.parameters) > 0 and fixture_value is not None:
                        method(fixture_value)
                    else:
                        method()
                    passed += 1
                    print(f"  PASS  {full_name}")
                except Exception as e:
                    failed += 1
                    errors.append((full_name, e))
                    print(f"  FAIL  {full_name}: {e}")

        # Summary
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
        if errors:
            print(f"\nFailed tests:")
            for name, err in errors:
                print(f"  {name}")
                traceback.print_exception(type(err), err, err.__traceback__)
        print(f"{'='*60}")

        if failed > 0:
            exit(1)
