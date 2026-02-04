import pytest
import tensorflow as tf
from multitask.config import ModelConfig
from multitask.models.hierarchical import MultiTaskModel

@pytest.fixture
def dummy_model():
    """provides a basic 2-task model for testing."""
    config = ModelConfig(task_structure=[[2]], task_types=['binary', 'binary'])
    return MultiTaskModel(config=config, embedding_dim=128)

@pytest.fixture
def mismatched_config():
    """provides a config that intentionally mismatches the dummy_model."""
    return ModelConfig(task_structure=[[5]], task_types=['multiclass']*5)

@pytest.fixture
def dummy_model_config():
    """provides a basic ModelConfig for testing."""
    return ModelConfig(
        task_structure=[[2]], 
        task_types=['binary', 'binary'],
        task_names=['task_a', 'task_b']
    )