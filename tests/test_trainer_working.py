import pytest
import tensorflow as tf
import numpy as np
from multitask.trainers.trainer import Trainer
from multitask.config import TrainingConfig, ModelConfig
from multitask.models.hierarchical import MultiTaskModel

@pytest.fixture
def dummy_trainer():
    """Initialize a trainer with the correct ModelConfig structure."""
    # 1. Define Config
    config = ModelConfig(
        task_structure=[[3]], 
        task_names=["task_binary", "task_multiclass", "task_multilabel"],
        task_types=["binary", "multiclass", "multilabel"],
        num_classes_per_task=[2, 3, 2]
    )
    
    # 2. Initialize Model (Trainer needs this to know output shapes)
    # Assuming standard encoder_dim, e.g., 768 for BERT
<<<<<<< HEAD
    model = MultiTaskModel(config=config, embedding_dim=768)
=======
    model = MultiTaskModel(config=config, encoder_dim=768)
>>>>>>> 1e4dba2b471c9d6a94e5ceef8e983820dc028337
    
    # 3. Initialize Trainer
    train_config = TrainingConfig()
    trainer = Trainer(
        model=model,
        config=train_config
    )
    
    return trainer

def test_route_inputs_column_logic(dummy_trainer):
    """
    Verifies that the routing helper correctly slices a concatenated tensor.
    Columns: Binary(1) + Multiclass(1) + Multilabel(2) = 4 Total.
    """
    # Batch size 8, 4 columns
    mock_y_true = tf.ones((8, 4), dtype=tf.float32)
    mock_y_pred = {} # Predictions are already dicts from model
    
    y_true_dict, _ = dummy_trainer._route_inputs(mock_y_true, mock_y_pred)
    
    # Assertions based on ModelConfig shapes
    assert y_true_dict['task_binary'].shape == (8, 1)
    assert y_true_dict['task_multiclass'].shape == (8, 1)
    assert y_true_dict['task_multilabel'].shape == (8, 2)

def test_dtype_safety_fix(dummy_trainer):
    """
    Ensures that mixing float64 (Python default) and float32 (TF) does not crash.
    Specifically tests the RA task checklist for 'tf.where' safety.
    """
    y_true = {'task_binary': tf.constant([0, 1], dtype=tf.float32)}
    y_pred = {'task_binary': tf.constant([[0.1], [0.9]], dtype=tf.float32)}
    
    # Python dicts use float64 by default
    class_weights = [{"task_binary": {0: 1.0, 1: 5.0}}] 
    task_weights = [1.0, 1.0, 1.0]
    
    loss_fn = dummy_trainer._build_loss_function(
        num_tasks=3, 
        class_weights=class_weights, 
        task_weights=task_weights
    )
    
    # This call would crash without the explicit tf.cast(..., tf.float32) fix
    loss = loss_fn(y_true, y_pred)
    assert loss.dtype == tf.float32
    assert loss > 0

def test_broadcasting_safety_reshape(dummy_trainer):
    """
    Tests the 'reshape(-1)' fix to prevent negative loss values caused by 
    incorrect geometric broadcasting.
    """
    # Scenario: Prediction is (2, 1) but Mask is accidentally (2,)
    # This triggers implicit expansion to (2, 2) and causes garbage loss.
    y_true = {'task_binary': tf.constant([[1.0], [-1.0]], dtype=tf.float32)} # One masked
    y_pred = {'task_binary': tf.constant([[0.8], [0.2]], dtype=tf.float32)}
    
    loss_fn = dummy_trainer._build_loss_function(
        num_tasks=1, 
        class_weights=[{0: 1.0, 1: 1.0}], 
        task_weights=[1.0]
    )
    
    loss = loss_fn(y_true, y_pred)
    # If fix is successful, loss is a standard positive scalar
    assert loss.shape == () 
    assert not np.isnan(loss.numpy())

def test_paper_compliant_normalization(dummy_trainer):
    """
    Verifies that the loss follows Equation 2 (Global Label-wise Normalization).
    Tasks with more valid labels should contribute more to the denominator.
    """
    # task_binary has 1 valid label; task_multilabel has 2 valid labels.
    # Total valid entries should be 3.
    y_true = {
        'task_binary': tf.constant([[1.0]], dtype=tf.float32),
        'task_multilabel': tf.constant([[1.0, 0.0]], dtype=tf.float32)
    }
    y_pred = {
        'task_binary': tf.constant([[0.5]], dtype=tf.float32),
        'task_multilabel': tf.constant([[0.5, 0.5]], dtype=tf.float32)
    }
    
    loss_fn = dummy_trainer._build_loss_function(
        num_tasks=2, 
        class_weights=[{0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}], 
        task_weights=[1.0, 1.0]
    )
    
    loss = loss_fn(y_true, y_pred)
    # Binary CE for 0.5 is approx 0.693. 
    # Total Loss = (0.693 * 1) + (0.693 * 2) = 2.079
    # Global Denominator = 3 labels
    # Expected = 0.693
    assert pytest.approx(loss.numpy(), 0.01) == 0.693