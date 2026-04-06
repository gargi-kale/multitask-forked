import os
import tensorflow as tf
import numpy as np
import pytest
import keras
from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer

def build_model():
    task_structure = [[5]]
    total_outputs = 5

    model_config = ModelConfig(
        task_structure=task_structure,
        task_names=[f"task_{i}" for i in range(total_outputs)],
        num_classes_per_task=[2] * total_outputs,
    )

    return MultiTaskModel(config=model_config, 
                          encoder=None,
                          embedding_dim=128,)


def generate_mock_data(num_samples=32, num_tasks=5):
    x = np.random.rand(num_samples, 128).astype(np.float32)
    y_dict = {
        f"task_{i}": np.random.randint(0, 2, (num_samples, 1)).astype(np.float32)
        for i in range(num_tasks)
    }
    return (
        tf.data.Dataset
        .from_tensor_slices((x, y_dict))
        .batch(32)
)

optimizers = [
    "adamw",
    "adam",
    "sgd",
    "rmsprop",
    "nadam",
    "adagrad",
    "adadelta",
    "adamax", 
    "ftrl",
    "lion", 
    # 'lamb', # not available
    # 'muon', # not available
    # 'lossscaleoptimizer', # this is a wrapper, not a standalone optimizer
]

@pytest.mark.parametrize("optimizer_name", optimizers)
def test_optimizer_instantiation(optimizer_name):
    model = build_model()
    config = TrainingConfig(optimizer_type=optimizer_name, epochs=1, use_warmup=False, learning_rate=1e-2)
    trainer = Trainer(model=model, config=config)
    optimizer = trainer._build_optimizer(steps_per_epoch=2)
    
    # Robust check: Ensure it has the interface of an optimizer
    assert hasattr(optimizer, 'apply_gradients') or hasattr(optimizer, 'minimize')

@pytest.mark.parametrize("optimizer_name", optimizers)
def test_optimizer_single_training_step(optimizer_name):
    model = build_model()

    config = TrainingConfig(optimizer_type=optimizer_name, epochs=1, use_warmup=False, learning_rate=1e-2)
    trainer = Trainer(model=model, config=config)
    
    train_ds = generate_mock_data(num_tasks=5) # Match model's 5 tasks!
    sample_batch = next(iter(train_ds))
    model(sample_batch[0])
    initial_weights = [np.copy(w) for w in model.get_weights()]

    trainer.fit(train_ds, train_ds, num_tasks=5, verbose=0) # More steps

    final_weights = model.get_weights()
    weight_changed = any(not np.allclose(w0, w1) for w0, w1 in zip(initial_weights, final_weights))
    
    assert weight_changed, f"Optimizer {optimizer_name} did not update weights."
    

@pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "sgd"])
@pytest.mark.parametrize("schedule", ["polynomial_decay", "cosine_decay"])
def test_optimizer_with_schedule(optimizer_name, schedule):
    config = TrainingConfig(
        optimizer_type=optimizer_name,
        use_lr_schedule=True,  # ← Test your actual config!
        lr_schedule_type=schedule
    )
    trainer = Trainer(model=build_model(), config=config)
    optimizer = trainer._build_optimizer(steps_per_epoch=100)
    lr_obj = getattr(optimizer, "_learning_rate", optimizer.learning_rate)
    assert isinstance(lr_obj, keras.optimizers.schedules.LearningRateSchedule), (
        f"Expected schedule object, but got {type(lr_obj)}"
    )

# Test 2: Warmup Wrapper
def test_optimizer_with_warmup():
    config = TrainingConfig(
        optimizer_type='adam',
        use_warmup=True,  # ← Test your custom feature!
        warmup_percent=10
    )
    trainer = Trainer(model=build_model(), config=config)
    optimizer = trainer._build_optimizer(steps_per_epoch=100)
    lr_obj = getattr(optimizer, "_learning_rate", optimizer.learning_rate)
    
    from multitask.utils.compat import WarmUpWrapper
    assert isinstance(lr_obj, WarmUpWrapper), (
        f"Expected WarmUpWrapper, but got {type(lr_obj)}"
    )