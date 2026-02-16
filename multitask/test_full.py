import os
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
# from legacy.models import model
from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer
from multitask.models.persistence import load_model

# get feature columns
prop_task = ['propaganda', 'Loaded_Language', 'Slogans']
ibm_task = ['WA_bi']
iac_task = ['emotion_fact_bi']
final_task = prop_task + ibm_task + iac_task


def prepare_data(file_name, sample_size=200):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", file_name)
    df = pd.read_csv(file_path).sample(n=sample_size, random_state=42)
    
    x = np.random.rand(len(df), 128).astype(np.float32)
    
    y_dict = {}
    # map the tasks used in our test config
    for task in final_task:
        raw_col = task.replace('_', ',') if ',' in task else task
        y_dict[task] = df[raw_col].fillna(-1).values.reshape(-1, 1).astype(np.float32)
        
    return tf.data.Dataset.from_tensor_slices((x, y_dict)).batch(8)

def get_configs():
    num_tasks = len(final_task)

    model_cfg = ModelConfig(
        task_names=final_task, 
        task_types=['binary'] * num_tasks,
        num_classes_per_task=[2] * num_tasks,
    
        task_structure=[[3], [3, 1, 1]],
        
        shared_layer_sizes=[256, 128],
        branch_layer_sizes=[[64, 32], [16, 8]] 
    )
    
    train_cfg = TrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=0.0003,
        use_warmup=True,
        dropout_rate=0.4
    )
    
    return model_cfg, train_cfg

def run_test():
    # Setup paths
    save_path = "integration_model_final"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # Data preparation
    sample_size = 400
    batch_size = 8
    steps = sample_size // batch_size

    print("Preparing data")
    train_ds = prepare_data('data/raw/no_prop_overlap_train.csv', sample_size)
    val_ds = prepare_data('data/raw/no_prop_overlap_val.csv', 100)
    print(f"data loaded. Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")
    
    # Model initialization
    print("\ninitializing the model and trainer")
    model_cfg, train_cfg = get_configs()
    model = MultiTaskModel(config=model_cfg, embedding_dim=128)
    trainer = Trainer(model=model, config=train_cfg)
    print(f"Model initialized with {len(final_task)} tasks")

    # Training
    print("\ntraining start...")
    history, thresholds = trainer.fit(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_tasks=len(final_task),
        steps_per_epoch=steps,
        optimize_thresholds=True
    )
    
    # Verify training outputs
    assert history is not None, "Training history is None"
    assert thresholds is not None, "Thresholds not returned"
    assert len(thresholds) == len(final_task), \
        f"Expected {len(final_task)} thresholds, got {len(thresholds)}"
    print(f"Training completed")
    print(f"Optimized {len(thresholds)} thresholds")

    # Evaluation
    print("\nEvaluation (trainer.evaluate)")
    eval_results = trainer.evaluate(val_ds, thresholds=thresholds)
    print(f"Evaluation completed")
    print(f"Results: {eval_results}")

    # Persistence
    print("\n>>> Persistence (save/load)")
    trainer.save_checkpoint(save_path, thresholds=thresholds)
    print(f"Model saved to {save_path}/")
    
    loaded_model, loaded_thresholds = load_model(save_path)
    assert len(loaded_thresholds) == len(final_task), \
        f"Threshold count mismatch: expected {len(final_task)}, got {len(loaded_thresholds)}"
    print(f"Model loaded with {len(loaded_thresholds)} thresholds")
    
    # Prediction with loaded model
    print("\nPredictions")
    dummy_input = tf.random.uniform((1, 128))
    preds = loaded_model(dummy_input, training=False)
    
    assert isinstance(preds, dict), f"Expected dict predictions, got {type(preds)}"
    assert len(preds) == len(final_task), \
        f"Prediction count mismatch: expected {len(final_task)}, got {len(preds)}"
    
    # Verify each prediction has correct shape and range
    for task_name, pred in preds.items():
        assert pred.shape == (1, 1), \
            f"Task {task_name}: expected shape (1, 1), got {pred.shape}"
        assert 0 <= pred.numpy()[0, 0] <= 1, \
            f"Task {task_name}: prediction {pred.numpy()[0, 0]} not in [0, 1]"
    
    print(f"All {len(preds)} predictions have correct shape and range")
    
    print("\n" + "="*60)
    print("test passed!")
    print("="*60)

if __name__ == "__main__":
    run_test()