import os
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
from multitask.config import ModelConfig, TrainingConfig
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer
from multitask.models.persistence import load_model

task_cols = [
    'Loaded_Language', 'Slogans', 'Black-and-White_Fallacy', 'Flag-Waving', 
    'Name_Calling', 'Whataboutism', 'Causal_Oversimplification', 
    'Exaggeration,Minimisation', 'Doubt', 'Appeal_to_Authority', 
    'Repetition', 'Appeal_to_fear-prejudice', 'Red_Herring', 
    'Thought-terminating_Cliches', 'Reductio_ad_hitlerum', 'Bandwagon', 
    'Obfuscation,Intentional_Vagueness,Confusion', 'Straw_Men'
]
clean_task_cols = [name.replace(',', '_') for name in task_cols]

core_tasks = ['propaganda', 'WA_bi', 'emotion_fact_bi']
all_tasks = core_tasks + clean_task_cols

def prepare_data(file_name, sample_size=200):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", file_name)
    df = pd.read_csv(file_path).sample(n=sample_size, random_state=42)
    x = np.random.rand(len(df), 128).astype(np.float32)
    
    y_dict = {}
    # Map core tasks
    for task in core_tasks:
        y_dict[task] = df[task].fillna(-1).values.reshape(-1, 1).astype(np.float32)
    
    # Map techniques: Use RAW for df lookup, CLEAN for dictionary key
    for raw, clean in zip(task_cols, clean_task_cols):
        y_dict[clean] = df[raw].fillna(-1).values.reshape(-1, 1).astype(np.float32)
        
    return tf.data.Dataset.from_tensor_slices((x, y_dict)).batch(8)

# hierarchical config
task_names = core_tasks + clean_task_cols
task_types = ['binary'] * len(task_names)
num_classes_per_task = [2] * len(task_names)

def get_configs():
    # Architecture: 21 parallel heads
    model_cfg = ModelConfig(
        task_names=all_tasks,
        task_types=['binary'] * len(all_tasks),
        num_classes_per_task=[2] * len(all_tasks),
        task_structure=[[len(all_tasks)]],
        shared_layer_sizes=[256, 128],
        branch_layer_sizes=[[64, 32]]
    )
    
    # Trainer config using dropout_rate as defined in config.py
    train_cfg = TrainingConfig(epochs=2, batch_size=8, use_warmup=True, dropout_rate=0.1)
    return model_cfg, train_cfg


def run_test():
    # Setup paths
    save_path = "integration_model_final"
    if os.path.exists(save_path): shutil.rmtree(save_path)

    # Data
    train_ds = prepare_data('data/raw/no_prop_overlap_train.csv', 200)
    val_ds = prepare_data('data/raw/no_prop_overlap_val.csv', 100)
    
    # Model Initialization
    model_cfg, train_cfg = get_configs()
    
    model = MultiTaskModel(config=model_cfg, embedding_dim=128)
    
    # Trainer Setup
    trainer = Trainer(model=model, config=train_cfg)

    # Training Loop
    print("\n Training (trainer.fit)")
    
    history, thresholds = trainer.fit(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_tasks=len(all_tasks),
        steps_per_epoch=200 // 8,
        optimize_thresholds=True
    )

    # Verify normalization behavior
    print("\n>>> VERIFYING: Sample-level normalization")
    test_y_true = {all_tasks[0]: tf.constant([[1.0]], dtype=tf.float32)}
    test_y_pred = {all_tasks[0]: tf.constant([[0.5]], dtype=tf.float32)}
    test_loss = trainer.loss_fn(test_y_true, test_y_pred)
    expected_loss = 0.693  # -log(0.5)
    assert abs(test_loss.numpy() - expected_loss) < 0.01, \
        f"Loss normalization failed: expected ~{expected_loss}, got {test_loss.numpy()}"
    print(f"✓ Normalization correct: {test_loss.numpy():.3f}")

    # Evaluation
    print("\n Evaluation (trainer.evaluate)")
    eval_results = trainer.evaluate(val_ds)
    print(f"Overall Metrics: {eval_results['overall_metrics']}")

    # Persistence
    print("\n Persistence (save/load)")
    trainer.save_checkpoint(save_path, thresholds=thresholds)
    
    loaded_model, loaded_thresholds = load_model(save_path)
    assert len(loaded_thresholds) == len(all_tasks), \
        f"Threshold count mismatch: expected {len(all_tasks)}, got {len(loaded_thresholds)}"
    print(f"✓ Successfully loaded model with {len(loaded_thresholds)} thresholds.")
    
    # Final Sanity Check: Prediction with loaded model
    dummy_input = tf.random.uniform((1, 128))
    preds = loaded_model(dummy_input, training=False)
    assert len(preds) == len(all_tasks), \
        f"Prediction count mismatch: expected {len(all_tasks)}, got {len(preds)}"
    print(f"✓ Prediction successful for {len(preds)} tasks.")
    
    print("\nfull test passed")

if __name__ == "__main__":
    run_test()