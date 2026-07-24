"""Example of distributed training with the multitask library.

This script demonstrates:
1. Cross-platform distributed training setup
2. Using MirroredStrategy with multiple GPUs
3. Handling Apple Silicon and other platform-specific quirks
4. Verifying dict outputs work correctly in distributed context
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from multitask import MultiTaskModel, Trainer
from multitask.config import EncoderConfig, EncoderInputType, ModelConfig, TrainingConfig
from multitask.utils.distributed import (
    get_distribution_strategy,
    setup_gpu_memory_growth,
    print_device_info,
    check_distributed_compatibility
)

# Task column names (order matches flat task heads in task_structure [[5]])
TASK_NAMES = [
    'binary_task',
    'multiclass_task',
    'multilabel_task_1',
    'multilabel_task_2',
    'multilabel_task_3',
]


def create_synthetic_data(n_samples: int = 1000):
    """Create synthetic data for testing distributed training."""
    # Random embeddings (simulating pre-computed embeddings)
    embeddings = np.random.randn(n_samples, 768).astype(np.float32)
    
    # Create mixed task types
    data = pd.DataFrame({
        'embedding': list(embeddings),
        'binary_task': np.random.randint(0, 2, n_samples),  # Binary classification
        'multiclass_task': np.random.randint(0, 5, n_samples),  # 5-class classification
        'multilabel_task_1': np.random.randint(0, 2, n_samples),  # Scalar 0/1 (binary head each)
        'multilabel_task_2': np.random.randint(0, 2, n_samples),
        'multilabel_task_3': np.random.randint(0, 2, n_samples),
    })
    
    return data


def dataframe_to_tf_dataset(df: pd.DataFrame, batch_size: int, shuffle: bool = False):
    """Build (embeddings, labels_dict) batches for Trainer.fit."""
    embeddings = np.stack(df['embedding'].values)
    labels = {name: df[name].values.astype(np.int32) for name in TASK_NAMES}
    ds = tf.data.Dataset.from_tensor_slices((embeddings, labels))
    if shuffle:
        ds = ds.shuffle(min(len(df), 1000))
    return ds.batch(batch_size)


def main():
    """Run distributed training test."""
    print("\n" + "=" * 70)
    print("DISTRIBUTED TRAINING VERIFICATION")
    print("=" * 70 + "\n")
    
    # Step 1: Check platform compatibility
    print("Step 1: Checking platform compatibility...")
    compat = check_distributed_compatibility()
    print(f"Platform: {compat['platform']} ({compat['architecture']})")
    print(f"GPUs available: {compat['num_gpus']}")
    print(f"Recommendation: {compat['recommendation']}")
    if compat['notes']:
        for note in compat['notes']:
            print(f"   {note}")
    print()
    
    # Step 2: Print detailed device info
    print("Step 2: Device information...")
    print_device_info()
    
    # Step 3: Setup GPU memory growth (if applicable)
    print("Step 3: Setting up GPU memory management...")
    setup_gpu_memory_growth()
    print()
    
    # Step 4: Get distribution strategy
    print("Step 4: Selecting distribution strategy...")
    strategy, info = get_distribution_strategy('auto', verbose=True)
    
    # Step 5: Create synthetic data
    print("Step 5: Creating synthetic data...")
    train_data = create_synthetic_data(n_samples=1000)
    val_data = create_synthetic_data(n_samples=200)
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print()
    
    # Step 6: Configure model
    print("Step 6: Configuring model...")
    # Flat five-way split: binary (2 classes), multiclass (5 classes), three binary columns
    model_config = ModelConfig(
        task_structure=[[5]],
        task_names=TASK_NAMES,
        task_types=['binary', 'multiclass', 'binary', 'binary', 'binary'],
        num_classes_per_task=[2, 5, 2, 2, 2],
        shared_layer_sizes=[256, 128],
        default_layer_sizes=[64],
        encoder_config=EncoderConfig(
            encoder_identifier=None,
            encoder_input_type=EncoderInputType.PRECOMPUTED,
            embedding_dim=768,
        ),
    )

    training_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        epochs=5,
        early_stopping_patience=3,
    )

    print(f"Tasks configured: {model_config.task_names}")
    print("Mixed task types: binary, multiclass, binary (three additional binary heads)")
    print()

    # Step 7: Create model and trainer
    # Do NOT wrap in strategy.scope() — Trainer.fit() enters the scope internally
    # during compilation. Pre-scoping causes Keras to mis-route dict-output tensors
    # under single-GPU (OneDevice) strategies.
    print("Step 7: Creating model and trainer...")
    model = MultiTaskModel(model_config)
    trainer = Trainer(
        model=model,
        config=training_config,
        strategy=strategy,
    )
    print("Model and trainer created")
    print()
    
    train_ds = dataframe_to_tf_dataset(train_data, training_config.batch_size, shuffle=True)
    val_ds = dataframe_to_tf_dataset(val_data, training_config.batch_size, shuffle=False)
    
    # Step 8: Train model
    print("Step 8: Training model...")
    print("-" * 70)
    try:
        history, thresholds = trainer.fit(
            train_ds,
            val_ds,
            num_tasks=len(TASK_NAMES),
            optimize_thresholds=True,
        )
        print("-" * 70)
        print("Training completed successfully!")
        print()
        
        # Step 9: Verify outputs
        print("Step 9: Verifying model outputs...")
        sample_X = np.stack(val_data.head(10)['embedding'].values)
        predictions = model.predict(sample_X)
        
        # Check that predictions is a dictionary
        assert isinstance(predictions, dict), "Predictions should be a dictionary"
        print(f"Predictions returned as dictionary: {list(predictions.keys())}")
        
        # Check each task output
        for task_name, task_preds in predictions.items():
            assert isinstance(task_preds, np.ndarray), f"{task_name} should be numpy array"
            assert len(task_preds) == 10, f"{task_name} should have 10 predictions"
            print(f"  {task_name}: shape {task_preds.shape}, dtype {task_preds.dtype}")
        
        print()
        
        # Step 10: Verify thresholds (binary tasks; multiclass uses argmax)
        print("Step 10: Verifying optimized thresholds...")
        assert isinstance(thresholds, dict), "Thresholds should be a dictionary"
        print(f"Thresholds: {thresholds}")
        
        # Binary and multilabel tasks should have thresholds, multiclass should be None
        assert thresholds['binary_task'] is not None, "Binary task should have threshold"
        assert thresholds['multiclass_task'] is None, "Multiclass task should have None threshold"
        assert thresholds['multilabel_task_1'] is not None, "Binary head should have threshold"
        assert thresholds['multilabel_task_2'] is not None, "Binary head should have threshold"
        assert thresholds['multilabel_task_3'] is not None, "Binary head should have threshold"
        print("  Binary task threshold:", thresholds['binary_task'])
        print("  Multiclass task threshold:", thresholds['multiclass_task'])
        print("  Extra binary head thresholds:", {k: v for k, v in thresholds.items() if 'multilabel' in k})
        print()
        
        # Step 11: Test evaluation
        print("Step 11: Testing evaluation...")
        metrics = trainer.evaluate(val_ds)
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        overall = metrics.get('overall_metrics', {})
        print(f"Evaluation overall metrics: {list(overall.keys())}")
        for metric_name, value in overall.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
        print()
        
        # Step 12: Test saving/loading with distributed training
        print("Step 12: Testing save/load in distributed context...")
        save_path = '/tmp/distributed_test_model'
        model.save(save_path, thresholds=thresholds)
        print(f"Model saved to {save_path}")
        
        from multitask import load_model
        loaded_model, loaded_thresholds = load_model(save_path)
        print(f"Model loaded from {save_path}")
        
        # Verify loaded model produces same outputs
        loaded_predictions = loaded_model.predict(sample_X)
        for task_name in predictions.keys():
            diff = np.abs(predictions[task_name] - loaded_predictions[task_name]).max()
            assert diff < 1e-5, f"Loaded model predictions differ for {task_name}"
        print("Loaded model produces identical predictions")
        print()
        
        # Success summary
        print("=" * 70)
        print("✅ DISTRIBUTED TRAINING VERIFICATION COMPLETE")
        print("=" * 70)
        print("\nAll tests passed:")
        print("  Cross-platform strategy selection")
        print("  Model creation with distributed strategy")
        print("  Training with distributed strategy")
        print("  Dictionary outputs working correctly")
        print("  Multi-GPU coordination (if applicable)")
        print("  Threshold optimization")
        print("  Evaluation with distributed data")
        print("  Model saving/loading in distributed context")
        print("\nDistributed training is working correctly! 🎉\n")
        
    except Exception as e:
        print("=" * 70)
        print("DISTRIBUTED TRAINING VERIFICATION FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check that TensorFlow is installed correctly")
        print("  2. Verify GPU drivers are up to date (if using GPUs)")
        print("  3. On Apple Silicon, ensure tensorflow-metal is installed")
        print("  4. Try strategy_type='cpu' to test CPU-only training")
        print("  5. Check TensorFlow version compatibility with your hardware")
        print()
        raise


if __name__ == '__main__':
    main()
