"""
Trainer class for multi-task models.

This module provides the Trainer class that handles training loops, loss computation,
optimization, and threshold selection for multi-task learning models.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
from official.nlp import optimization
from sklearn.metrics import (
    roc_curve, f1_score, precision_score, recall_score, 
    roc_auc_score, average_precision_score, accuracy_score
)

from multitask.config import TrainingConfig
from multitask.models.hierarchical import MultiTaskModel


class Trainer:
    """Trainer for multi-task learning models.
    
    Handles the complete training pipeline including:
    - Loss computation with task weighting and masking
    - Learning rate scheduling with warmup
    - Early stopping
    - Optional threshold optimization on validation data
    - Distributed training support
    
    All components (loss function, optimizer, scheduler) can be customized by the user.
    
    Args:
        model: The MultiTaskModel to train
        config: TrainingConfig with hyperparameters
        strategy: TensorFlow distribution strategy (optional, for distributed training)
        loss_fn: Custom loss function (if None, uses default weighted masked loss)
        optimizer: Custom optimizer (if None, builds from config)
        
    Example:
        >>> config = TrainingConfig(epochs=20, batch_size=32, learning_rate=3e-4)
        >>> trainer = Trainer(model, config)
        >>> history = trainer.fit(train_dataset, val_dataset, num_tasks=10)
        >>> # With threshold optimization:
        >>> history, thresholds = trainer.fit(train_dataset, val_dataset, 
        ...                                    num_tasks=10, optimize_thresholds=True)
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        config: Optional[TrainingConfig] = None,
        strategy: Optional[tf.distribute.Strategy] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.strategy = strategy or tf.distribute.get_strategy()
        self.custom_loss_fn = loss_fn
        self.custom_optimizer = optimizer
        
        # Will be set during training
        self.history = None
        self.optimal_thresholds = None
        
    def _build_loss_function(
        self,
        num_tasks: int,
        class_weights: Optional[List[Dict[int, float]]] = None,
        task_weights: Optional[List[float]] = None,
        mask_value: int = -1
    ) -> Callable:
        """Build weighted and masked loss function.
        
        Args:
            num_tasks: Total number of output tasks
            class_weights: Per-task class weights, list of {0: weight, 1: weight} dicts
            task_weights: Relative importance weights for each task
            mask_value: Value indicating missing labels to be masked
            
        Returns:
            Loss function compatible with model.compile()
        """
        if class_weights is None:
            class_weights = [{0: 1.0, 1: 1.0} for _ in range(num_tasks)]
        
        if task_weights is None:
            task_weights = [1.0] * num_tasks
        
        def weighted_masked_loss(y_true, y_pred):
            """Compute weighted binary crossentropy with masking."""
            # Create mask for valid labels
            mask = tf.cast(tf.not_equal(y_true, mask_value), tf.float32)
            
            # Compute binary crossentropy
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            
            # Apply mask
            masked_loss = bce * mask
            
            # Apply class weights (simplified - can be enhanced)
            # For now, just apply task weights
            task_weight_tensor = tf.constant(task_weights, dtype=tf.float32)
            task_weight_tensor = tf.reshape(task_weight_tensor, [1, num_tasks])
            
            weighted_loss = masked_loss * task_weight_tensor
            
            # Average over valid entries
            return tf.reduce_sum(weighted_loss) / (tf.reduce_sum(mask) + 1e-7)
        
        return weighted_masked_loss
    
    def _build_lr_schedule(
        self,
        steps_per_epoch: int
    ) -> Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Build learning rate schedule.
        
        Args:
            steps_per_epoch: Number of training steps per epoch
            
        Returns:
            Learning rate or learning rate schedule
        """
        if not self.config.use_lr_schedule:
            return self.config.learning_rate
        
        num_train_steps = steps_per_epoch * self.config.epochs
        
        # Build decay schedule
        if self.config.lr_schedule_type == 'polynomial_decay':
            schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=num_train_steps,
                end_learning_rate=0.0
            )
        elif self.config.lr_schedule_type == 'exponential_decay':
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=0.96
            )
        elif self.config.lr_schedule_type == 'cosine_decay':
            schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.config.learning_rate,
                decay_steps=num_train_steps
            )
        elif self.config.lr_schedule_type == 'constant':
            return self.config.learning_rate
        else:
            raise ValueError(f"Unknown lr_schedule_type: {self.config.lr_schedule_type}")
        
        # Add warmup if enabled
        if self.config.use_warmup:
            num_warmup_steps = num_train_steps // self.config.warmup_percent
            schedule = optimization.WarmUp(
                initial_learning_rate=self.config.learning_rate,
                decay_schedule_fn=schedule,
                warmup_steps=num_warmup_steps
            )
        
        return schedule
    
    def _build_optimizer(
        self,
        steps_per_epoch: int
    ) -> tf.keras.optimizers.Optimizer:
        """Build optimizer with learning rate schedule.
        
        Args:
            steps_per_epoch: Number of training steps per epoch
            
        Returns:
            Configured optimizer
        """
        lr_schedule = self._build_lr_schedule(steps_per_epoch)
        
        # Build optimizer based on type
        if self.config.optimizer_type.lower() == 'adamw':
            optimizer = optimization.AdamWeightDecay(
                learning_rate=lr_schedule,
                weight_decay_rate=self.config.weight_decay_rate
            )
        elif self.config.optimizer_type.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {self.config.optimizer_type}")
        
        return optimizer
    
    def _find_optimal_thresholds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask_value: int = -1
    ) -> Tuple[List[float], List[float]]:
        """Find optimal classification thresholds using validation data.
        
        Uses Youden's J statistic (max sensitivity + specificity) to find
        the optimal threshold for each task independently.
        
        Args:
            y_true: True labels, shape [n_samples, n_tasks]
            y_pred: Predicted probabilities, shape [n_samples, n_tasks]
            mask_value: Value indicating missing labels
            
        Returns:
            Tuple of (thresholds, f1_scores) - one per task
        """
        thresholds = []
        f1_scores = []
        
        for task_idx in range(y_true.shape[1]):
            y_task_true = y_true[:, task_idx]
            y_task_pred = y_pred[:, task_idx]
            
            # Filter out masked values
            valid_mask = y_task_true != mask_value
            y_task_true = y_task_true[valid_mask]
            y_task_pred = y_task_pred[valid_mask]
            
            if len(y_task_true) == 0:
                # No valid samples for this task
                thresholds.append(0.5)
                f1_scores.append(0.0)
                continue
            
            # Find optimal threshold using Youden's J statistic
            fpr, tpr, thresh = roc_curve(y_task_true, y_task_pred)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresh[optimal_idx]
            
            # Calculate F1 with optimal threshold
            y_task_pred_binary = (y_task_pred > optimal_threshold).astype(int)
            f1 = f1_score(y_task_true, y_task_pred_binary, average='weighted')
            
            thresholds.append(float(optimal_threshold))
            f1_scores.append(float(f1))
        
        return thresholds, f1_scores
    
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        num_tasks: int,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        class_weights: Optional[List[Dict[int, float]]] = None,
        task_weights: Optional[List[float]] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        optimize_thresholds: bool = False
    ) -> Union[tf.keras.callbacks.History, Tuple[tf.keras.callbacks.History, List[float]]]:
        """Train the model.
        
        Args:
            train_dataset: Training dataset (TensorFlow Dataset)
            val_dataset: Validation dataset (TensorFlow Dataset)
            num_tasks: Total number of output tasks
            steps_per_epoch: Number of steps per epoch (auto-calculated if None)
            validation_steps: Number of validation steps (auto-calculated if None)
            class_weights: Optional per-task class weights for loss function
            task_weights: Optional task importance weights for loss function
            callbacks: Additional Keras callbacks
            optimize_thresholds: Whether to optimize classification thresholds on validation data (default: False)
            
        Returns:
            If optimize_thresholds=False: training history
            If optimize_thresholds=True: tuple of (training history, optimal thresholds per task)
        """
        with self.strategy.scope():
            # Build or use custom loss function
            if self.custom_loss_fn is not None:
                loss_fn = self.custom_loss_fn
            else:
                loss_fn = self._build_loss_function(
                    num_tasks=num_tasks,
                    class_weights=class_weights,
                    task_weights=task_weights
                )
            
            # Determine steps per epoch
            if steps_per_epoch is None:
                # Try to infer from dataset cardinality
                steps_per_epoch = int(train_dataset.cardinality())
                if steps_per_epoch < 0:
                    raise ValueError("Could not determine steps_per_epoch automatically. Please provide it explicitly.")
            
            # Build or use custom optimizer
            if self.custom_optimizer is not None:
                optimizer = self.custom_optimizer
            else:
                optimizer = self._build_optimizer(steps_per_epoch)
            
            # Define metrics
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
            
            # Compile model
            self.model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics
            )
            
            # Setup callbacks
            callback_list = callbacks or []
            
            if self.config.early_stopping:
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=self.config.early_stopping_min_delta,
                    patience=self.config.early_stopping_patience,
                    mode='min',
                    restore_best_weights=True,
                    verbose=1
                )
                callback_list.append(early_stop)
            
            # Train model
            print(f"\nTraining with config:")
            print(f"  Epochs: {self.config.epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Learning rate: {self.config.learning_rate}")
            print(f"  Weight decay: {self.config.weight_decay_rate}")
            print(f"  Dropout: {self.config.dropout}")
            print(f"  Steps per epoch: {steps_per_epoch}\n")
            
            self.history = self.model.fit(
                train_dataset,
                epochs=self.config.epochs,
                validation_data=val_dataset,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callback_list,
                verbose=1
            )
            
            # Optionally find optimal thresholds on validation data
            if optimize_thresholds:
                print("\nFinding optimal classification thresholds...")
                y_val_pred = self.model.predict(val_dataset, verbose=0)
                
                # Extract true labels from validation dataset
                y_val_true = []
                for _, labels in val_dataset:
                    y_val_true.append(labels.numpy())
                y_val_true = np.concatenate(y_val_true, axis=0)
                
                self.optimal_thresholds, f1_scores = self._find_optimal_thresholds(
                    y_val_true, y_val_pred
                )
                
                print(f"Average validation F1: {np.mean(f1_scores):.4f}")
                return self.history, self.optimal_thresholds
            
        return self.history
    
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        thresholds: Optional[List[float]] = None,
        metrics: Optional[List[str]] = None,
        mask_value: int = -1
    ) -> Dict[str, Any]:
        """Evaluate the model on a dataset.
        
        Computes specified metrics on the provided dataset. Can use stored thresholds
        from training (if optimize_thresholds=True was used in fit()), custom thresholds,
        or compute probability-based metrics only.
        
        Threshold Usage:
            - If thresholds is None and self.optimal_thresholds exists: uses stored thresholds
            - If thresholds is provided: uses those thresholds
            - If neither: only computes probability-based metrics (AUC, AP)
        
        Args:
            dataset: TensorFlow Dataset to evaluate on
            thresholds: Optional list of thresholds per task (length = num_tasks)
                       If None, will use self.optimal_thresholds if available
            metrics: List of metric names to compute. Options:
                     Binary metrics (require thresholds): 'accuracy', 'precision', 'recall', 'f1'
                     Probability metrics (no threshold): 'auc', 'average_precision'
                     If None, computes all applicable metrics based on threshold availability
            mask_value: Value indicating missing labels to mask out
            
        Returns:
            Dictionary containing:
                - 'probabilities': Array of predicted probabilities [n_samples, n_tasks]
                - 'predictions': Array of binary predictions [n_samples, n_tasks] (if thresholds provided)
                - 'task_metrics': Per-task metrics dict {task_name: {metric: value}}
                - 'overall_metrics': Overall metrics dict {metric: average_value}
                - 'thresholds_used': List of thresholds used (if applicable)
                
        Example:
            >>> # After training with optimize_thresholds=True
            >>> results = trainer.evaluate(test_dataset)  # Uses stored thresholds
            >>> 
            >>> # With custom thresholds
            >>> results = trainer.evaluate(test_dataset, thresholds=[0.5]*10)
            >>> 
            >>> # Probability-based metrics only
            >>> results = trainer.evaluate(test_dataset, metrics=['auc', 'average_precision'])
        """
        # Get predictions
        print("Generating predictions...")
        y_pred = self.model.predict(dataset, verbose=0)
        
        # Extract true labels
        y_true = []
        for _, labels in dataset:
            y_true.append(labels.numpy())
        y_true = np.concatenate(y_true, axis=0)
        
        num_tasks = y_true.shape[1]
        
        # Determine which thresholds to use
        use_thresholds = thresholds if thresholds is not None else self.optimal_thresholds
        
        # Determine which metrics to compute
        if metrics is None:
            if use_thresholds is not None:
                # Can compute all metrics
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision']
            else:
                # Only probability-based metrics
                metrics = ['auc', 'average_precision']
        
        # Validate metrics
        binary_metrics = {'accuracy', 'precision', 'recall', 'f1'}
        prob_metrics = {'auc', 'average_precision'}
        valid_metrics = binary_metrics | prob_metrics
        
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Unknown metric: {metric}. Valid options: {valid_metrics}")
            if metric in binary_metrics and use_thresholds is None:
                raise ValueError(
                    f"Metric '{metric}' requires thresholds. Either provide thresholds parameter "
                    f"or train with optimize_thresholds=True."
                )
        
        # Validate threshold length
        if use_thresholds is not None and len(use_thresholds) != num_tasks:
            raise ValueError(
                f"Number of thresholds ({len(use_thresholds)}) must match "
                f"number of tasks ({num_tasks})"
            )
        
        # Apply thresholds if available
        y_pred_binary = None
        if use_thresholds is not None:
            y_pred_binary = np.zeros_like(y_pred)
            for task_idx in range(num_tasks):
                y_pred_binary[:, task_idx] = (y_pred[:, task_idx] > use_thresholds[task_idx]).astype(int)
        
        # Get task names if available
        task_names = None
        if self.model.task_names is not None:
            task_names = self.model.task_names
        else:
            task_names = [f'task_{i}' for i in range(num_tasks)]
        
        # Compute metrics per task
        task_metrics = {}
        overall_metrics = {metric: [] for metric in metrics}
        
        for task_idx in range(num_tasks):
            task_name = task_names[task_idx]
            y_task_true = y_true[:, task_idx]
            y_task_pred_probs = y_pred[:, task_idx]
            
            # Filter out masked values
            valid_mask = y_task_true != mask_value
            y_task_true_valid = y_task_true[valid_mask]
            y_task_pred_probs_valid = y_task_pred_probs[valid_mask]
            
            if len(y_task_true_valid) == 0:
                # No valid samples for this task
                task_metrics[task_name] = {metric: None for metric in metrics}
                continue
            
            task_results = {}
            
            # Compute binary metrics if thresholds available
            if use_thresholds is not None:
                y_task_pred_binary = y_pred_binary[:, task_idx][valid_mask]
                
                if 'accuracy' in metrics:
                    acc = accuracy_score(y_task_true_valid, y_task_pred_binary)
                    task_results['accuracy'] = float(acc)
                    overall_metrics['accuracy'].append(acc)
                
                if 'precision' in metrics:
                    prec = precision_score(y_task_true_valid, y_task_pred_binary, zero_division=0)
                    task_results['precision'] = float(prec)
                    overall_metrics['precision'].append(prec)
                
                if 'recall' in metrics:
                    rec = recall_score(y_task_true_valid, y_task_pred_binary, zero_division=0)
                    task_results['recall'] = float(rec)
                    overall_metrics['recall'].append(rec)
                
                if 'f1' in metrics:
                    f1 = f1_score(y_task_true_valid, y_task_pred_binary, zero_division=0)
                    task_results['f1'] = float(f1)
                    overall_metrics['f1'].append(f1)
            
            # Compute probability-based metrics
            if 'auc' in metrics:
                try:
                    auc = roc_auc_score(y_task_true_valid, y_task_pred_probs_valid)
                    task_results['auc'] = float(auc)
                    overall_metrics['auc'].append(auc)
                except ValueError:
                    # Only one class present
                    task_results['auc'] = None
            
            if 'average_precision' in metrics:
                try:
                    ap = average_precision_score(y_task_true_valid, y_task_pred_probs_valid)
                    task_results['average_precision'] = float(ap)
                    overall_metrics['average_precision'].append(ap)
                except ValueError:
                    # Only one class present
                    task_results['average_precision'] = None
            
            task_metrics[task_name] = task_results
        
        # Average overall metrics
        for metric in overall_metrics:
            valid_scores = [s for s in overall_metrics[metric] if s is not None]
            if valid_scores:
                overall_metrics[metric] = float(np.mean(valid_scores))
            else:
                overall_metrics[metric] = None
        
        # Build results dictionary
        results = {
            'probabilities': y_pred,
            'task_metrics': task_metrics,
            'overall_metrics': overall_metrics
        }
        
        if y_pred_binary is not None:
            results['predictions'] = y_pred_binary
            results['thresholds_used'] = use_thresholds
        
        # Print summary
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for metric, value in overall_metrics.items():
            if value is not None:
                print(f"{metric.upper():20s}: {value:.4f}")
            else:
                print(f"{metric.upper():20s}: N/A")
        print("=" * 60 + "\n")
        
        return results

