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
    
    Handles the complete training pipeline for multi-task learning including:
    - Multi-task loss computation: computes individual losses per task based on type,
      applies masking and weighting, and combines them for joint optimization
    - Learning rate scheduling with optional warmup
    - Early stopping on validation loss
    - Optional threshold optimization for binary/multilabel tasks on validation data
    - Distributed training support via TensorFlow strategies
    - Model evaluation with task-specific and overall metrics
    
    All components (loss function, optimizer, scheduler) can be customized by the user.
    
    Args:
        model: The MultiTaskModel to train
        config: TrainingConfig with hyperparameters
        strategy: TensorFlow distribution strategy (optional, for distributed training)
        loss_fn: Custom loss function (if None, uses built-in multi-task loss)
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
        """Build multi-task loss function with task-specific losses.
        
        This function implements the core of multi-task learning: it computes
        individual losses for each task based on its type, applies masking and
        weighting, then combines them into a single scalar loss for optimization.
        
        The loss function handles different loss types per task:
        - Binary: binary crossentropy with sigmoid outputs
        - Multiclass: categorical crossentropy with softmax outputs
        - Multilabel: binary crossentropy with sigmoid outputs (one per class)
        
        Masking: Labels with value mask_value (-1 by default) are excluded from
        both loss computation and normalization, enabling handling of missing labels.
        
        Task weighting: Each task can have importance weight applied to its loss,
        allowing prioritization of certain tasks during training.
        
        Args:
            num_tasks: Total number of output tasks
            class_weights: Per-task class weights, list of {0: weight, 1: weight} dicts
            task_weights: Relative importance weights for each task (default: all 1.0)
            mask_value: Value indicating missing labels to be excluded (default: -1)
            
        Returns:
            Loss function that takes (y_true_dict, y_pred_dict) and returns scalar loss
        """
        if class_weights is None:
            class_weights = [{0: 1.0, 1: 1.0} for _ in range(num_tasks)]
        
        if task_weights is None:
            task_weights = [1.0] * num_tasks
        
        def dict_weighted_masked_loss(y_true_dict, y_pred_dict):
            """Compute multi-task loss combining individual task losses.
            
            For each task:
            1. Compute task-specific loss based on task type (binary/multiclass/multilabel)
            2. Apply mask to exclude invalid labels
            3. Apply task weight to prioritize certain tasks
            4. Accumulate into total loss
            
            Final loss is averaged over all valid (non-masked) samples.
            """
            total_loss = 0.0
            num_valid_samples = 0
            
            # Get task names from y_true_dict (should match y_pred_dict)
            task_names = sorted(y_true_dict.keys())
            
            for task_idx, task_name in enumerate(task_names):
                y_task_true = y_true_dict[task_name]
                y_task_pred = y_pred_dict[task_name]
                
                # Get task type
                task_type = self.model.config.get_task_type(task_idx)
                
                # Create mask for valid labels
                mask = tf.cast(tf.not_equal(y_task_true, mask_value), tf.float32)
                
                # Compute appropriate loss based on task type
                if task_type == 'binary':
                    # Binary classification: single output with sigmoid
                    loss_val = tf.keras.losses.binary_crossentropy(y_task_true, y_task_pred)
                elif task_type == 'multiclass':
                    # Multiclass: convert true labels to one-hot if needed
                    y_task_true_valid = tf.cast(y_task_true, tf.int32)
                    loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y_task_true_valid,
                        logits=y_task_pred
                    ) if y_task_pred.shape[-1] > 1 else tf.keras.losses.sparse_categorical_crossentropy(
                        y_task_true, y_task_pred, from_logits=False
                    )
                elif task_type == 'multilabel':
                    # Multilabel: binary crossentropy on each class
                    loss_val = tf.keras.losses.binary_crossentropy(y_task_true, y_task_pred)
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
                
                # Apply mask
                masked_loss = loss_val * mask
                
                # Apply task weight
                weighted_loss = masked_loss * task_weights[task_idx]
                
                total_loss += tf.reduce_sum(weighted_loss)
                num_valid_samples += tf.reduce_sum(mask)
            
            # Average over valid entries
            return total_loss / (num_valid_samples + 1e-7)
        
        return dict_weighted_masked_loss
    
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
        y_true_dict: Dict[str, np.ndarray],
        y_pred_dict: Dict[str, np.ndarray],
        mask_value: int = -1
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Find optimal classification thresholds using validation data.
        
        Uses Youden's J statistic (max sensitivity + specificity) to find
        the optimal threshold for each binary/multilabel task independently.
        Multiclass tasks don't use thresholds (use argmax instead).
        
        Args:
            y_true_dict: Dictionary of true labels {task_name: [n_samples, n_classes]}
            y_pred_dict: Dictionary of predicted probabilities {task_name: [n_samples, n_classes]}
            mask_value: Value indicating missing labels
            
        Returns:
            Tuple of (thresholds_dict, f1_scores_dict)
            where task_name -> float for tasks with thresholds, None for multiclass
        """
        thresholds = {}
        f1_scores = {}
        
        task_names = sorted(y_true_dict.keys())
        
        for task_name in task_names:
            task_idx = list(self.model.task_names).index(task_name) if self.model.task_names else int(task_name.split('_')[-1])
            task_type = self.model.config.get_task_type(task_idx)
            
            y_task_true = y_true_dict[task_name]
            y_task_pred = y_pred_dict[task_name]
            
            # For binary/multilabel, flatten to 1D if needed
            if task_type in ['binary', 'multilabel'] and y_task_pred.ndim > 1:
                y_task_pred = y_task_pred[:, 0]
                y_task_true = y_task_true[:, 0] if y_task_true.ndim > 1 else y_task_true
            
            # Filter out masked values
            valid_mask = y_task_true != mask_value
            y_task_true = y_task_true[valid_mask]
            y_task_pred = y_task_pred[valid_mask]
            
            if len(y_task_true) == 0:
                # No valid samples for this task
                thresholds[task_name] = 0.5
                f1_scores[task_name] = 0.0
                continue
            
            if task_type == 'multiclass':
                # Multiclass doesn't use thresholds - use argmax
                thresholds[task_name] = None
                f1_scores[task_name] = 0.0
            else:
                # Binary or multilabel: find optimal threshold
                fpr, tpr, thresh = roc_curve(y_task_true, y_task_pred)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresh[optimal_idx]
                
                # Calculate F1 with optimal threshold
                y_task_pred_binary = (y_task_pred > optimal_threshold).astype(int)
                f1 = f1_score(y_task_true, y_task_pred_binary, average='weighted', zero_division=0)
                
                thresholds[task_name] = float(optimal_threshold)
                f1_scores[task_name] = float(f1)
        
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
    ) -> Union[tf.keras.callbacks.History, Tuple[tf.keras.callbacks.History, Dict[str, float]]]:
        """Train the multi-task model.
        
        The training process:
        1. Compiles model with custom multi-task loss function
        2. Trains on provided dataset with optional early stopping
        3. Optionally optimizes classification thresholds on validation data
        
        The multi-task loss function computes individual losses for each task,
        applies masking and weighting, and combines them for joint optimization.
        
        Args:
            train_dataset: Training TensorFlow Dataset yielding (inputs, labels_dict)
            val_dataset: Validation TensorFlow Dataset yielding (inputs, labels_dict)
            num_tasks: Total number of output tasks
            steps_per_epoch: Number of steps per epoch (auto-calculated if None)
            validation_steps: Number of validation steps (auto-calculated if None)
            class_weights: Optional per-task class weights for loss function
            task_weights: Optional task importance weights for relative task prioritization
            callbacks: Additional Keras callbacks
            optimize_thresholds: Whether to find optimal classification thresholds
                                on validation data (only for binary/multilabel tasks)
            
        Returns:
            If optimize_thresholds=False: training history object
            If optimize_thresholds=True: tuple of (training history, thresholds dict)
                where thresholds dict maps task_name -> optimal_threshold (or None for multiclass)
        """
        with self.strategy.scope():
            # Build or use custom loss function
            # This function encapsulates true multi-task learning: computing and combining
            # individual task losses
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
            
            # Compile model with multi-task loss function
            # The loss function internally computes task-specific losses and combines them
            # for joint optimization, implementing the core of multi-task learning
            self.model.compile(
                optimizer=optimizer,
                loss=loss_fn
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
                y_val_pred_dict = self.model.predict(val_dataset, verbose=0)
                
                # Extract true labels from validation dataset as dictionary
                y_val_true_dict = {}
                task_names = sorted(y_val_pred_dict.keys())
                
                for task_name in task_names:
                    y_val_true_dict[task_name] = []
                
                for _, labels_dict in val_dataset:
                    for task_name in task_names:
                        y_val_true_dict[task_name].append(labels_dict[task_name].numpy())
                
                for task_name in task_names:
                    y_val_true_dict[task_name] = np.concatenate(y_val_true_dict[task_name], axis=0)
                
                self.optimal_thresholds, f1_scores = self._find_optimal_thresholds(
                    y_val_true_dict, y_val_pred_dict
                )
                
                valid_f1_scores = [f1 for f1 in f1_scores.values() if f1 is not None and f1 > 0]
                if valid_f1_scores:
                    print(f"Average validation F1: {np.mean(valid_f1_scores):.4f}")
                return self.history, self.optimal_thresholds
            
        return self.history
    
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        thresholds: Optional[Dict[str, float]] = None,
        metrics: Optional[List[str]] = None,
        mask_value: int = -1
    ) -> Dict[str, Any]:
        """Evaluate the model on a dataset.
        
        Computes specified metrics on the provided dataset. Handles mixed task types
        (binary, multiclass, multilabel). Can use stored thresholds from training
        (if optimize_thresholds=True was used in fit()), custom thresholds, or
        compute probability-based metrics only.
        
        Threshold Usage:
            - If thresholds is None and self.optimal_thresholds exists: uses stored thresholds
            - If thresholds is provided: uses those thresholds (dict format)
            - If neither: only computes probability-based metrics (AUC, AP)
            - Multiclass tasks ignore thresholds (use argmax)
        
        Args:
            dataset: TensorFlow Dataset to evaluate on (yields (inputs, labels_dict))
            thresholds: Optional dict of thresholds per task {task_name: threshold_value}
                       If None, will use self.optimal_thresholds if available
            metrics: List of metric names to compute. Options:
                     Binary metrics (require thresholds): 'accuracy', 'precision', 'recall', 'f1'
                     Probability metrics (no threshold): 'auc', 'average_precision'
                     If None, computes all applicable metrics based on threshold availability
            mask_value: Value indicating missing labels to mask out
            
        Returns:
            Dictionary containing:
                - 'probabilities_dict': Dict of predicted probabilities {task_name: [n_samples, n_classes]}
                - 'predictions_dict': Dict of binary predictions (if thresholds provided)
                - 'task_metrics': Per-task metrics dict {task_name: {metric: value}}
                - 'overall_metrics': Overall metrics dict {metric: average_value}
                - 'thresholds_used': Dict of thresholds used (if applicable)
                
        Example:
            >>> # After training with optimize_thresholds=True
            >>> results = trainer.evaluate(test_dataset)  # Uses stored thresholds
            >>> 
            >>> # With custom thresholds
            >>> results = trainer.evaluate(test_dataset, thresholds={'task_0': 0.6, ...})
            >>> 
            >>> # Probability-based metrics only
            >>> results = trainer.evaluate(test_dataset, metrics=['auc', 'average_precision'])
        """
        # Get predictions
        print("Generating predictions...")
        y_pred_dict = self.model.predict(dataset, verbose=0)
        
        # Extract true labels as dictionary
        y_true_dict = {}
        task_names = sorted(y_pred_dict.keys())
        
        for task_name in task_names:
            y_true_dict[task_name] = []
        
        for _, labels_dict in dataset:
            for task_name in task_names:
                y_true_dict[task_name].append(labels_dict[task_name].numpy())
        
        for task_name in task_names:
            y_true_dict[task_name] = np.concatenate(y_true_dict[task_name], axis=0)
        
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
        
        # Apply thresholds if available to create binary predictions
        y_pred_binary_dict = None
        if use_thresholds is not None:
            y_pred_binary_dict = {}
            for task_name in task_names:
                task_idx = list(self.model.task_names).index(task_name) if self.model.task_names else task_names.index(task_name)
                task_type = self.model.config.get_task_type(task_idx)
                
                if task_type == 'multiclass':
                    # Multiclass: use argmax
                    y_pred_binary_dict[task_name] = np.argmax(y_pred_dict[task_name], axis=1)
                else:
                    # Binary/multilabel: use threshold
                    threshold = use_thresholds.get(task_name, 0.5)
                    y_pred_binary_dict[task_name] = (y_pred_dict[task_name][:, 0] > threshold).astype(int)
        
        # Compute metrics per task
        task_metrics = {}
        overall_metrics = {metric: [] for metric in metrics}
        
        for task_name in task_names:
            task_idx = list(self.model.task_names).index(task_name) if self.model.task_names else task_names.index(task_name)
            task_type = self.model.config.get_task_type(task_idx)
            
            y_task_true = y_true_dict[task_name]
            y_task_pred_probs = y_pred_dict[task_name]
            
            # Filter out masked values
            valid_mask = y_task_true != mask_value
            if y_task_true.ndim > 1:
                # For multiclass, mask first dimension
                valid_mask = valid_mask[:, 0] if valid_mask.shape[1] > 0 else np.ones(y_task_true.shape[0], dtype=bool)
            
            y_task_true_valid = y_task_true[valid_mask]
            y_task_pred_probs_valid = y_task_pred_probs[valid_mask]
            
            if len(y_task_true_valid) == 0:
                # No valid samples for this task
                task_metrics[task_name] = {metric: None for metric in metrics}
                continue
            
            task_results = {}
            
            # Compute binary metrics if thresholds available
            if use_thresholds is not None and task_type != 'multiclass':
                y_task_pred_binary = y_pred_binary_dict[task_name][valid_mask]
                y_task_true_binary = y_task_true_valid[:, 0] if y_task_true_valid.ndim > 1 else y_task_true_valid
                
                if 'accuracy' in metrics:
                    acc = accuracy_score(y_task_true_binary, y_task_pred_binary)
                    task_results['accuracy'] = float(acc)
                    overall_metrics['accuracy'].append(acc)
                
                if 'precision' in metrics:
                    prec = precision_score(y_task_true_binary, y_task_pred_binary, zero_division=0)
                    task_results['precision'] = float(prec)
                    overall_metrics['precision'].append(prec)
                
                if 'recall' in metrics:
                    rec = recall_score(y_task_true_binary, y_task_pred_binary, zero_division=0)
                    task_results['recall'] = float(rec)
                    overall_metrics['recall'].append(rec)
                
                if 'f1' in metrics:
                    f1 = f1_score(y_task_true_binary, y_task_pred_binary, zero_division=0)
                    task_results['f1'] = float(f1)
                    overall_metrics['f1'].append(f1)
            elif task_type == 'multiclass' and use_thresholds is not None:
                # Multiclass with argmax
                y_task_pred_classes = y_pred_binary_dict[task_name][valid_mask]
                y_task_true_classes = y_task_true_valid[:, 0] if y_task_true_valid.ndim > 1 else y_task_true_valid
                
                if 'accuracy' in metrics:
                    acc = accuracy_score(y_task_true_classes, y_task_pred_classes)
                    task_results['accuracy'] = float(acc)
                    overall_metrics['accuracy'].append(acc)
                
                if 'f1' in metrics:
                    f1 = f1_score(y_task_true_classes, y_task_pred_classes, average='weighted', zero_division=0)
                    task_results['f1'] = float(f1)
                    overall_metrics['f1'].append(f1)
            
            # Compute probability-based metrics
            if 'auc' in metrics and task_type in ['binary', 'multilabel']:
                try:
                    y_probs_for_auc = y_task_pred_probs_valid[:, 0] if y_task_pred_probs_valid.ndim > 1 else y_task_pred_probs_valid
                    y_true_for_auc = y_task_true_valid[:, 0] if y_task_true_valid.ndim > 1 else y_task_true_valid
                    auc = roc_auc_score(y_true_for_auc, y_probs_for_auc)
                    task_results['auc'] = float(auc)
                    overall_metrics['auc'].append(auc)
                except ValueError:
                    task_results['auc'] = None
            
            if 'average_precision' in metrics and task_type in ['binary', 'multilabel']:
                try:
                    y_probs_for_ap = y_task_pred_probs_valid[:, 0] if y_task_pred_probs_valid.ndim > 1 else y_task_pred_probs_valid
                    y_true_for_ap = y_task_true_valid[:, 0] if y_task_true_valid.ndim > 1 else y_task_true_valid
                    ap = average_precision_score(y_true_for_ap, y_probs_for_ap)
                    task_results['average_precision'] = float(ap)
                    overall_metrics['average_precision'].append(ap)
                except ValueError:
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
            'probabilities_dict': y_pred_dict,
            'task_metrics': task_metrics,
            'overall_metrics': overall_metrics
        }
        
        if y_pred_binary_dict is not None:
            results['predictions_dict'] = y_pred_binary_dict
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

