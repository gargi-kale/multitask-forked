# Output Pools

Aggregate predictions across related tasks with no additional loss terms or trainable parameters.

## What Are Output Pools?

An output pool combines predictions from multiple tasks into a single derived output.

Pools are computed during the forward pass but have no trainable parameters and no loss term. The model trains on individual tasks exactly as before (with only a small extra forward-pass compute cost for deriving pooled outputs).

## Quick Start
```python
from multitask.config import ModelConfig

config = ModelConfig(
    task_structure=[[2], [3, 2]],
    task_names=["technique_a", "technique_b", "technique_c", "stance", "sentiment"],
    output_pools={
        "any_technique": {
            "tasks": ["technique_a", "technique_b", "technique_c"],
            "strategy": "max",
        },
    },
)
```
That's it. Everything downstream handles pools automatically.

```python
from multitask.models.hierarchical import MultiTaskModel
from multitask.trainers.trainer import Trainer
from multitask.config import TrainingConfig

model = MultiTaskModel(config=config, embedding_dim=768)
trainer = Trainer(model, TrainingConfig(epochs=10))

# Train
history, thresholds = trainer.fit(train_ds, val_ds, num_tasks=5)
# thresholds includes "any_technique" alongside the 5 task thresholds

# Evaluate: pool metrics reported separately
results = trainer.evaluate(test_ds)
# results["pool_metrics"]["any_technique"] = {"f1": ..., "auc": ...}

# Inference
outputs = model.predict(test_ds)
outputs["any_technique"]  # max over technique_a, technique_b, technique_c
```
## Configuration

### Pool Definition

Each pool is a dictionary entry in `output_pools`:

```python
output_pools={
    "pool_name": {
        "tasks": ["task_a", "task_b"],   # required: at least 2 tasks
        "strategy": "max",               # required: "max", "mean", or "min"
        "task_type": "binary",
        "num_classes": 2,
    },
}
```

### Multiple Pools

Define as many pools as you need. Each task can only belong to one pool:

```python
output_pools={
    "any_propaganda": {
        "tasks": propaganda_task_names,   # 18 tasks
        "strategy": "max",
    },
    "mean_argumentation": {
        "tasks": argumentation_task_names,  # 8 tasks
        "strategy": "mean",
    },
}
```

### Validation Rules
`ModelConfig` validates pools at construction time and raises errors if:
- `task_names` is not provided (required when using pools)
- A pool name collides with a task name
- A referenced task doesn't exist in `task_names`
- A task appears in more than one pool
- A pool has fewer than 2 tasks
- Strategy is not one of `max`, `mean`, `min`
- A referenced task is a regression task (regression cannot be pooled)
- Optional `task_type` is invalid (must be `binary`, `multiclass`, or `multilabel`)
- Optional `num_classes` is invalid or inconsistent with `task_type`

## How it Works

### Predictions

During every forward pass, `_compute_pooled_outputs()` gathers constituent task tensors, stacks them, and applies `tf.reduce_max/mean/min`. The pooled output appears in the same dictionary as individual task predictions:

```python
outputs = model(inputs, training=False)
# outputs has N task keys + M pool keys
```

### Training
Pools do not affect training. The custom `train_step` and `test_step` filter pool keys out of `y_pred` before passing to `compute_loss`, so Keras only sees the architectural task outputs that match the dataset labels.

### Ground-Truth Labels (trainer.py)
The dataset never contains pool labels. During threshold optimization and evaluation, `_derive_pooled_labels()` derives them from the task labels:
- Masked labels (`-1`) are excluded via NaN replacement
- If all constituents are masked for an observation, the pooled label is masked
- The strategy (max/mean/min) is applied over non-masked constituents

### Evaluation

`evaluate()` reports pool metrics separately from task metrics:

```python
results = trainer.evaluate(test_ds)

results["task_metrics"]      # per-task metrics (tasks only)
results["pool_metrics"]      # per-pool metrics (pools only)
results["overall_metrics"]   # averages over tasks only (pools excluded)
```

Pool metrics are excluded from `overall_metrics` to prevent the averages from shifting based on how many pools you define.

### Save and Load

Pool configuration is persisted automatically:
- `config.json` includes the `output_pools` dictionary
- `thresholds.json` includes pool thresholds alongside task thresholds (when thresholds are saved)
- On load, the reconstructed model produces pooled outputs immediately

```python
model.save("my_model", thresholds=thresholds)

from multitask.models.persistence import load_model
loaded_model, loaded_thresholds = load_model("my_model")
# loaded_model.config.has_pools → True
# loaded_thresholds includes pool thresholds (if thresholds were saved)
```

## Helper Methods

`ModelConfig` provides several methods for working with pools:

```python
config.has_pools                        # True if any pools configured
config.pool_names                       # sorted list of pool names
config.all_output_names                 # task names + pool names
config.is_pool("any_technique")
config.get_pool_config("any_technique") # full config with defaults filled in
config.get_pool_task_type("any_technique")
config.get_pool_num_classes("any_technique")
```

`MultiTaskModel` also has:

```python
model.get_pool_info()   # list of dicts with pool metadata
model.summary_tree()    # includes pool info in the printed summary
```