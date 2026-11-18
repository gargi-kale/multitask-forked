"""Training methods built on the multitask library.

Modules:
    data          loading splits, batch builders, class weights
    encoders      encoder presets (small BERT, ELECTRA, OpenAI embeddings)
    optim         AdamW with layer-wise learning-rate decay
    trainer       the pooled-output loss trainer every method builds on
    balancers     FAMO and DB-MTL adaptive task balancing
    mlm           masked-word auxiliary loss
    consistency   consistency training (famo_consf1_m5) and teacher variants
    scoring       per-epoch validation scoring and early stopping
    stage2        train/eval harness and frozen stage 2
    columns       IAC / Propaganda column names (dataset-specific, temporary)
"""

import tensorflow as tf

# mixed_float16 makes model outputs float16, which breaks the trainer's loss
# arithmetic (float16 * float32 Mul dtype mismatch in dict_weighted_masked_loss).
# Keeping float32 globally avoids that without changing the trainer.
tf.keras.mixed_precision.set_global_policy("float32")
