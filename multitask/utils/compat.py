import tensorflow as tf
import keras

class WarmUpWrapper(keras.optimizers.schedules.LearningRateSchedule):
    """
    Keras 3 native warmup wrapper.
    """
    def __init__(self, warmup_steps, decay_schedule_fn, initial_learning_rate, name=None):
        super().__init__()
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_schedule_fn = decay_schedule_fn
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)
        self.name = name

    def __call__(self, step):
        # keras 3 optimizer automatically provides the current training step here
        step = tf.cast(step, tf.float32)
        
        # Linear Warmup (slopes from 0 to initial_learning_rate)
        warmup_lr = self.initial_learning_rate * (step / self.warmup_steps)
        
        # Provided Decay Schedule (Polynomial, Cosine, etc.)
        decay_lr = self.decay_schedule_fn(step)

        # Logic to switch between phases
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        """Ensures the schedule can be serialized for model saving/loading."""
        return {
            "warmup_steps": float(self.warmup_steps.numpy()) if hasattr(self.warmup_steps, 'numpy') else float(self.warmup_steps),
            "decay_schedule_fn": keras.optimizers.schedules.serialize(self.decay_schedule_fn),
            "initial_learning_rate": float(self.initial_learning_rate.numpy()) if hasattr(self.initial_learning_rate, 'numpy') else float(self.initial_learning_rate),
            "name": self.name,
        }