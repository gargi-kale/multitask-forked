import tensorflow as tf

class WarmUpWrapper(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Compatibility wrapper for leagcy warmup scedules under Keras 3.
    Make the step argument optional."""
    def __init__(self, warmup_instance):
        self.warmup_instance = warmup_instance
        
    def __call__(self, step=None):
        if step is None:
            # Pull from the global counter if the optimizer doesn't provide it
            step = tf.cast(tf.compat.v1.train.get_or_create_global_step(), tf.float32)
        return self.warmup_instance(step)
        
    def get_config(self):
        return self.warmup_instance.get_config()