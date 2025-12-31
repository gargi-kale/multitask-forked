
import tensorflow as tf, numpy as np


def compute_class_weights(y, strategy):
    with strategy.scope():
        # Count the number of instances in each class
        ids = np.where(~(y==-1))
        y = y[ids[0]]
        class_counts = np.bincount(y)
        if (len(class_counts)==1):
            if (y[0]==0):
                class_counts = np.array([1000, 0])
            elif (y[0]==1):
                class_counts = np.array([0, 1000])

        # Compute the class frequencies
        class_freqs = class_counts / len(y)

        # Compute the class weights
        class_weights = 1 / class_freqs

        # Normalize the weights (optional)
        class_weights = class_weights / np.sum(class_weights)

        return class_weights.astype(np.float32)



def get_all_class_weights(labels, strategy):
    with strategy.scope():
        # Compute class weights for each label
        class_weights = []
        for col in range(labels.shape[1]):
            y = labels[:, col]
            y = y[y != -1]  # Exclude missing labels
            class_weights.append(compute_class_weights(y, strategy))
        return class_weights



"""### Implementing loss functions"""
def binary_cross_entropy(y_true, y_pred):
    #with strategy.scope():
    epsilon = 1e-7  # small constant added to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)  # ensure values are within [epsilon, 1-epsilon]
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    return loss