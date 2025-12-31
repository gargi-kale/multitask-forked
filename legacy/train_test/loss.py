
import tensorflow as tf
from utils.misc_utils import pool_matrix
from utils.loss_utils import *


"""## Loss

### Task and class weights
"""
def get_loss(strategy,
             ff_train, task_sizes, main_task_names):

    with strategy.scope():
        MASK_VALUE = -1

        def build_doubleWeighted_masked_loss(loss_function, class_weights,
                                             gamma=None,
                                             task_target_sizes=task_sizes):
            """Builds a loss function that masks based on targets and applies class weights

            Args:
                loss_function: The loss function to mask
                class_weights: A dictionary mapping class indices to weights
                mask_value: The value to mask in the targets
                gamma: the task weights

            Returns:
                function: a loss function that acts like loss_function with masked inputs and class weights
            """
            # this gets class weights for all entries, i.e. weights is of the same shape as y_true
            def get_weights(y_true):
                weights = np.ones_like(y_true, dtype=np.float32)
                for col in range(y_true.shape[1]):
                    ids0 = np.where(y_true[:, col] == 0)
                    ids1 = np.where(y_true[:, col] == 1)
                    ids_masked = np.where(y_true[:, col] == MASK_VALUE)
                    weights[ids0[0], col] *= class_weights[col][0]
                    weights[ids1[0], col] *= class_weights[col][1]
                    weights[ids_masked[0], col] *= 0.

                return weights


            # --- By default set task weights as inverse of the enrichment of the task in the data
            if gamma is None:
                rec_sizes = []
                col_begin, col_end = 1, 1
                for task_target_size in task_target_sizes:
                    col_end += task_target_size
                    ids_task = np.where(~(np.any(ff_train.values[:,col_begin:col_end]==-1, axis=1)))
                    size = len(ids_task[0])
                    rec_sizes += [1. / float(size)]
                    col_begin += task_target_size
                gamma = [size/sum(rec_sizes) for size in rec_sizes]


            def weighted_masked_loss_function(y_true, y_pred):
                task_weights = []
                for t_idx in range(len(task_target_sizes)):
                    task_weights += [tf.ones((tf.shape(y_true)[0], 1), dtype=tf.float32)]
                main_tasks_weights = tf.concat([task_weights[t_idx]*gamma[t_idx]
                                                for t_idx in range(len(task_target_sizes))],
                                                axis=1)

                weights = tf.py_function(get_weights, [y_true], tf.float32)
                weights = tf.convert_to_tensor(weights, dtype=tf.float32)

                return weights * tf.cast(loss_function(y_true, y_pred), tf.float32)


            return weighted_masked_loss_function
        


        # specify metrics and loss
        # take the max over each subset of columns (for each main task)
        pooled_train = pool_matrix(ff_train.iloc[:,1:].values,
                                   task_sizes, main_task_names)
        class_weights = get_all_class_weights(pooled_train, strategy)
        loss = build_doubleWeighted_masked_loss(binary_cross_entropy, class_weights,
                                                task_target_sizes=task_sizes)
    
    return loss
