

import numpy as np


def pool_matrix(matrix_in, task_sizes, 
                main_task_names,
                pooled_task_name='Propaganda'):
    """
    Pool the matrix based on the task sizes and names.

    Args:
    matrix_in (np.array): Input matrix.
    task_sizes (list): List of task sizes.
    main_task_names (list): List of main task names.

    Returns:
    np.array: Pooled matrix.
    """
    if main_task_names[0] == pooled_task_name:
        pooled_task = matrix_in[:, :task_sizes[0]][np.newaxis].T.max(0)
        pooled_matrix = np.concatenate([pooled_task, matrix_in[:, task_sizes[0]:]], axis=1)
        return pooled_matrix
    else:
        return matrix_in