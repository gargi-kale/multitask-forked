

import tensorflow as tf, numpy as np
from utils.misc_utils import pool_matrix


def load_batch_process(df, batch_size, preprocess_model,
                       task_sizes, main_task_names, num_tasks,
                       data_type="int16"):

    #build tensorflow data object with only text and DV of interest
    data = tf.data.Dataset.from_tensor_slices((np.asarray(df.loc[:, "text"]).astype("str"),
                                               pool_matrix(np.asarray(df.iloc[:,1:num_tasks+1]).astype(data_type),
                                                           task_sizes, main_task_names)))

    #get number of observations
    n_obs = len(df)

    #batch data object
    data = data.batch(batch_size)

    #preprocess text and return tuple of tokens and DV of interest
    data = data.map(lambda text, out: (preprocess_model(text), out),
                    num_parallel_calls = batch_size)

    #cache result and pre-fetch with auto buffer sizes
    data = data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return data, n_obs


def load_batch_test(df, batch_size, preprocess_model,
                    task_sizes, main_task_names, num_tasks,
                    data_type="int16"):

    #build tensorflow data objects with text and DV of interest
    data = tf.data.Dataset.from_tensor_slices(np.asarray(df.loc[:, "text"]).astype("str"))
    y = pool_matrix(np.asarray(df.iloc[:,1:num_tasks+1]),
                    task_sizes, main_task_names).astype(data_type)

    #batch data object
    data = data.batch(batch_size)

    #preprocess text and return tuple of tokens and DV of interest
    data = data.map(lambda text: preprocess_model(text), num_parallel_calls = batch_size)

    #cache result and pre-fetch with auto buffer sizes
    data = data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return data, y


def load_batch_target(df, batch_size, preprocess_model):

    #build tensorflow data objects with text
    data = tf.data.Dataset.from_tensor_slices(np.asarray(df.loc[:, "text"]).astype("str"))

    #batch data object
    data = data.batch(batch_size)

    #preprocess text and return tuple of tokens and DV of interest
    data = data.map(lambda text: preprocess_model(text), num_parallel_calls = batch_size)

    #cache result and pre-fetch with auto buffer sizes
    data = data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return data