

import tensorflow as tf, numpy as np
from utils.misc_utils import pool_matrix


# def get_embedding(text, txt2embed, embedding_dim=4096):
#     # Fallback to a zero vector if text not found
#     return txt2embed.get(text, np.zeros(shape=(embedding_dim,)))



def load_batch_process(df, batch_size, txt2embed,
                       task_sizes, main_task_names, num_tasks,
                       embedding_dim=4096):
    
    get_embedding = lambda text: txt2embed.get(text, np.zeros(shape=(embedding_dim,)))

    # Convert each text to its embedding, ensuring output is a NumPy array of floats
    embeddings_array = np.array(df['text'].apply(get_embedding).tolist(), dtype=np.float32)

    # Now use embeddings_array and labels to create your tf.data.Dataset
    data = tf.data.Dataset.from_tensor_slices((embeddings_array,
                                               pool_matrix(np.asarray(df.iloc[:,1:num_tasks+1]).astype('float32'),
                                                           task_sizes, main_task_names)))

    #get number of observations
    n_obs = len(df)

    #batch data object
    data = data.batch(batch_size)

    #cache result and pre-fetch with auto buffer sizes
    data = data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return data, n_obs



def load_batch_test(df, batch_size, txt2embed,
                    task_sizes, main_task_names, num_tasks,
                    embedding_dim=4096, data_type="int16"):
    
    get_embedding = lambda text: txt2embed.get(text, np.zeros(shape=(embedding_dim,)))

    # Convert each text to its embedding, ensuring output is a NumPy array of floats
    embeddings_array = np.array(df['text'].apply(get_embedding).tolist(), dtype=np.float32)

    #build tensorflow data objects with text and DV of interest
    data = tf.data.Dataset.from_tensor_slices(embeddings_array)
    y = pool_matrix(np.asarray(df.iloc[:,1:num_tasks+1]),
                    task_sizes, main_task_names).astype(data_type)

    #batch data object
    data = data.batch(batch_size)

    #cache result and pre-fetch with auto buffer sizes
    data = data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return data, y


def load_batch_target(df, batch_size, preprocess_model, txt2embed, embedding_dim=4096):

    #build tensorflow data objects with text
    get_embedding = lambda text: txt2embed.get(text, np.zeros(shape=(embedding_dim,)))

    # Convert each text to its embedding, ensuring output is a NumPy array of floats
    embeddings_array = np.array(df['text'].apply(get_embedding).tolist(), dtype=np.float32)
    
    data = tf.data.Dataset.from_tensor_slices(embeddings_array)

    #batch data object
    data = data.batch(batch_size)
    
    #cache result and pre-fetch with auto buffer sizes
    data = data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return data