import numpy as np
import tensorflow as tf 

def dataframe_to_dataset(df, task_names, batch_size=32, shuffle=False):
    """Convert pandas DataFrame to tf.data.Dataset for Universal Sentence Encoder.
    
    Args:
        df: DataFrame with 'text' column and task label columns
        task_names: List of task names matching the columns in df
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        
    Returns:
        tf.data.Dataset yielding (text_input, labels_dict) tuples
    """
    # Extract texts and labels
    # For Universal Sentence Encoder, we need to ensure texts are strings
    texts = [str(t) if t is not None and t == t else "" for t in df['text'].tolist()]  # Convert to list to preserve strings
    labels_dict = {}
    
    for task_name in task_names:
        if task_name in df.columns:
            labels = df[task_name].values
            # Handle missing values (replace NaN with -1 for masking)
            labels = np.where(np.isnan(labels), -1, labels)
            labels_dict[task_name] = labels.astype(np.float32)
        else:
            raise ValueError(f"Task column '{task_name}' not found in DataFrame")
    
    # Create dataset from tensors
    # Convert texts to TensorFlow string tensor explicitly
    texts_tensor = tf.constant(texts, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((texts_tensor, labels_dict))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def embeddings_and_labels_to_dataset(
    embeddings_array: np.ndarray,
    df,
    task_names,
    batch_size: int = 32,
    shuffle: bool = False,
):
    """Build a tf.data.Dataset from precomputed embeddings and label columns in a DataFrame.

    Row order of embeddings_array must match the DataFrame (row i = row i).

    Args:
        embeddings_array: Shape (n_samples, embedding_dim), float32
        df: DataFrame with task label columns (same task_names)
        task_names: List of task names matching the columns in df
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset

    Returns:
        tf.data.Dataset yielding (embedding_tensor, labels_dict) tuples.
        Each batch: inputs shape (batch_size, embedding_dim), labels_dict as in dataframe_to_dataset.
    """
    n = len(df)
    if len(embeddings_array) != n:
        raise ValueError(
            f"embeddings_array length ({len(embeddings_array)}) must match DataFrame length ({n})"
        )
    labels_dict = {}
    for task_name in task_names:
        if task_name in df.columns:
            labels = df[task_name].values
            labels = np.where(np.isnan(labels), -1, labels)
            labels_dict[task_name] = labels.astype(np.float32)
        else:
            raise ValueError(f"Task column '{task_name}' not found in DataFrame")
    embeddings_tensor = tf.constant(np.asarray(embeddings_array, dtype=np.float32), dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((embeddings_tensor, labels_dict))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=n)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset