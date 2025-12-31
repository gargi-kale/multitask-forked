
import numpy as np
import pandas as pd
import json
import pickle as pkl

def load_data(file_path, select_columns):
    """
    Load data from a CSV file and select specific columns.

    Args:
    file_path (str): Path to the CSV file.
    select_columns (list): List of columns to select from the CSV.

    Returns:
    DataFrame: A pandas DataFrame with the selected columns.
    """
    data = pd.read_csv(file_path)
    if type(select_columns[0])==str:
        data = data[select_columns]
    else: # in this case select_columns is a list of integers
        data = data.iloc[:, select_columns]
    return data[data["text"].notna()]



def prepare_data(data_path, select_columns, task_sizes, main_task_names, 
                 embedding_path,
                 from_pkl=True, # load embeddings from pkl files
                 sample_size_ratio=1.0, data_type="int16"):
    """
    Prepare the data for training, validation, and testing.

    Args:
    data_path (str): Base path where the data files are located.
    select_columns (list): Columns to select from the data files.
    task_sizes (list): Sizes of each task.
    main_task_names (list): Names of the main tasks.
    sample_size_ratio (float): Ratio of the sample size to use. Defaults to 1.0.
    data_type (str): Data type to use for the DataFrame. Defaults to "int16".

    Returns:
    tuple: A tuple containing the training, validation, and test DataFrames.
    """
    # Load data
    train_data = load_data(data_path + "_train.csv", select_columns)
    val_data = load_data(data_path + "_val.csv", select_columns)
    test_data = load_data(data_path + "_test.csv", select_columns)


    # --- load embeddings ----
    if from_pkl:
        is_invalid = lambda row: ((row[-1] == ['NaN']) or (row[-1] == ['Nan']))
        with open(embedding_path + '_train.pkl', 'rb') as fin:
            df_embed_train = pkl.load(fin)
            mask = df_embed_train.apply(is_invalid, axis=1)
            df_embed_train = df_embed_train[~mask].reset_index(drop=True)
            train_data = train_data[~mask].reset_index(drop=True)
        with open(embedding_path + '_val.pkl', 'rb') as fin:
            df_embed_val = pkl.load(fin)
            mask = df_embed_val.apply(is_invalid, axis=1)
            df_embed_val = df_embed_val[~mask].reset_index(drop=True)
            val_data = val_data[~mask].reset_index(drop=True)
        with open(embedding_path + '_test.pkl', 'rb') as fin:
            df_embed_test = pkl.load(fin)
            mask = df_embed_test.apply(is_invalid, axis=1)
            df_embed_test = df_embed_test[~mask].reset_index(drop=True)
            test_data = test_data[~mask].reset_index(drop=True)
    else:
        df_embed_train = pd.read_csv(embedding_path + '_train.csv')
        mask = df_embed_train.iloc[:, -1]=="['Nan']"
        df_embed_train = df_embed_train[~mask].reset_index(drop=True)
        train_data = train_data[~mask].reset_index(drop=True)

        df_embed_val = pd.read_csv(embedding_path + '_val.csv')
        mask = df_embed_val.iloc[:, -1]=="['Nan']"
        df_embed_val = df_embed_val[~mask].reset_index(drop=True)
        val_data = val_data[~mask].reset_index(drop=True)

        df_embed_test = pd.read_csv(embedding_path + '_test.csv')
        mask = df_embed_test.iloc[:, -1]=="['Nan']"
        df_embed_test = df_embed_test[~mask].reset_index(drop=True)
        test_data = test_data[~mask].reset_index(drop=True)


    # Sample data
    if sample_size_ratio < 1.0:
        sample_size = min([len(train_data), max([int(len(train_data) * sample_size_ratio), 500])])
        ids = np.arange(len(train_data))
        selected_ids = np.random.choice(ids, sample_size) # rows to add
        train_data = train_data.iloc[selected_ids,:]
        df_embed_train = df_embed_train.iloc[selected_ids,:]

        sample_size = min([len(val_data), max([int(len(val_data) * sample_size_ratio), 500])])
        ids = np.arange(len(val_data))
        selected_ids = np.random.choice(ids, sample_size) # rows to add
        val_data = val_data.iloc[selected_ids,:]
        df_embed_val = df_embed_val.iloc[selected_ids,:]

        sample_size = min([len(test_data), max([int(len(test_data) * sample_size_ratio), 500])])
        ids = np.arange(len(test_data))
        selected_ids = np.random.choice(ids, sample_size) # rows to add
        test_data = test_data.iloc[selected_ids,:]
        df_embed_test = df_embed_test.iloc[selected_ids,:]



    # ---- get the dictionary
    if from_pkl:
        txt2embed = {train_data.iloc[idx, 0]: np.array(df_embed_train.iloc[idx, -1])
                    for idx in range(train_data.shape[0])}
        txt2embed.update({val_data.iloc[idx, 0]: np.array(df_embed_val.iloc[idx, -1])
                          for idx in range(val_data.shape[0])})
        txt2embed.update({test_data.iloc[idx, 0]: np.array(df_embed_test.iloc[idx, -1])
                          for idx in range(test_data.shape[0])})
    else:
        txt2embed = {train_data.iloc[idx, 0]: np.array(json.loads(df_embed_train.iloc[idx, -1]))
                    for idx in range(train_data.shape[0])}
        txt2embed.update({val_data.iloc[idx, 0]: np.array(json.loads(df_embed_val.iloc[idx, -1]))
                          for idx in range(val_data.shape[0])})
        txt2embed.update({test_data.iloc[idx, 0]: np.array(json.loads(df_embed_test.iloc[idx, -1]))
                          for idx in range(test_data.shape[0])})


    # Fill missing values and change data types
    train_data = train_data.fillna(-1)
    val_data = val_data.fillna(-1)
    test_data = test_data.fillna(-1)
     
    ### Change dtypes for performance
    DATA_TYPE = "int16"
    dtype_dict = {col: DATA_TYPE for col in train_data.iloc[:,1:].columns}
    train_data = train_data.astype(dtype= dtype_dict)
    val_data = val_data.astype(dtype= dtype_dict)
    test_data = test_data.astype(dtype= dtype_dict)


    return train_data, val_data, test_data, txt2embed
