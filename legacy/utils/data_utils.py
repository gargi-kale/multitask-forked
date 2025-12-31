
import numpy as np
import pandas as pd

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



def prepare_data(data_path, select_columns, task_sizes, main_task_names, sample_size_ratio=1.0, data_type="int16"):
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


    # Sample data
    if sample_size_ratio < 1.0:
        sample_size = min([len(train_data), max([int(len(train_data) * sample_size_ratio), 500])])
        ids = np.arange(len(train_data))
        selected_ids = np.random.choice(ids, sample_size) # rows to add
        train_data = train_data.iloc[selected_ids,:]

        sample_size = min([len(val_data), max([int(len(val_data) * sample_size_ratio), 500])])
        ids = np.arange(len(val_data))
        selected_ids = np.random.choice(ids, sample_size) # rows to add
        val_data = val_data.iloc[selected_ids,:]

        sample_size = min([len(test_data), max([int(len(test_data) * sample_size_ratio), 500])])
        ids = np.arange(len(test_data))
        selected_ids = np.random.choice(ids, sample_size) # rows to add
        test_data = test_data.iloc[selected_ids,:]


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


    return train_data, val_data, test_data
