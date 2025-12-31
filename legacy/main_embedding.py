

"""# Dependencies """
import os
path = os.getcwd()

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import json
import tensorflow_text as text

from utils.data_utils_embedding import prepare_data
from train_test.train_test_from_embedding import train
from train_test.train_test_from_embedding import test
from models.model_from_embedding import BertClassifier


import argparse
import yaml





def main(config):
    # Setup
    """# Setup """
    GPU = tf.config.list_physical_devices("GPU")

    strategy = tf.distribute.MirroredStrategy()


    # ----- read in the arguments 
    """# Target tasks"""
    main_task_names = config['main_task_names']
    task_sizes = config['task_sizes']
    pooled_task_name = config['pooled_task_name']
    
    with open(config['targets_file'], 'r') as fin:
        targets = json.load(fin)
    """# Loading data"""
    #read in training and target data and remove leading column
    select_columns = config['select_columns']
    num_tasks = len(select_columns) - 1
    data_path = config['data_path']
    embedding_path = config['embedding_path']

    # txt2embed is the dictionary matching text to embeddings
    ff_train, ff_val, ff_test, txt2embed = prepare_data(data_path, select_columns, task_sizes, main_task_names,
                                                        embedding_path = embedding_path,
                                                        sample_size_ratio=config['sample_size_ratio'],
                                                        data_type="int16")

    task_list=[target_name.replace("_bi", "" ) for target_name in list(ff_train.columns)[1:]]



    """# Training

    ## Training parameters
    """

    #specify training settings

    seq_length = tf.constant(512) #cannot be greater than 512
    workers = 1 #greater than 1 if using MultiWorkerMirroredStrategy
    epochs = 20

    #define list of parameters for grid search

    # --- grid search params
    # pars_grid = [(init_lr, wt_decay_rate, warmup_pct, dropout, batch_size)
    #              for init_lr in [15e-5, 3e-4]
    #              for wt_decay_rate in [0.01, 0.05, 0.1]
    #              for warmup_pct in [5, 20]
    #              for dropout in [0.1, 0.25, 0.4]
    #              for batch_size in [32, 64]]

    # best parameters found from the above grid search
    params_dict = config['training_params']
    epochs, init_lr = params_dict['epochs'], params_dict['init_lr']
    wt_decay_rate, warmup_pct = params_dict['wt_decay_rate'], params_dict['warmup_pct']
    dropout, batch_size = params_dict['dropout'], params_dict['batch_size']
    workers, seq_length = params_dict['workers'], params_dict['seq_length']
    ablate = config.get('ablate', []) # this is a list that can either be empty or have any of 'shared', 'task-type', or 'task'
    embedding_dim = config.get('embedding_dim', 4096)
    #params = [init_lr, wt_decay_rate, warmup_pct, dropout, batch_size, epochs]


    # Create an instance of the model
    with strategy.scope():
        bert_classifier = BertClassifier(num_classes=2, dropout=dropout,
                                        task_target_sizes=task_sizes,
                                        main_task_names=main_task_names,
                                        pooled_task_name=config['pooled_task_name'],
                                        ablate=ablate)

    """## Train"""
    #train model
    bert_classifier, pars_best = train(strategy, bert_classifier,
                                    ff_train, ff_val,
                                    task_sizes, main_task_names, num_tasks,
                                    txt2embed, embedding_dim=embedding_dim,
                                    epochs=epochs, init_lr=init_lr, wt_decay_rate=wt_decay_rate, 
                                    warmup_pct=warmup_pct, dropout=dropout, batch_size=batch_size,
                                    workers=workers, seq_length=seq_length)

    """### Saving parameters and thresholds"""

    import pickle
    import datetime
    fname_time = '{date:%Y_%m_%d_%H_%M_%S}'.format(date=datetime.datetime.now())

    fname_prefix = './'

    with open(f'{fname_prefix}{config["dataset_name"]}_pars_best_{fname_time}.pkl', 'wb') as fout:
        pickle.dump(pars_best, fout, pickle.HIGHEST_PROTOCOL)




    """# Testing"""

    metrics_dict, metrics_str = test(strategy, bert_classifier,
                                    ff_test, pars_best,
                                    task_list, task_sizes, main_task_names, num_tasks,
                                    txt2embed, embedding_dim=embedding_dim,
                                    pooled_task_name=pooled_task_name,
                                    batch_size=batch_size,
                                    workers=workers)

    fname_metrics = f"./results/metrics_{config['dataset_name']}_{fname_time}.json"
    with open(fname_metrics, "w") as fout:
        json.dump(metrics_dict, fout)
    print(f"\n ====== Testing metrics saved in {fname_metrics} ====== \n")

    fname_report = f"./results/output_{config['dataset_name']}_{fname_time}.txt"
    with open(fname_report, 'w') as fout:
        fout.write(metrics_str)
    print(f"\n ====== Testing report saved in {fname_report} ====== \n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training and testing pipeline.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)