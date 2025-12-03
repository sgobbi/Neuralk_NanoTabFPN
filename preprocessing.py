import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import openml


def create_nthing():
    return False

def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id, download_all_files=True)
    X, y, _, _ = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format="dataframe")
    print("table size = ", X.shape, flush = True)
    return (X,y)
    

def create_h5_prior_from_X_y(X, y, filename, 
                                 num_tasks=5000,
                                 total_rows=40,
                                 train_rows=30, 
                                 num_features = 8, 
                                 shuffle_columns_for_each_task = True):

    X_np = np.ascontiguousarray(X.to_numpy(), dtype='float32')
    y_np = np.ascontiguousarray(pd.Categorical(y).codes, dtype='int32')

    num_features_full_database = X_np.shape[1]
    max_num_classes = len(np.unique(y_np))

    if not shuffle_columns_for_each_task:
        #on prend un sample de taille num_features de l'array [0, num_features_full_database]
        fixed_cols = np.random.choice(num_features_full_database, size=num_features, replace=False)

    with h5py.File(filename, "w") as f:
        f.create_dataset("X", shape=(num_tasks, total_rows, num_features), dtype='float32')
        f.create_dataset("y", shape=(num_tasks, total_rows), dtype='float32')

        f.create_dataset("num_features", shape=(num_tasks,), dtype='int32')
        f.create_dataset("num_datapoints", shape=(num_tasks,), dtype='int32')
        f.create_dataset("single_eval_pos", shape=(num_tasks,), dtype='int32')
        f.create_dataset("max_num_classes", data=[max_num_classes])

        n_samples = len(X_np)

        for i in range(num_tasks):
            # on choisit aleatoirement total_rows rows parmi celle du dataset avec remplacement (on peut avoir plusieurs fois la meme)
            idx = np.random.choice(n_samples, total_rows, replace=True)

            #on choisit les colonnes (soit rechoisit aleatoirement a chaque task, soit on prend celle fixes au debut)
            if shuffle_columns_for_each_task:
                cols = np.random.choice(num_features_full_database, size=num_features, replace=False)
            else:
                cols = fixed_cols


            X_task = X_np[idx][:, cols].astype('float32')
            y_task = y_np[idx].astype('float32')

    
            f["X"][i] = X_task
            f["y"][i] = y_task

            f["num_features"][i] = num_features
            f["num_datapoints"][i] = total_rows
            f["single_eval_pos"][i] = train_rows

    print(f"Saved tasks here here  to {filename}")


def create_h5_prior_from_dataset(dataset_id, filename, num_tasks = 5000, total_rows = 40, train_rows = 30, num_features = 8, shuffle_columns_for_each_task = True):
    X,y = load_dataset(dataset_id)
    create_h5_prior_from_X_y(X, y ,filename, num_tasks, total_rows, train_rows, num_features, shuffle_columns_for_each_task) 