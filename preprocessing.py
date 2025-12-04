import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
from matplotlib import pyplot as plt
import functools
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import numpy as np
import openml
import pandas as pd
from openml.tasks import TaskType
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline




def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id, download_all_files=True)
    X, y, _, _ = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format="dataframe")
    print("table size = ", X.shape, flush = True)
    size_train = int(X.shape[0] * 0.7)
    train_idx = np.random.choice(X.shape[0], size_train, replace=False)

    
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[train_idx] = True

    #
    X_train = X[mask]
    y_train = y[mask]

    X_eval = X[~mask]
    y_eval = y[~mask]
    print(X.iloc[0].to_dict())
    print("y size", y.shape, flush = True )
    return (X_train,y_train, X_eval, y_eval)
    

def create_h5_prior_from_X_y(X, y, filename, 
                                 num_tasks=5000,
                                 total_rows=40,
                                 train_rows=30, 
                                 num_features = 8, 
                                 shuffle_columns_for_each_task = True):
    print()

    X_np = np.ascontiguousarray(X, dtype='float32')
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


def create_h5_prior_from_dataset(dataset_id, train_filename, test_filename,
                                 num_tasks=5000, 
                                 total_rows=40, 
                                 train_rows=30, 
                                 num_features=8, 
                                 shuffle_columns_for_each_task=True):
    """
    Download an OpenML dataset, preprocess it (numeric + categorical features),
    encode labels, separe entre un train et un test set and create an HDF5 file ready for PFN training.

    - download openML dataset
    - preprocess it avec le preprocessor du codebase original
    - create a H5 file avec des tasks samplees depuis la base de donnees 
    """
    
    X, y,  X_eval, y_eval = load_dataset(dataset_id)
    
    
    print("X_train ", X.shape)
    print("y train ", y.shape)
    print("X test ", X_eval.shape)
    print("y test ", y_eval.shape)


    preprocessor = get_feature_preprocessor(X)
    X_train_processed = preprocessor.fit_transform(X).astype(np.float32)
    X_test_processed = preprocessor.fit_transform(X_eval).astype(np.float32)

    y_train_encoded = pd.Categorical(y).codes.astype(np.int32)
    y_test_encoded = pd.Categorical(y_eval).codes.astype(np.int32)

  
    create_h5_prior_from_X_y(X_train_processed, y_train_encoded, train_filename,
                             num_tasks= int(0.7*num_tasks),
                             total_rows=total_rows,
                             train_rows=train_rows,
                             num_features=num_features,
                             shuffle_columns_for_each_task=shuffle_columns_for_each_task)
    
    create_h5_prior_from_X_y(X_test_processed, y_test_encoded, test_filename,
                             num_tasks=int(0.3 *num_tasks),
                             total_rows=total_rows,
                             train_rows=train_rows,
                             num_features=num_features,
                             shuffle_columns_for_each_task=shuffle_columns_for_each_task)







def create_h5_prior_from_dataset_old(dataset_id, filename_train, filename_test, num_tasks = 5000, total_rows = 40, train_rows = 30, num_features = 8, shuffle_columns_for_each_task = True):
    X,y, X_eval, y_eval = load_dataset(dataset_id)
    create_h5_prior_from_X_y(X, y ,filename_train, int(0.7* num_tasks), total_rows, train_rows, num_features, shuffle_columns_for_each_task) 
    create_h5_prior_from_X_y(X_eval, y_eval ,filename_test, int(0.3*num_tasks), total_rows, train_rows, num_features, shuffle_columns_for_each_task)

# preprocessing from the original codebase

def get_feature_preprocessor_safe(X):
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []

    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum()
        non_nan_entries = X[col].notna().sum()
        num_mask.append(numeric_entries == non_nan_entries)
        cat_mask.append(numeric_entries != non_nan_entries)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    # numeric: impute NaN with mean
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('to_float', FunctionTransformer(lambda x: x.astype(np.float32)))
    ])

    # categorical: impute NaN with -1 and ordinal encode
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('encoder', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_mask),
        ('cat', cat_transformer, cat_mask)
    ])
    return preprocessor

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features

    Creates masks for numeric and categorical columns
    Creates pipelines to convert numeric columns to numbers and categorical columns as integers (ordinal encoding)
    Returns a ColumnTransformer that can be used like so:
    preprocessor = get_feature_preprocessor(df)
    X_processed = preprocessor.fit_transform(df)

    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors='coerce').to_numpy())), # in case numeric columns are stored as strings
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor

def get_openml_datasets(
        max_features_eval: int = 10, 
        new_instances_eval: int = 200, 
        target_classes_filter: int = 2,
        **kwargs,
        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load OpenML tabarena datasets with at most `max_features` features and subsampled (stratified) to `new_instances` instances.



    This function:
    Loads a curated list of OpenML datasets.
    Filters datasets based on:
    Number of features
    Number of classes
    Missing values
    Class balance
    Optionally subsamples the dataset to a manageable number of instances.
    Converts data to numeric arrays and encodes labels.
    Returns a dictionary of fully preprocessed (X, y) datasets ready for ML experiments.

    """
    task_ids = [
        363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
        363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
        363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
        363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
        363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
        363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
        363708, 363711, 363712
    ] # TabArena v0.1
    datasets = {}
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue  # skip task, only classification
        dataset = task.get_dataset(download_data=False)

        if dataset.qualities["NumberOfFeatures"] > max_features_eval or (dataset.qualities["NumberOfClasses"] > target_classes_filter) or dataset.qualities["PercentageOfInstancesWithMissingValues"] > 0 or dataset.qualities["MinorityClassPercentage"] < 2.5:
            continue
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )
        if new_instances_eval < len(y):
            _, X_sub, _, y_sub = train_test_split(
                X, y,
                test_size=new_instances_eval,
                stratify=y,
                random_state=0,
            )
        else:
            X_sub = X
            y_sub = y
        
        X = X_sub.to_numpy(copy=True)
        y = y_sub.to_numpy(copy=True)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        preprocessor = get_feature_preprocessor(X)
        X = preprocessor.fit_transform(X)
        datasets[dataset.name] = (X, y)
    return datasets
