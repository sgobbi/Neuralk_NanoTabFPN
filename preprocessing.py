import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import openml

# -----------------------------
# 1. Load a dataset
# -----------------------------
dataset_id = 31  # Example: "credit-g" from OpenML
dataset = openml.datasets.get_dataset(dataset_id)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")

# Convert to numpy arrays
X = X.values
y = y.values

# -----------------------------
# 2. Define synthetic "tasks"
# -----------------------------
num_tasks = 1000       # how many tasks you want to generate
context_rows = 20      # number of rows per task/context
num_features = X.shape[1]

# Preallocate arrays for HDF5
X_tasks = np.zeros((num_tasks, context_rows, num_features), dtype=np.float32)
y_tasks = np.zeros((num_tasks, context_rows), dtype=np.int64)
train_test_split_indices = np.zeros(num_tasks, dtype=np.int64)

for i in range(num_tasks):
    # Randomly sample context_rows rows for this task
    idx = np.random.choice(len(X), context_rows, replace=False)
    X_task = X[idx]
    y_task = y[idx]

    # Split train/test within the context
    split = int(0.8 * context_rows)  # 80% train
    train_test_split_indices[i] = split

    X_tasks[i, :context_rows, :] = X_task
    y_tasks[i, :context_rows] = y_task

# -----------------------------
# 3. Save to HDF5
# -----------------------------
filename = "my_dataset_prior.h5"
with h5py.File(filename, "w") as f:
    f.create_dataset("X", data=X_tasks)
    f.create_dataset("y", data=y_tasks)
    f.create_dataset("num_features", data=[num_features])
    f.create_dataset("num_datapoints", data=[context_rows]*num_tasks)
    f.create_dataset("single_eval_pos", data=train_test_split_indices)
    # optional metadata
    f.create_dataset("max_num_classes", data=[len(np.unique(y))])

print(f"Saved prior to {filename}")