import random
import time

import h5py
import numpy as np
import schedulefree
import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel
from sklearn.datasets import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(0)

def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device

datasets = []
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0))

def eval(classifier):
    scores = {
        "roc_auc": 0,
        "acc": 0,
        "balanced_acc": 0
    }
    for  X_train, X_test, y_train, y_test in datasets:
         classifier.fit(X_train, y_train)
         prob = classifier.predict_proba(X_test)
         pred = prob.argmax(axis=1) # avoid a second forward pass by not calling predict
         if prob.shape[1]==2:
             prob = prob[:,:1]
         scores["roc_auc"] += float(roc_auc_score(y_test, prob, multi_class="ovr"))
         scores["acc"] += float(accuracy_score(y_test, pred))
         scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    scores = {k:v/len(datasets) for k,v in scores.items()}
    return scores

def train(model: NanoTabPFNModel, prior: DataLoader, eval_loader = None, 
          lr: float = 1e-4, device: torch.device = None, steps_per_eval=10, eval_func=None):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        lr: (float) learning rate
        device: (torch.device) the device we are using
        steps_per_eval: (int) how many steps we wait before running evaluation again
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets

    Returns:
        (model) our trained numpy model
        (list) a list containing our eval history, each entry is the real time used for training so far together
               with a dict mapping metric names to their average values accross a list of datasets
    """
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    cumulative_time = 0
    prev_eval_time = 0
    history=[]
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            torch.cuda.reset_peak_memory_stats(device)
            

            train_test_split_index = full_data["train_test_split_index"]
            #if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
            #    continue
            data = (full_data["x"].to(device),
                    full_data["y"][:, :train_test_split_index].to(device))
            targets = full_data["y"].to(device)

           
            output = model(data, train_test_split_index=train_test_split_index)
            targets = targets[:, train_test_split_index:]


            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            #print("output shape: ", output.shape)
            #print("target shape: ", targets.shape)
            #print("output : ", output[0])
            #print("target shape: ", targets[0])
            


            loss = criterion(output, targets).mean()
            loss.backward()
            total_loss = loss.cpu().detach().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()

            peak_mem_MB = torch.cuda.max_memory_allocated(device) / 1e6
            current_mem_MB = torch.cuda.memory_allocated(device) / 1e6
            step_train_duration = time.time() - step_start_time
            cumulative_time += step_train_duration

            

            # evaluate
            if step % steps_per_eval == steps_per_eval-1:
                

                history_entry = {"step": step, 
                                 "cumulative time":cumulative_time, 
                                 "step time": step_train_duration, 
                                 "train loss": total_loss, 
                                 "current mem MB": current_mem_MB, 
                                 "peak mem MB": peak_mem_MB}

                if eval_loader is not None:
                    model.eval()
                    classifier = NanoTabPFNClassifier(model, device)

                    all_preds, all_targets = [], []
                    datasets = []

                    with torch.no_grad():
                        for batch in eval_loader:
                            x_eval = batch["x"].to(device)
                            y_eval = batch["y"].to(device).reshape(-1)
                            X_np = x_eval.cpu().numpy()
                            y_np = y_eval.cpu().numpy().reshape(-1)
                            X_flat = X_np.reshape(-1, X_np.shape[2])  # (batch_size * seq_len, num_features)
                            y_flat = y_np.reshape(-1)
                            datasets.append((X_flat, y_flat))
                            #print("x_eval shape:", x_eval.shape)
                            #print("y_eval shape:", y_eval.shape)


                            #y_pred = classifier.predict(x_eval)
                            #all_preds.append(y_pred)
                            #all_targets.append(y_eval.cpu().numpy())
                    metrics = evaluate_model(classifier, datasets)

                    print("metric keys: " , metrics.keys())
                    
                    for k, v in metrics.items():
                        history_entry[k] = v

                    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    print(f"time {cumulative_time:.1f}s | loss {total_loss:.4f} | {metric_str}")

                    model.train()
                else:
                    print(f"time {cumulative_time:7.1f}s | loss {total_loss:7.4f}")

                history.append(history_entry)
    except KeyboardInterrupt:
        pass

    return model, history


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.
    Est compatible avec des tables de tailles differentes puisqu'il va 
    yield un tenseur de taille (batch size, max_rows in the batch, max features in the batch)

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename, num_steps, batch_size, device=None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.pointer = 0
        if device is None:
            device = get_default_device()
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps

if __name__ == "__main__":
    device = get_default_device()
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=2500, batch_size=32, device=device)
    model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
    print("Final evaluation:")
    print(eval(NanoTabPFNClassifier(model, device)))


_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
def evaluate_model(model, datasets):
    """Evaluates a model on multiple datasets and returns metrics
    
    This method takes a model we want to evaluate as input, and a dictionnary of datasets where the key is the name of the dataset
    and the value being a tuple (X,y) of the features and labels 
    It then does a 5 fold cross validation for every dataset, calculating its Roc_Auc_score 
    It returns a dictionnary {dataset_name/ROC_AUC: roc_auc_score} with also an average of this metric over all datasets 
    """ 
    metrics = {}
    for i, (X,y)  in enumerate(datasets):
        targets = []
        probabilities = []
        
        for train_idx, test_idx in _skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test  = y[train_idx], y[test_idx]
            targets.append(y_test)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # binary classification with neural network
                y_proba = y_proba[:, 1]
            probabilities.append(y_proba)
    
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)

        metrics[f"{i}/ROC AUC"] = roc_auc_score(targets, probabilities, multi_class="ovr")
    
    metric_names = list({key.split("/")[-1] for key in metrics.keys()})
    for metric_name in metric_names:
        avg_metric = np.mean([metrics[key] for key in metrics.keys() if key.endswith(metric_name)])
        metrics[f"{metric_name}"] = float(avg_metric)
    
    return {"ROC AUC" : metrics["ROC AUC"]}