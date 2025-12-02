import matplotlib.pyplot as plt
import torch
import pandas as pd
from model import NanoTabPFNModel
import time

def plot_train_loss_vs_step(model_histories, label=None):
    """
    Plots training loss en fonction du step number
    """
    plt.figure(figsize=(6,4))
    for model_name, history in model_histories.items():
        plt.plot(history["step"], history["train loss"], label= f"{model_name} train loss")
    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Step")
    plt.legend(loc = 'upper right', frameon = True)
    plt.grid(True)
    plt.show()

def plot_train_loss_vs_time(model_histories, label=None):
    """
    Plots training loss en fonction du total training time
    """
    plt.figure(figsize=(6,4))
    for model_name, history in model_histories.items():
        plt.plot(history["cumulative time"], history["train loss"], label = f"{model_name} train loss")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Time")
    plt.legend(loc = 'upper right', frameon = True)
    plt.grid(True)
    plt.show()

def plot_eval_roc_auc_vs_time(model_histories, label=None):
    """
    Plots evaluation ROC-AUC en fonction du total training time
    """
    plt.figure(figsize=(6,4))
    for model_name, history in model_histories.items():
        plt.plot(history["cumulative time"], history["ROC AUC"], label= f"{model_name} ROC AUC")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("ROC AUC")
    plt.title("Evaluation ROC-AUC vs Time")
    
    plt.legend(loc = 'lower right', frameon = True)
    plt.grid(True)
    plt.show()

def plot_eval_roc_auc_vs_step(model_histories, label = None):
    """
    Plots evaluation ROC AUC en fonction du nombre de training steps
    """
    plt.figure(figsize =(6,4))
    for model_name, history in model_histories.items():
        plt.plot(history["step"], history["ROC AUC"], label = f"{model_name} ROC AUC")
    plt.xlabel("Training Step")
    plt.ylabel("ROC AUC")
    plt.title("Evaluation ROC-AUC vs Training step")
   
    plt.legend(loc = 'upper right', frameon = True)
    plt.grid(True)
    plt.show()


def plot_metric(metric: str, model_histories: dict):
    if metric == "ROC vs time":
        plot_eval_roc_auc_vs_time(model_histories)
    elif metric == "ROC vs step":
        plot_eval_roc_auc_vs_step(model_histories)
    elif metric == "train loss vs time":
        plot_train_loss_vs_time(model_histories)
    elif metric == "train loss vs step":
        plot_train_loss_vs_step(model_histories)


def measure_inference_time_from_state_dict(state_dict_path: str, embedding_size, num_heads, mlp_hidden_size, num_layers, num_outputs, 
                                           attention_type, 
                                           dataset, 
                                           device: torch.device = None, model_class = NanoTabPFNModel, **model_kwargs) -> float:
    """
    On load le modele depuis son state_dict et on mesure le inference time per sample moyen.
    
    Args:
        model_class: the class of your model (e.g., NanoTabPFNModel)
        state_dict_path: path to the .pt file storing the state dict
        dataset: Dataset or DataLoader to run inference on
        batch_size: batch size for inference
        device: torch.device to use. If None, uses CUDA if available.
        **model_kwargs: keyword arguments to instantiate your model_class
        
    Returns:
        avg_time_per_sample: average inference time per sample (seconds)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    model = NanoTabPFNModel(embedding_size= embedding_size, 
                            num_attention_heads=num_heads, 
                            mlp_hidden_size=mlp_hidden_size,
                            num_layers = num_layers, 
                            num_outputs= num_outputs, 
                            attention_type=attention_type)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create DataLoader if dataset is not one already
    if not isinstance(dataset, torch.utils.data.DataLoader):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False)
    else:
        dataloader = dataset
    
    total_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            #print("batch keys: ", batch.keys(), flush = True)
            #print("batch['x'] type ", type(batch['x']))
            #print("batch X : ", batch['x'].size(), flush = True)
            x = batch['x'] 
            y = batch['y']
            train_test_split = batch['train_test_split_index']
            x = x.to(device)
            y = y.to(device)
           
            batch_size_actual = x.shape[0]
            #print("batch size", batch_size_actual, flush=True)
            
            start_time = time.time()
            _ = model((x, y), train_test_split )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += end_time - start_time
            total_samples += batch_size_actual
    
    avg_time_per_sample = total_time / total_samples
    return avg_time_per_sample