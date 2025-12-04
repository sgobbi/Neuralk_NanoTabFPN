# Making Attention Cheaper: Exploring Efficient Mechanisms in nanoTabPFN


Clone the repository, afterwards create the conda environment:
```
conda env create -f environment.yml
```

and activate it:
```
conda activate NanoTabPFN
```

### Our Code

- `model.py` contains the implementation of the architecture and a sklearn-like interface in less than 200 lines of code. 
- `train.py` implements a simple training loop and prior dump data loader in under 200 lines
- `attentions.py` implements the different attention mechanisms
- `visualisation.py` implements the different plotting methods (training loss, roc auc, inference time...)
- `preprocessing.py` implements the methods necessary to download datasets from openml, sample tasks from them, and turn them into a h5 file compatible with the DataLoader
- `experiment_adult.ipynb` reacreates the experiments we did to compare different attention mechanisms 


#### Step by Step explanation:

First we generate our train and test sets from an openml dataset of our choice (179 in our case)
```
create_h5_prior_from_dataset(179, "h5_files/TRAIN_500_adult_database_100_80_rows_13_features.h5","h5_files/TEST_500_adult_database_100_80_rows_13_features.h5" , num_tasks = 500, total_rows = 100, train_rows=80, num_features=13)
```

First we import our code from model.py and train.py
```py
from model import NanoTabPFNModel
from model import NanoTabPFNClassifier
from train import PriorDumpDataLoader
from train import train, get_default_device
```
Then we instantiate our model with the attention mechanism of your choice (Original, Scratch, Sparse, Local, Pooling)
```py
model = NanoTabPFNModel(
    embedding_size=96,
    num_attention_heads=4,
    mlp_hidden_size=192,
    num_layers=3,
    num_outputs=2,
    attention_type = "Scratch"
)
```
and our dataloaders from the train and test H5 files we have created and stored 
```py
prior = PriorDumpDataLoader(
        "h5_files/TRAIN_500_adult_database_100_80_rows_13_features.h5", 
        num_steps=2000, 
        batch_size=32, 
        device=device)
eval_loader = PriorDumpDataLoader(
        "h5_files/TRAIN_500_adult_database_100_80_rows_13_features.h5",
        num_steps=150,      # use all data once
        batch_size=1,
        device=device
    )
```
Now we can train our model:
```py
device = get_default_device()
model, _ = train(
    model,
    prior,
    eval_loader = eval_loader
    lr = 4e-3,
    device = device
)
```


