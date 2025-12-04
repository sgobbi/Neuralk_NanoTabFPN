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

First we import our code from model.py and train.py
```py
from model import NanoTabPFNModel
from model import NanoTabPFNClassifier
from train import PriorDumpDataLoader
from train import train, get_default_device
```
Then we instantiate our model
```py
model = NanoTabPFNModel(
    embedding_size=96,
    num_attention_heads=4,
    mlp_hidden_size=192,
    num_layers=3,
    num_outputs=2
)
```
and our dataloader
```py
prior = PriorDumpDataLoader(
    "300k_150x5_2.h5",
    num_steps=2500,
    batch_size=32,
)
```
Now we can train our model:
```py
device = get_default_device()
model, _ = train(
    model,
    prior,
    lr = 4e-3,
    device = device
)
```
and finally we can instantiate our classifier:
```py
clf = NanoTabPFNClassifier(model, device)
```
and use its `.fit`, `.predict` and `.predict_proba`:
```py
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)
pred = clf.predict(X_test)
print('ROC AUC', roc_auc_score(y_test, prob))
print('Accuracy', accuracy_score(y_test, pred))
```


