import pickle
from pathlib import Path

import torch

import dlmodel
import logics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluation for DiCleavePlus.
"""
data_path = 
param_path = 

with open(f"{data_path}/", "rb") as f:
    dl_eval_fold1 = pickle.load(f)
    
with open(f"{data_path}/", "rb") as f:
    dl_eval_fold2 = pickle.load(f)
    
with open(f"{data_path}/", "rb") as f:
    dl_eval_fold3 = pickle.load(f)
    
with open(f"{data_path}/", "rb") as f:
    dl_eval_fold4 = pickle.load(f)
    
with open(f"{data_path}/", "rb") as f:
    dl_eval_fold5 = pickle.load(f)
    
dl_eval = [dl_eval_fold1, dl_eval_fold2, dl_eval_fold3, dl_eval_fold4, dl_eval_fold5]

params = {
    "fold1": [file for file in param_path.glob(*.pt) if "fold1" in file.name],
    "fold2": [file for file in param_path.glob(*.pt) if "fold2" in file.name],
    "fold3": [file for file in param_path.glob(*.pt) if "fold3" in file.name],
    "fold4": [file for file in param_path.glob(*.pt) if "fold4" in file.name],
    "fold5": [file for file in param_path.glob(*.pt) if "fold5" in file.name],
}

for fold in range(1, 6):
    for param in params[f"fold{fold}"]:
        model =
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl_eval[fold-1])
"""
