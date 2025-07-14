import pickle
from pathlib import Path

import torch

import dlmodel
import logics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluation for DiCleavePlus.
data_path = Path("expt/20250713/aff_f_14_2/epoch_25/1")
param_path = Path("expt/20250713/aff_f_14_2/epoch_25/1")

with open(f"{data_path}/dl_eval_fold1_144440.pkl", "rb") as f:
    dl_eval_fold1 = pickle.load(f)
    
with open(f"{data_path}/dl_eval_fold2_154744.pkl", "rb") as f:
    dl_eval_fold2 = pickle.load(f)
    
with open(f"{data_path}/dl_eval_fold3_165046.pkl", "rb") as f:
    dl_eval_fold3 = pickle.load(f)
    
with open(f"{data_path}/dl_eval_fold4_175351.pkl", "rb") as f:
    dl_eval_fold4 = pickle.load(f)
    
with open(f"{data_path}/dl_eval_fold5_185653.pkl", "rb") as f:
    dl_eval_fold5 = pickle.load(f)
    
dl_eval = [dl_eval_fold1, dl_eval_fold2, dl_eval_fold3, dl_eval_fold4, dl_eval_fold5]

params = {
    "fold1": [file for file in param_path.glob("*.pt") if "fold1" in file.name],
    "fold2": [file for file in param_path.glob("*.pt") if "fold2" in file.name],
    "fold3": [file for file in param_path.glob("*.pt") if "fold3" in file.name],
    "fold4": [file for file in param_path.glob("*.pt") if "fold4" in file.name],
    "fold5": [file for file in param_path.glob("*.pt") if "fold5" in file.name],
}

for fold in range(1, 6):
    for param in params[f"fold{fold}"]:
        model = dlmodel.ModelAffFlex(
            embed_feature=2*32,
            pattern_size=14
        )

        loss_fn_weight = torch.ones(14)
        loss_fn_weight[0] = 0.05
        model.loss_function = torch.nn.NLLLoss(weight=loss_fn_weight)

        model.load_state_dict(torch.load(param, map_location=device))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl_eval[fold-1])
