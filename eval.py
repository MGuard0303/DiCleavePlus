"""
Evaluate saved models.
"""


from pathlib import Path

import torch

import dmodel
import logics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = Path("")

# Find and load saved evaluation DataLoaders.
# Load evaluation data
eval1 = ""
eval2 = ""
eval3 = ""
eval4 = ""

paras_path = [
    [file for file in path.glob("*.pt") if "fold1" in file.name],
    [file for file in path.glob("*.pt") if "fold2" in file.name],
    [file for file in path.glob("*.pt") if "fold3" in file.name],
    [file for file in path.glob("*.pt") if "fold4" in file.name],
    [file for file in path.glob("*.pt") if "fold5" in file.name],
]

for fold in range(5):
    for i in range(len(paras_path[fold])):
        model = dmodel.TFModel()
        model.loss_function = torch.nn.NLLLoss()
        model.load_state_dict(torch.load(paras_path[fold][i]))
        logics.evaluate(model)
