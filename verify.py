import pickle
from pathlib import Path

import torch

import dlmodel
import logics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dl_aff_ds1_path = Path("./verify/model-aff/dataset1")
dl_aff_ds2_path = Path("./verify/model-aff/dataset2")
param_aff_ds1_path = Path(f"{dl_aff_ds1_path}/param")
param_aff_ds2_path = Path(f"{dl_aff_ds2_path}/param")
dl_concat_ds1_path = Path("./verify/model-concat/dataset1")
dl_concat_ds2_path = Path("./verify/model-concat/dataset2")
param_concat_ds1_path = Path(f"{dl_concat_ds1_path}/param")
param_concat_ds2_path = Path(f"{dl_concat_ds2_path}/param")

embed_feature = 32

# Model-aff results on Dataset-1
dl_files = [file for file in dl_aff_ds1_path.iterdir() if file.is_file()]
param_files = [file for file in param_aff_ds1_path.iterdir() if file.is_file()]
dls = {
    "fold1": [file for file in dl_files if "fold1" in file.name],
    "fold2": [file for file in dl_files if "fold2" in file.name],
    "fold3": [file for file in dl_files if "fold3" in file.name],
    "fold4": [file for file in dl_files if "fold4" in file.name],
    "fold5": [file for file in dl_files if "fold5" in file.name]
}
params = {
    "fold1": [file for file in param_files if "fold1" in file.name],
    "fold2": [file for file in param_files if "fold2" in file.name],
    "fold3": [file for file in param_files if "fold3" in file.name],
    "fold4": [file for file in param_files if "fold4" in file.name],
    "fold5": [file for file in param_files if "fold5" in file.name]
}

print("Results of model-aff on Dataset-1")

for fold in range(1, 6):
    with open(dls[f"fold{fold}"][0], "rb") as f:
        dl = pickle.load(f)

    model = dlmodel.ModelAFF(
        embed_feature=2*embed_feature,
        linear_hidden_feature=64,
        num_attn_head=8,
        tf_dim_forward=256,
        num_tf_layer=3
    )
    model.loss_function = torch.nn.NLLLoss()

    for param in params[f"fold{fold}"]:
        model.name = f"model-aff-{param.name}"
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl)


# Model-aff results on Dataset-2
dl_files = [file for file in dl_aff_ds2_path.iterdir() if file.is_file()]
param_files = [file for file in param_aff_ds2_path.iterdir() if file.is_file()]
dls = {
    "fold1": [file for file in dl_files if "fold1" in file.name],
    "fold2": [file for file in dl_files if "fold2" in file.name],
    "fold3": [file for file in dl_files if "fold3" in file.name],
    "fold4": [file for file in dl_files if "fold4" in file.name],
    "fold5": [file for file in dl_files if "fold5" in file.name]
}
params = {
    "fold1": [file for file in param_files if "fold1" in file.name],
    "fold2": [file for file in param_files if "fold2" in file.name],
    "fold3": [file for file in param_files if "fold3" in file.name],
    "fold4": [file for file in param_files if "fold4" in file.name],
    "fold5": [file for file in param_files if "fold5" in file.name]
}

print("Results of model-aff on Dataset-2")

for fold in range(1, 6):
    with open(dls[f"fold{fold}"][0], "rb") as f:
        dl = pickle.load(f)

    model = dlmodel.ModelAFF(
        embed_feature=2 * embed_feature,
        linear_hidden_feature=64,
        num_attn_head=8,
        tf_dim_forward=256,
        num_tf_layer=3
    )
    model.loss_function = torch.nn.NLLLoss()

    for param in params[f"fold{fold}"]:
        model.name = f"model-aff-{param.name}"
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl)


# Model-concat results on Dataset-1
dl_files = [file for file in dl_concat_ds1_path.iterdir() if file.is_file()]
param_files = [file for file in param_concat_ds1_path.iterdir() if file.is_file()]
dls = {
    "fold1": [file for file in dl_files if "fold1" in file.name],
    "fold2": [file for file in dl_files if "fold2" in file.name],
    "fold3": [file for file in dl_files if "fold3" in file.name],
    "fold4": [file for file in dl_files if "fold4" in file.name],
    "fold5": [file for file in dl_files if "fold5" in file.name]
}
params = {
    "fold1": [file for file in param_files if "fold1" in file.name],
    "fold2": [file for file in param_files if "fold2" in file.name],
    "fold3": [file for file in param_files if "fold3" in file.name],
    "fold4": [file for file in param_files if "fold4" in file.name],
    "fold5": [file for file in param_files if "fold5" in file.name]
}

print("Results of model-concat on Dataset-1")

for fold in range(1, 6):
    with open(dls[f"fold{fold}"][0], "rb") as f:
        dl = pickle.load(f)

    model = dlmodel.ModelConcat(
        embed_feature=2 * embed_feature,
        linear_hidden_feature=64,
        num_attn_head=8,
        tf_dim_forward=256,
        num_tf_layer=3
    )
    model.loss_function = torch.nn.NLLLoss()

    for param in params[f"fold{fold}"]:
        model.name = f"model-concat-{param.name}"
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl)


# Model-concat results on Dataset-2
dl_files = [file for file in dl_concat_ds2_path.iterdir() if file.is_file()]
param_files = [file for file in param_concat_ds2_path.iterdir() if file.is_file()]
dls = {
    "fold1": [file for file in dl_files if "fold1" in file.name],
    "fold2": [file for file in dl_files if "fold2" in file.name],
    "fold3": [file for file in dl_files if "fold3" in file.name],
    "fold4": [file for file in dl_files if "fold4" in file.name],
    "fold5": [file for file in dl_files if "fold5" in file.name]
}
params = {
    "fold1": [file for file in param_files if "fold1" in file.name],
    "fold2": [file for file in param_files if "fold2" in file.name],
    "fold3": [file for file in param_files if "fold3" in file.name],
    "fold4": [file for file in param_files if "fold4" in file.name],
    "fold5": [file for file in param_files if "fold5" in file.name]
}

print("Results of model-concat on Dataset-2")
for fold in range(1, 6):
    with open(dls[f"fold{fold}"][0], "rb") as f:
        dl = pickle.load(f)

    model = dlmodel.ModelConcat(
        embed_feature=2 * embed_feature,
        linear_hidden_feature=64,
        num_attn_head=8,
        tf_dim_forward=256,
        num_tf_layer=3
    )
    model.loss_function = torch.nn.NLLLoss()

    for param in params[f"fold{fold}"]:
        model.name = f"model-concat-{param.name}"
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl)
