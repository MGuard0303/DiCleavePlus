import pickle
from pathlib import Path

import torch

import dlmodel
import logics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_aff_1 = Path("./verification/model_aff_1")
path_aff_2 = Path("./verification/model_aff_2")
path_concat_1 = Path("./verification/model_concat_1")
path_concat_2 = Path("./verification/model_concat_2")

embed_feature = 32

# Verify results from model-aff on Dataset-1.
subdir = sorted([ele for ele in path_aff_1.iterdir() if ele.is_dir()])

for i in range(0, 5):
    offset = 2 * i
    pattern_size = 10 + offset

    dl_evals = {
        "fold1": tuple(subdir[i].glob("*fold1.pkl"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pkl"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pkl"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pkl"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pkl"))[0],
    }

    params = {
        "fold1": tuple(subdir[i].glob("*fold1.pt"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pt"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pt"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pt"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pt"))[0],
    }

    model = dlmodel.ModelAffFlex(
        embed_feature=embed_feature,
        pattern_size=pattern_size,
        name=f"model-aff-{pattern_size}"
    )
    loss_fn_weight = torch.ones(pattern_size)
    loss_fn_weight[0] = 0.5
    model.loss_function = torch.nn.NLLLoss(weight=loss_fn_weight)

    print(f"Verifying results of model-aff on Dataset-1, pattern size: {pattern_size}")

    for fold in range(1, 6):
        with open(dl_evals[f"fold{fold}"], "rb") as f:
            dl_eval = pickle.load(f)

        param = params[f"fold{fold}"]
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl_eval)


# Verify results from model-aff on Dataset-2.
subdir = sorted([ele for ele in path_aff_2.iterdir() if ele.is_dir()])

for i in range(0, 5):
    offset = 2 * i
    pattern_size = 10 + offset

    dl_evals = {
        "fold1": tuple(subdir[i].glob("*fold1.pkl"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pkl"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pkl"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pkl"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pkl"))[0],
    }

    params = {
        "fold1": tuple(subdir[i].glob("*fold1.pt"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pt"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pt"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pt"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pt"))[0],
    }

    model = dlmodel.ModelAffFlex(
        embed_feature=embed_feature,
        pattern_size=pattern_size,
        name=f"model-aff-{pattern_size}"
    )
    model.loss_function = torch.nn.NLLLoss()

    print(f"Verifying results of model-aff on Dataset-2, pattern size: {pattern_size}")

    for fold in range(1, 6):
        with open(dl_evals[f"fold{fold}"], "rb") as f:
            dl_eval = pickle.load(f)

        param = params[f"fold{fold}"]
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl_eval)


# Verify results from model-concat on Dataset-1.
subdir = sorted([ele for ele in path_concat_1.iterdir() if ele.is_dir()])

for i in range(0, 5):
    offset = 2 * i
    pattern_size = 10 + offset

    dl_evals = {
        "fold1": tuple(subdir[i].glob("*fold1.pkl"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pkl"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pkl"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pkl"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pkl"))[0],
    }

    params = {
        "fold1": tuple(subdir[i].glob("*fold1.pt"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pt"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pt"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pt"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pt"))[0],
    }

    model = dlmodel.ModelConcatFlex(
        embed_feature=embed_feature,
        pattern_size=pattern_size,
        name=f"model-aff-{pattern_size}"
    )
    loss_fn_weight = torch.ones(pattern_size)
    loss_fn_weight[0] = 0.5
    model.loss_function = torch.nn.NLLLoss(weight=loss_fn_weight)

    print(f"Verifying results of model-concat on Dataset-1, pattern size: {pattern_size}")

    for fold in range(1, 6):
        with open(dl_evals[f"fold{fold}"], "rb") as f:
            dl_eval = pickle.load(f)

        param = params[f"fold{fold}"]
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl_eval)


# Verify results from model-concat on Dataset-2.
subdir = sorted([ele for ele in path_concat_2.iterdir() if ele.is_dir()])

for i in range(0, 5):
    offset = 2 * i
    pattern_size = 10 + offset

    dl_evals = {
        "fold1": tuple(subdir[i].glob("*fold1.pkl"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pkl"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pkl"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pkl"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pkl"))[0],
    }

    params = {
        "fold1": tuple(subdir[i].glob("*fold1.pt"))[0],
        "fold2": tuple(subdir[i].glob("*fold2.pt"))[0],
        "fold3": tuple(subdir[i].glob("*fold3.pt"))[0],
        "fold4": tuple(subdir[i].glob("*fold4.pt"))[0],
        "fold5": tuple(subdir[i].glob("*fold5.pt"))[0],
    }

    model = dlmodel.ModelConcatFlex(
        embed_feature=embed_feature,
        pattern_size=pattern_size,
        name=f"model-aff-{pattern_size}"
    )
    model.loss_function = torch.nn.NLLLoss()

    print(f"Verifying results of model-concat on Dataset-2, pattern size: {pattern_size}")

    for fold in range(1, 6):
        with open(dl_evals[f"fold{fold}"], "rb") as f:
            dl_eval = pickle.load(f)

        param = params[f"fold{fold}"]
        model.load_state_dict(torch.load(param))
        model.to(device)
        logics.evaluate(model=model, eval_loader=dl_eval)
