import datetime
import math
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import dlmodel
import logics
import utils


# Hyper parameters.
date = datetime.datetime.now().strftime("%Y%m%d")
task = "aff_14_1"  # "model type, pattern size, dataset type".
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pattern_size = 14
epoch_size = 25


# Load dataset and separate data for k-fold.
dataset_path = "dataset/luna/human/dataset_14_1.csv"
df = pd.read_csv(dataset_path)
fold_size, _ = divmod(len(df), 5)

# Load pre-precessed data.
with open("dataset/luna/human/preprocessed_14_1.pkl", "rb") as f:
    pp = pickle.load(f)

embed_feature = 32

embedding_layer_seq = torch.nn.Embedding(num_embeddings=85, embedding_dim=embed_feature, padding_idx=0).to(device)
embedding_layer_sec = torch.nn.Embedding(num_embeddings=40, embedding_dim=embed_feature, padding_idx=0).to(device)

for fold in range(1, 6):
    # Get training data and evaluation data.
    seq_trn, seq_eval = utils.separate_tensor(inputs=pp["sequence"], curr_fold=fold-1, total_fold=5,
                                              fold_size=fold_size)
    sec_trn, sec_eval = utils.separate_tensor(inputs=pp["sec"], curr_fold=fold - 1, total_fold=5, fold_size=fold_size)
    patt_trn, patt_eval = utils.separate_tensor(inputs=pp["pattern"], curr_fold=fold-1, total_fold=5,
                                                fold_size=fold_size)
    patt_sec_trn, patt_sec_eval = utils.separate_tensor(inputs=pp["pattern_sec"], curr_fold=fold - 1, total_fold=5,
                                                        fold_size=fold_size)
    lbl2_trn, lbl2_eval = utils.separate_tensor(inputs=pp["label2"], curr_fold=fold - 1, total_fold=5,
                                                fold_size=fold_size)

    seq_trn = embedding_layer_seq(seq_trn)
    seq_eval = embedding_layer_seq(seq_eval)
    sec_trn = embedding_layer_sec(sec_trn)
    sec_eval = embedding_layer_sec(sec_eval)
    patt_trn = embedding_layer_seq(patt_trn)
    patt_eval = embedding_layer_seq(patt_eval)
    patt_sec_trn = embedding_layer_sec(patt_sec_trn)
    patt_sec_eval = embedding_layer_sec(patt_sec_eval)

    sequence_trn = torch.cat((seq_trn, sec_trn), dim=2)
    sequence_eval = torch.cat((seq_eval, sec_eval), dim=2)
    pattern_trn = torch.cat((patt_trn, patt_sec_trn), dim=2)
    pattern_eval = torch.cat((patt_eval, patt_sec_eval), dim=2)

    # Get validation data from training data.
    ori_trn_size = len(sequence_trn)
    vld_size = math.floor(ori_trn_size * 0.1)
    idx_perm = torch.randperm(ori_trn_size)
    idx_vld = idx_perm[:vld_size]
    idx_trn = idx_perm[vld_size:]

    sequence_vld = sequence_trn[idx_vld]
    sequence_trn = sequence_trn[idx_trn]

    pattern_vld = pattern_trn[idx_vld]
    pattern_trn = pattern_trn[idx_trn]

    lbl_vld = lbl2_trn[idx_vld]
    lbl2_trn = lbl2_trn[idx_trn]

    # Wrap data to Dataset and DataLoader.
    ds_trn = TensorDataset(sequence_trn, pattern_trn, lbl2_trn)
    ds_vld = TensorDataset(sequence_vld, pattern_vld, lbl_vld)
    ds_eval = TensorDataset(sequence_eval, pattern_eval, lbl2_eval)

    dl_trn = DataLoader(ds_trn, batch_size=256, shuffle=True)
    dl_vld = DataLoader(ds_vld, batch_size=256, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=256, shuffle=False)

    # Save evaluation data for each fold.
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    save_path = Path(f"expt/{date}/{task}/epoch_{epoch_size}")

    if not save_path.exists():
        save_path.mkdir(parents=True)

    with open(f"{save_path}/dl_eval_fold{fold}_{timestamp}.pkl", "wb") as f:
        pickle.dump(dl_eval, f)

    # Initial model
    model = dlmodel.ModelAffFlex(
        embed_feature=2*embed_feature,
        pattern_size=pattern_size,
        num_attn_head=8,
        tf_dim_forward=512,
        num_tf_layer=4,
        linear_hidden_feature=128,
    )
    loss_fn_weight = torch.ones(pattern_size)
    loss_fn_weight[0] = 0.05
    model.loss_function = torch.nn.NLLLoss(weight=loss_fn_weight.to(device))
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.to(device)

    # Training setup.
    print(f"fold{fold}")
    model_queue, model_fnl = logics.train(model=model, train_loader=dl_trn, valid_loader=dl_vld, epochs=epoch_size,
                                          valid_per_epochs=5, is_return=True)

    # Save model parameters.
    for idx, mdl in enumerate(model_queue.queue, start=1):
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        mdl.name = f"model{idx}_fold{fold}_{timestamp}"
        utils.save_parameter(model=mdl, path=f"{save_path}", filename=f"{mdl.name}.pt")

    # Save final model parameters each fold.
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    model_fnl.name = f"model_fnl_fold{fold}_{timestamp}"
    utils.save_parameter(model=model_fnl, path=f"{save_path}", filename=f"{model_fnl.name}.pt")
