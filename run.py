import datetime
import math
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import dmodel
import logics
import utils


date = datetime.datetime.now().strftime("%Y%m%d")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training codes for SPOT-RNA secondary structure
# Load dataset and separate data for k-fold
path = "dataset/rnafold/dataset_adj/dataset_adj.csv"
df = pd.read_csv(path)
fold_size, _ = divmod(len(df), 5)

# Load pre-precessed data
with open("dataset/rnafold/dataset_adj/preprocessed.pickle", "rb") as f:
    pp = pickle.load(f)

embed_feature = 32

embedding_layer_seq = torch.nn.Embedding(num_embeddings=85, embedding_dim=embed_feature, padding_idx=0).to(device)
embedding_layer_sec = torch.nn.Embedding(num_embeddings=40, embedding_dim=embed_feature, padding_idx=0).to(device)

union_fusion_seq = dmodel.AttentionalFeatureFusionLayer(glo_pool_size=(200, embed_feature), pool_type="2d").to(device)
union_fusion_patt = dmodel.AttentionalFeatureFusionLayer(glo_pool_size=(14, embed_feature), pool_type="2d").to(device)

# for fold in range(adj, 6):
for fold in range(1, 2):
    # Get training data and evaluation data
    seq_trn, seq_eval = utils.separate_tensor(inputs=pp["seq_3"], curr_fold=fold, total_fold=5, fold_size=fold_size)
    db_trn, db_eval = utils.separate_tensor(inputs=pp["db_3"], curr_fold=fold, total_fold=5, fold_size=fold_size)
    patt_trn, patt_eval = utils.separate_tensor(inputs=pp["patt_3"], curr_fold=fold, total_fold=5, fold_size=fold_size)
    patt_db_trn, patt_db_eval = utils.separate_tensor(inputs=pp["patt_db_3"], curr_fold=fold, total_fold=5,
                                                      fold_size=fold_size)
    lbl_trn, lbl_eval = utils.separate_tensor(inputs=pp["label"], curr_fold=fold, total_fold=5, fold_size=fold_size)

    seq_trn = embedding_layer_seq(seq_trn)
    seq_eval = embedding_layer_seq(seq_eval)
    db_trn = embedding_layer_sec(db_trn)
    db_eval = embedding_layer_sec(db_eval)
    patt_trn = embedding_layer_seq(patt_trn)
    patt_eval = embedding_layer_seq(patt_eval)
    patt_db_trn = embedding_layer_sec(patt_db_trn)
    patt_db_eval = embedding_layer_sec(patt_db_eval)

    sequence_trn, _ = union_fusion_seq(seq_trn, db_trn)
    sequence_eval, _ = union_fusion_seq(seq_eval, db_eval)
    pattern_trn, _ = union_fusion_patt(patt_trn, patt_db_trn)
    pattern_eval, _ = union_fusion_patt(patt_eval, patt_db_eval)

    # Get validation data from training data
    ori_trn_size = len(sequence_trn)
    vld_size = math.floor(ori_trn_size * 0.1)
    idx_perm = torch.randperm(ori_trn_size)
    idx_vld = idx_perm[:vld_size]
    idx_trn = idx_perm[vld_size:]

    sequence_vld = sequence_trn[idx_vld]
    sequence_trn = sequence_trn[idx_trn]

    pattern_vld = pattern_trn[idx_vld]
    pattern_trn = pattern_trn[idx_trn]

    lbl_vld = lbl_trn[idx_vld]
    lbl_trn = lbl_trn[idx_trn]

    # Wrap data to Dataset
    ds_trn = TensorDataset(sequence_trn, pattern_trn, lbl_trn)
    ds_vld = TensorDataset(sequence_vld, pattern_vld, lbl_vld)
    ds_eval = TensorDataset(sequence_eval, pattern_eval, lbl_eval)

    dl_trn = DataLoader(ds_trn, batch_size=256, shuffle=True)
    dl_vld = DataLoader(ds_vld, batch_size=256, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=256, shuffle=False)

    # Save evaluation data for each fold.
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    path = Path(f"expt/{date}/tf")

    if not path.exists():
        path.mkdir(parents=True)

    with open(f"{path}/dl_eval_fold{fold}_{timestamp}.pkl", "wb") as f:
        pickle.dump(dl_eval, f)

    # Initial model
    model = dmodel.TFModel(
        embed_feature=embed_feature,
        linear_hidden_feature=128,
        num_attn_head=8,
        tf_dim_forward=512,
        num_tf_layer=6,
        seq_conv_kernel_size=5,
        patt_conv_kernel_size=3,
    )
    model.loss_function = torch.nn.NLLLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.to(device)

    # Training setup.
    epoch = 70
    print(f"fold_{fold}")
    model_queue, model_fnl = logics.train(model=model, train_loader=dl_trn, valid_loader=dl_vld, epochs=epoch,
                                          valid_per_epochs=5, returns=True)

    # Evaluation setup.
    for idx, mdl in enumerate(model_queue.queue, start=1):
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        mdl.name = f"model{idx}_fold{fold}_{timestamp}"
        utils.save_parameter(model=mdl, path=f"expt/{date}/tf", filename=f"{mdl.name}.pt")
        logics.evaluate(model=mdl, eval_loader=dl_eval)

    timestamp = datetime.datetime.now().strftime("%H%M%S")
    model_fnl.name = f"model_fnl_fold{fold}_{timestamp}"
    utils.save_parameter(model=model_fnl, path=f"expt/{date}/tf", filename=f"{model_fnl.name}.pt")
    logics.evaluate(model=model_fnl, eval_loader=dl_eval)
