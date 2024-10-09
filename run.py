import math
import pickle

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import logics
import dmodel
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training codes for SPOT-RNA secondary structure
# Load dataset and separate data for k-fold
path = "dataset/rnafold/dataset_b/dataset_b.csv"
df = pd.read_csv(path)
fold_size, _ = divmod(len(df), 5)

# Load pre-precessed data
with open("dataset/rnafold/dataset_b/pre_processed.pickle", "rb") as f:
    pp = pickle.load(f)

HIDDEN_FEATURE = 32

sequence_embedding = (dmodel.EmbeddingLayer(hidden_feature=HIDDEN_FEATURE, softmax_dim=1, is_secondary_structure=False).
                      to(device))
secondary_embedding = (dmodel.EmbeddingLayer(hidden_feature=HIDDEN_FEATURE, softmax_dim=1, is_secondary_structure=True).
                       to(device))

union_fusion_sequence = (dmodel.AttentionalFeatureFusionLayer(glo_pool_size=(200, HIDDEN_FEATURE), pool_type="2d").
                         to(device))
union_fusion_pattern = (dmodel.AttentionalFeatureFusionLayer(glo_pool_size=(14, HIDDEN_FEATURE), pool_type="2d").
                        to(device))

for fold in range(5):
    # Get training data and evaluation data
    seq_trn_1, seq_eval_1 = utils.separate_tensor(inputs=pp["seq_1"], curr_fold=fold, total_fold=5, fold_size=fold_size)
    seq_trn_2, seq_eval_2 = utils.separate_tensor(inputs=pp["seq_2"], curr_fold=fold, total_fold=5, fold_size=fold_size)
    seq_trn_3, seq_eval_3 = utils.separate_tensor(inputs=pp["seq_3"], curr_fold=fold, total_fold=5, fold_size=fold_size)

    db_trn_1, db_eval_1 = utils.separate_tensor(inputs=pp["db_1"], curr_fold=fold, total_fold=5, fold_size=fold_size)
    db_trn_2, db_eval_2 = utils.separate_tensor(inputs=pp["db_2"], curr_fold=fold, total_fold=5, fold_size=fold_size)
    db_trn_3, db_eval_3 = utils.separate_tensor(inputs=pp["db_3"], curr_fold=fold, total_fold=5, fold_size=fold_size)

    patt_trn_1, patt_eval_1 = utils.separate_tensor(inputs=pp["patt_1"], curr_fold=fold, total_fold=5,
                                                    fold_size=fold_size)
    patt_trn_2, patt_eval_2 = utils.separate_tensor(inputs=pp["patt_2"], curr_fold=fold, total_fold=5,
                                                    fold_size=fold_size)
    patt_trn_3, patt_eval_3 = utils.separate_tensor(inputs=pp["patt_3"], curr_fold=fold, total_fold=5,
                                                    fold_size=fold_size)

    patt_db_trn_1, patt_db_eval_1 = utils.separate_tensor(inputs=pp["patt_db_1"], curr_fold=fold, total_fold=5,
                                                          fold_size=fold_size)
    patt_db_trn_2, patt_db_eval_2 = utils.separate_tensor(inputs=pp["patt_db_2"], curr_fold=fold, total_fold=5,
                                                          fold_size=fold_size)
    patt_db_trn_3, patt_db_eval_3 = utils.separate_tensor(inputs=pp["patt_db_3"], curr_fold=fold, total_fold=5,
                                                          fold_size=fold_size)

    lbl_trn, lbl_eval = utils.separate_tensor(inputs=pp["label"], curr_fold=fold, total_fold=5, fold_size=fold_size)

    seq_trn = sequence_embedding(seq_trn_1, seq_trn_2, seq_trn_3)
    seq_eval = sequence_embedding(seq_eval_1, seq_eval_2, seq_eval_3)
    db_trn = secondary_embedding(db_trn_1, db_trn_2, db_trn_3)
    db_eval = secondary_embedding(db_eval_1, db_eval_2, db_eval_3)
    patt_trn = sequence_embedding(patt_trn_1, patt_trn_2, patt_trn_3)
    patt_eval = sequence_embedding(patt_eval_1, patt_eval_2, patt_eval_3)
    patt_db_trn = secondary_embedding(patt_db_trn_1, patt_db_trn_2, patt_db_trn_3)
    patt_db_eval = secondary_embedding(patt_db_eval_1, patt_db_eval_2, patt_db_eval_3)

    sequence_trn, _ = union_fusion_sequence(seq_trn, db_trn)
    sequence_eval, _ = union_fusion_sequence(seq_eval, db_eval)
    pattern_trn, _ = union_fusion_pattern(patt_trn, patt_db_trn)
    pattern_eval, _ = union_fusion_pattern(patt_eval, patt_db_eval)

    # Get validation data from training data
    vld_size = math.floor(len(seq_trn_1) * 0.1)
    idx_perm = torch.randperm(len(seq_trn_1))
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

    dl_trn = DataLoader(ds_trn, batch_size=128, shuffle=True)
    dl_vld = DataLoader(ds_vld, batch_size=128, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=128, shuffle=False)

    # Initial model
    model = dmodel.TFModel(embed_feature=32, hidden_feature=64)
    model.loss_function = torch.nn.NLLLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    # Training setup.
    epoch = 50
    print(f"fold_{fold}")
    model_queue, model_fnl = logics.train(model=model, train_loader=dl_trn, valid_loader=dl_vld, epochs=epoch,
                                          valid_per_epochs=5, returns=True)

    # Delete files if necessary.
    try:
        utils.delete_files()
    except FileNotFoundError as e:
        print(f"Catch error {e}")
    except IsADirectoryError as e:
        print(f"Catch error {e}")

    # Evaluation setup.
    for idx, mdl in enumerate(model_queue.queue):
        mdl.name = f"model{idx}_fold{fold}"
        utils.save_parameter(model=mdl, path="env/20241009", filename=f"{mdl.name}.pt")
        logics.evaluate(model=mdl, eval_loader=dl_eval)

    model_fnl.name = f"model_fnl_fold{fold}"
    utils.save_parameter(model=model_fnl, path="env/20241009", filename=f"{model_fnl.name}.pt")
    logics.evaluate(model=model_fnl, eval_loader=dl_eval)
