import datetime
import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import logics
import Model
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load dataset and separate data for k-fold
path = "./dataset/spotrna/dataset_1.csv"
df = pd.read_csv(path)
n_fold, _ = divmod(len(df), 5)
seq_token = ["A", "C", "G", "U"]
db_token = [".", "(", ")", "[", "]", "{", "}", "<", ">"]


for fold in range(5):
    if fold != 4:
        df_tst = df[fold * n_fold:(fold + 1) * n_fold]
    else:
        df_tst = df[fold * n_fold:]

    df_trn = df.loc[~df.index.isin(df_tst.index)]
    df_vld = df_trn.sample(frac=0.1)
    df_trn = df_trn.loc[~df_trn.index.isin(df_vld.index)]

    df_trn = df_trn.sample(frac=1.0).reset_index(drop=True)
    df_vld = df_vld.sample(frac=1.0).reset_index(drop=True)
    df_tst = df_tst.sample(frac=1.0).reset_index(drop=True)

    # Training data processing
    seq_trn = utils.one_hot_encoding(inputs=df_trn["sequence"].to_numpy(), token=seq_token, pad=True, max_length=200)
    db_trn = utils.one_hot_encoding(inputs=df_trn["db"].to_numpy(), token=db_token, pad=True, max_length=200)
    sequence_trn = torch.cat([seq_trn, db_trn], dim=2).to(device)
    patt_trn = utils.one_hot_encoding(inputs=df_trn["pattern"].to_numpy(), token=seq_token)
    patt_db_trn = utils.one_hot_encoding(inputs=df_trn["pattern_db"].to_numpy(), token=db_token)
    pattern_trn = torch.cat([patt_trn, patt_db_trn], dim=2).to(device)
    label_trn = torch.tensor(df_trn["label_2"].to_numpy(), dtype=torch.float).unsqueeze(1).to(device)

    # Validation data processing
    seq_vld = utils.one_hot_encoding(inputs=df_vld["sequence"].to_numpy(), token=seq_token, pad=True, max_length=200)
    db_vld = utils.one_hot_encoding(inputs=df_vld["db"].to_numpy(), token=db_token, pad=True, max_length=200)
    sequence_vld = torch.cat([seq_vld, db_vld], dim=2).to(device)
    patt_vld = utils.one_hot_encoding(inputs=df_vld["pattern"].to_numpy(), token=seq_token)
    patt_db_vld = utils.one_hot_encoding(inputs=df_vld["pattern_db"].to_numpy(), token=db_token)
    pattern_vld = torch.cat([patt_vld, patt_db_vld], dim=2).to(device)
    label_vld = torch.tensor(df_vld["label_2"].to_numpy(), dtype=torch.float).unsqueeze(1).to(device)

    # Testing data processing
    seq_tst = utils.one_hot_encoding(inputs=df_tst["sequence"].to_numpy(), token=seq_token, pad=True, max_length=200)
    db_tst = utils.one_hot_encoding(inputs=df_tst["db"].to_numpy(), token=db_token, pad=True, max_length=200)
    sequence_tst = torch.cat([seq_tst, db_tst], dim=2).to(device)
    patt_tst = utils.one_hot_encoding(inputs=df_tst["pattern"].to_numpy(), token=seq_token)
    patt_db_tst = utils.one_hot_encoding(inputs=df_tst["pattern_db"].to_numpy(), token=db_token)
    pattern_tst = torch.cat([patt_tst, patt_db_tst], dim=2).to(device)
    label_tst = torch.tensor(df_tst["label_2"].to_numpy(), dtype=torch.float).unsqueeze(1).to(device)

    ds_trn = TensorDataset(sequence_trn, pattern_trn, label_trn)
    ds_vld = TensorDataset(sequence_vld, pattern_vld, label_vld)
    ds_tst = TensorDataset(sequence_tst, pattern_tst, label_tst)

    dl_trn = DataLoader(ds_trn, batch_size=128, shuffle=True)
    dl_vld = DataLoader(ds_vld, batch_size=128, shuffle=True)
    dl_tst = DataLoader(ds_tst, batch_size=128, shuffle=False)

    # Initial model
    model = Model.DModel(hidden_feature=32)
    model.loss_function = torch.nn.NLLLoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.to(device)

    epoch = 100
    print(f"fold_{fold}")
    model_q, last_mdl = logics.train(model=model, train_loader=dl_trn, valid_loader=dl_vld, epochs=epoch,
                                     valid_per_epochs=5, returns=True)

    best_mdl = model_q.queue.popleft()
    best_mdl.eval()
    logics.predict(model=best_mdl, test_loader=dl_tst)
