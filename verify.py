import pickle
from pathlib import Path

import torch

import dmodel
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

# model-aff results on Dataset-1
