import pickle
from pathlib import Path

import torch

import dmodel
import logics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
