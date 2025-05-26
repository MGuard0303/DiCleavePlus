import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Obtain k-mer for a given sequence
def kmer(seq: str, k: int) -> list:
    kmers = []
    s_len = len(seq)
    e = s_len - k + 1

    for i in range(e):
        t = seq[i:i+k]
        kmers.append(t)

    for i in range(e, s_len):
        delta = s_len - i
        t = seq[i:i+delta] + "<PH>"*(k-delta)
        kmers.append(t)

    return kmers


# Pad k-mer string list to designated length
def pad_kmer(k_mer: list, max_length: int) -> list:
    length = len(k_mer)
    delta = max_length - length

    for i in range(delta):
        k_mer.append("<PAD>")

    return k_mer


# Convert kmer string list into pytorch Tensor via kmer vocabulary
def convert_kmer(k_mer: list, vocab: dict, dtype: torch.dtype = None) -> torch.Tensor:
    temp = []

    for string in k_mer:
        temp.append(vocab[string])

    if dtype is None:
        ts_k_mer = torch.tensor(temp)
    else:
        ts_k_mer = torch.tensor(temp, dtype=dtype)

    return ts_k_mer


# Obtain kmer embedding from inputs
def kmer_embed(inputs: list | np.ndarray, vocab: dict, k: int, is_pad: bool = False, max_length: int = None,
               dtype: torch.dtype = None) -> torch.Tensor:
    k_tensors = []

    for i in inputs:
        k_mer = kmer(seq=i, k=k)

        if is_pad is True:
            k_mer = pad_kmer(k_mer=k_mer, max_length=max_length)

        k_tensor = convert_kmer(k_mer=k_mer, vocab=vocab, dtype=dtype)
        k_tensor = k_tensor.unsqueeze(0)
        k_tensors.append(k_tensor)

    t = torch.cat(k_tensors, dim=0)

    # Deprecated code.
    """
    t = torch.cat([k_tensors[0], k_tensors[1]], dim=0)

    for i in range(2, len(k_tensors)):
        t = torch.cat([t, k_tensors[i]], dim=0)
    """

    return t


# Separate training set and testing set for a specific fold
# Return a tuple of (training_tensor, evaluation_tensor)
def separate_tensor(inputs: torch.Tensor, curr_fold: int, total_fold: int, fold_size: int) -> tuple:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if curr_fold == 0:
        evaluate = inputs[0:fold_size]
        train = inputs[fold_size:]
    elif curr_fold == total_fold - 1:
        evaluate = inputs[curr_fold * fold_size:]
        train = inputs[0:curr_fold * fold_size]
    else:
        evaluate = inputs[curr_fold * fold_size:(curr_fold + 1) * fold_size]
        trn1 = inputs[0:curr_fold * fold_size]
        trn2 = inputs[(curr_fold + 1) * fold_size:]
        train = torch.cat([trn1, trn2], dim=0)

    train = train.to(device)
    evaluate = evaluate.to(device)

    return train, evaluate


# Save model state dictionary to path
def save_parameter(model: nn.Module, path: str, filename: str) -> None:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    dest = Path(f"{path}/{filename}")
    torch.save(model.state_dict(), dest)


class ModelQ:
    def __init__(self, k: int):
        self.queue = deque()
        self.k = k

    def __repr__(self):
        return f"{self.queue}"

    def size(self):
        size = len(self.queue)
        return size

    # Main method of this class
    def stack(self, model):
        if len(self.queue) < self.k:
            self.queue.append(model)
        else:
            self.queue.popleft()
            self.queue.append(model)
