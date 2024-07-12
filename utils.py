import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sklearn import preprocessing


# This function generates one_hot_encoding from input sequences
# The shape of output is (Batch, Length, Dimension)
def one_hot_encoding(inputs: list | np.ndarray, token: list, pad: bool = False, max_length: int = None) -> torch.Tensor:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(token)
    one_hots = []

    for item in inputs:
        int_seq = label_encoder.transform(list(item))
        int_tensor = torch.tensor(int_seq)

        # Create a zeros tensor with size(Length, Dimension)
        oh_tensor = torch.zeros(len(int_tensor), len(token), dtype=torch.float)
        for i in range(len(int_tensor)):
            oh_tensor[i, int_tensor[i]] = 1.0

        if pad and max_length is not None:
            oh_tensor = torch.transpose(oh_tensor, dim0=0, dim1=1)
            oh_tensor = F.pad(oh_tensor, (0, max_length - oh_tensor.size(1)))
            oh_tensor = torch.transpose(oh_tensor, dim0=0, dim1=1)

        one_hots.append(oh_tensor)

    one_hot_tensor = torch.stack(one_hots)

    return one_hot_tensor


# Get the list of lengths of sequences
def get_length(inputs: list | np.ndarray) -> list:
    lengths = []

    for i in inputs:
        length = len(i)
        lengths.append(length)

    return lengths


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
def kmer_embed(inputs: list | np.ndarray, vocab: dict, k: int, pad: bool = False, max_length: int = None,
               dtype: torch.dtype = None) -> torch.Tensor:
    k_tensors = []

    for i in inputs:
        k_mer = kmer(seq=i, k=k)

        if pad is True:
            k_mer = pad_kmer(k_mer=k_mer, max_length=max_length)

        k_tensor = convert_kmer(k_mer=k_mer, vocab=vocab, dtype=dtype)
        k_tensor = k_tensor.unsqueeze(0)
        k_tensors.append(k_tensor)

    t = torch.cat([k_tensors[0], k_tensors[1]], dim=0)

    for i in range(2, len(k_tensors)):
        t = torch.cat([t, k_tensors[i]], dim=0)

    return t


# Separate training set and testing set for a specific fold
# Return a tuple of (training_tensor, test_tensor)
def separate_tensor(inputs: torch.Tensor, curr_fold: int, tol_fold: int, fold_size: int) -> tuple:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if curr_fold == 0:
        tst = inputs[0:fold_size]
        trn = inputs[fold_size:]

    elif curr_fold == tol_fold - 1:
        tst = inputs[curr_fold * fold_size:]
        trn = inputs[0:curr_fold * fold_size]

    else:
        tst = inputs[curr_fold * fold_size:(curr_fold + 1) * fold_size]
        trn1 = inputs[0:curr_fold * fold_size]
        trn2 = inputs[(curr_fold + 1) * fold_size:]
        trn = torch.cat([trn1, trn2], dim=0)

    trn = trn.to(device)
    tst = tst.to(device)

    return trn, tst


# Save model state dictionary to path
def save_parameter(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


class ModelQ:
    def __init__(self, k):
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


# Delete files with specified chars
def delete_files(path, chars):
    del_file = []
    files = os.listdir(path)

    for file in files:
        if file.find(chars) != -1:
            del_file.append(file)

    for f in del_file:
        del_path = os.path.join(path, f)

        # Check deleting a file
        if os.path.isfile(del_path):
            os.remove(del_path)
