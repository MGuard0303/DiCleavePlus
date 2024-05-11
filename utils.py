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
