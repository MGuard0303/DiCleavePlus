from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from sklearn.metrics import confusion_matrix


def kmer(seq: str, k: int) -> list:
    """
    Obtain k-mer for a given sequence.

    :param seq:
    :param k:
    :return: A list containing k-mers of the given sequence
    """

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


def pad_kmer(k_mer: list, max_length: int) -> list:
    """
    Pad k-mer string list to designated length.

    :param k_mer:
    :param max_length:
    :return: A list of padded sequence k-mers.
    """

    length = len(k_mer)
    delta = max_length - length

    for i in range(delta):
        k_mer.append("<PAD>")

    return k_mer


def convert_kmer(k_mer: list, vocab: dict, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Convert k-mer string list into PyTorch Tensor via k-mer vocabulary.

    :param k_mer:
    :param vocab:
    :param dtype:
    :return: K-mer Tensor
    """

    temp = []

    for string in k_mer:
        temp.append(vocab[string])

    if dtype is None:
        ts_k_mer = torch.tensor(temp)
    else:
        ts_k_mer = torch.tensor(temp, dtype=dtype)

    return ts_k_mer


def kmer_embed(inputs: list | np.ndarray, vocab: dict, k: int, is_pad: bool = False, max_length: int = None,
               dtype: torch.dtype = None) -> torch.Tensor:
    """
    Obtain k-mer embedding from inputs.

    :param inputs: Sequences list or sequences ndarray
    :param vocab:
    :param k:
    :param is_pad:
    :param max_length:
    :param dtype:
    :return: k-mer Tensor of input sequences
    """

    k_tensors = []

    for i in inputs:
        k_mer = kmer(seq=i, k=k)

        if is_pad is True:
            k_mer = pad_kmer(k_mer=k_mer, max_length=max_length)

        k_tensor = convert_kmer(k_mer=k_mer, vocab=vocab, dtype=dtype)
        k_tensor = k_tensor.unsqueeze(0)
        k_tensors.append(k_tensor)

    t = torch.cat(k_tensors, dim=0)

    return t


def build_preprocessed(inputs: pd.DataFrame, vocab_sequence: dict, vocab_sec: dict) -> dict:
    sequence_tensor = kmer_embed(inputs=inputs["sequence"].to_numpy(), vocab=vocab_sequence, k=3, is_pad=True,
                                 max_length=200, dtype=torch.int64)
    sec_tensor = kmer_embed(inputs=inputs["sec"].to_numpy(), vocab=vocab_sec, k=3, is_pad=True, max_length=200,
                            dtype=torch.int64)
    pattern_tensor = kmer_embed(inputs=inputs["pattern"].to_numpy(), vocab=vocab_sequence, k=3, is_pad=False,
                                dtype=torch.int64)
    pattern_sec_tensor = kmer_embed(inputs["pattern_sec"].to_numpy(), vocab=vocab_sec, k=3, is_pad=False,
                                    dtype=torch.int64)
    label2_tensor = torch.tensor(inputs["labels2"].to_numpy(), dtype=torch.float32)

    preprocessed = {
        "sequence": sequence_tensor,
        "sec": sec_tensor,
        "pattern": pattern_tensor,
        "pattern_sec": pattern_sec_tensor,
        "label2": label2_tensor
    }

    return preprocessed


def separate_tensor(inputs: torch.Tensor, curr_fold: int, total_fold: int, fold_size: int) -> tuple:
    """
    Separate training set and evaluation set for a specific fold.

    :param inputs:
    :param curr_fold:
    :param total_fold:
    :param fold_size:
    :return: A tuple of (training_tensor, evaluation_tensor)
    """

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


def save_parameter(model: nn.Module, path: str, filename: str) -> None:
    """
    Save model state dictionary.

    :param model:
    :param path:
    :param filename:
    :return: None
    """

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


def plot_heatmap(pred: torch.Tensor, label: torch.Tensor, num_labels: int, title: str, is_save: bool = False,
                 save_path: str | Path = None) -> None:
    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        _, label_pred = torch.max(pred, dim=1)
        label_pred.cpu().numpy()

        label = label.detach()
        label = label.cpu().numpy()

        labels = [i for i in range(num_labels)]

        cm = confusion_matrix(label, label_pred)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        ax.set_xlabel("Predicted Label ", fontsize=14, fontweight="bold", labelpad=20)
        ax.set_ylabel("True Label", fontsize=14, fontweight="bold", labelpad=20)

        ax.tick_params(left=False, bottom=False)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight="bold", rotation=45)

        fig.tight_layout()
        plt.show()

        if is_save:
            fig.savefig(save_path, dpi=600, format="png")
