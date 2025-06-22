import math

import torch
from sklearn.metrics import f1_score


# Extended Perfect Match Fraction (PMF), consider negative prediction and negative label.
def pmf_ext(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Calculate Extended Perfect Match Fraction (PMF), consider negative prediction and negative label.

    :param pred:
    :param label:
    :return: Extended PMF value
    """

    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        _, label_pred = torch.max(pred, dim=1)

        label = label.detach()

        corrected = 0
        num = len(pred)

        for i in range(num):
            if label_pred[i].item() == label[i].item():
                corrected += 1

        return corrected / num


def pmf(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Calculate Perfect Match Fraction (PMF), do not consider negative prediction and negative label.

    :param pred:
    :param label:
    :return: PMF value
    """

    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        _, label_pred = torch.max(pred, dim=1)

        label = label.detach()

        corrected = 0
        num = len(pred)

        for i in range(num):
            if label_pred[i] == 0 or label[i] == 0:
                num -= 1
            elif label_pred[i].item() == label[i].item():
                corrected += 1

        return corrected / num


def pse_ext(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Calculate Extended Positional Shift Error (PSE), consider negative prediction and negative label.

    :param pred:
    :param label:
    :return: Extended PSE value
    """

    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        label = label.detach()
        sum_delta = 0
        num = len(pred)

        _, label_pred = torch.max(pred, dim=1)

        for i in range(num):
            if label_pred[i].item() != label[i].item() and label_pred[i].item() * label[i].item() == 0:
                delta = 13
                sum_delta += delta
            else:
                delta = abs(label_pred[i].item() - label[i].item())
                sum_delta += delta

        return sum_delta / num


def pse(pred: torch.Tensor, label: torch.Tensor) -> float:
    """
    Calculate Positional Shift Error (PSE), do not consider negative prediction and negative label.

    :param pred:
    :param label:
    :return: PSE value
    """

    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        label = label.detach()
        sum_delta = 0
        num = len(pred)

        _, label_pred = torch.max(pred, dim=1)

        for i in range(num):
            if label_pred[i] == 0 or label[i] == 0:
                num -= 1
            else:
                delta = abs(label_pred[i].item() - label[i].item())
                sum_delta += delta

        return sum_delta / num


def binary_metric(pred: torch.Tensor, label: torch.Tensor) -> tuple:
    """
    Calculate confusion matrix for positive-negative binary assessment.

    :param pred:
    :param label:
    :return: A tuple consisting of (accuracy, specificity, sensitivity, mcc)
    """

    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        label = label.detach()

        epsilon = 1e-8
        tp, tn, fp, fn = 0, 0, 0, 0
        num = len(pred)

        _, label_pred = torch.max(pred, dim=1)

        for i in range(num):
            if label_pred[i].item() != 0:
                label_pred[i] = 1

            if label_pred[i].item() == 1 and label[i].item() != 0:
                tp += 1
            elif label_pred[i].item() == 0 and label[i].item() == 0:
                tn += 1
            elif label_pred[i].item() == 1 and label[i].item() == 0:
                fp += 1
            elif label_pred[i].item() == 0 and label[i].item() != 0:
                fn += 1

        acc = (tp + tn) / (tp + tn + fp + fn)
        spe = tn / (tn + fp)
        sen = tp / (tp + fn)
        mcc = ((tp * tn) - (fp * fn)) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + epsilon)

        return acc, spe, sen, mcc


def topk_acc(pred: torch.Tensor, label: torch.Tensor, k: int = 3) -> float:
    """
    Calculate top-k accuracy.

    :param pred:
    :param label:
    :param k:
    :return: top-k accuracy value
    """

    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        label = label.detach()

        num = len(pred)
        hit = 0

        _, top_k_pred = torch.topk(pred, k=k, dim=1)

        for i in range(num):
            if label[i] in top_k_pred[i]:
                hit += 1

        return hit / num


def multi_f1(pred: torch.Tensor, label: torch.Tensor, average: str) -> float:
    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        pred = pred.detach()
        pred = torch.exp(pred)
        label = label.detach()

        _, label_pred = torch.max(pred, dim=1)

        label = label.tolist()
        label_pred = label_pred.tolist()

        labels = [i for i in range(14)]
        f1 = f1_score(y_true=label, y_pred=label_pred, labels=labels, average=average)

        return f1
