import math

import torch
import sklearn


"""
# Return a confusion matrix whose row indicates true label and column indicates prediction
def confusion_mtx(predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    if len(predict) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        predict = predict.detach()
        label = label.detach()
        mtx = torch.zeros(3, 3, dtype=torch.int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        _, idx = torch.max(predict, dim=1, keepdim=True)

        for i in range(len(label)):
            mtx[label[i], idx[i]] += 1

        return mtx


# Calculate accuracy, precision, recall, f1 and mcc from confusion matrix
# Use macro-average approach
def metrics(matrix: torch.Tensor) -> list:
    epsilon = 1e-8
    rows = matrix.size(0)
    columns = matrix.size(1)

    if rows != columns:
        raise ValueError("Number of rows and number of columns does not match.")
    else:
        pres = []
        recs = []
        fs = []

        c = torch.sum(torch.diagonal(matrix)).item()
        s = torch.sum(matrix).item()
        pks = []
        tks = []

        acc = c / (s + epsilon)

        for i in range(rows):
            tp = matrix[i, i].item()
            fp = torch.sum(matrix[:, i] - tp)
            fn = torch.sum(matrix[i, :] - tp)

            p = tp / (tp + fp + epsilon)
            r = tp / (tp + fn + epsilon)
            f = 2 * ((p * r) / (p + r + epsilon))

            pres.append(p)
            recs.append(r)
            fs.append(f)

        avg_pre = sum(pres) / (len(pres) + epsilon)
        avg_rec = sum(recs) / (len(recs) + epsilon)
        avg_f = sum(fs) / (len(fs) + epsilon)

        # Calculate mcc
        for i in range(rows):
            pk = torch.sum(matrix[:, i]).item()
            tk = torch.sum(matrix[i, :]).item()
            pks.append(pk)
            tks.append(tk)

        p_multi_t = [float(pks[i] * tks[i]) for i in range(len(pks))]
        p_square = [math.pow(pks[i], 2) for i in range(len(pks))]
        t_square = [math.pow(tks[i], 2) for i in range(len(tks))]

        numerator = (c * s) - sum(p_multi_t)
        denominator = math.sqrt((math.pow(s, 2) - sum(p_square)) * (math.pow(s, 2) - sum(t_square)))
        mcc = numerator / (denominator + epsilon)

        return [acc, avg_pre, avg_rec, avg_f, mcc]
"""


# Extended Perfect Match Fraction (PMF), consider negative prediction and negative label.
def pmf_ext(pred: torch.Tensor, label: torch.Tensor) -> float:
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


# Perfect Match Fraction (PMF), do not consider negative samples for prediction and label.
def pmf(pred: torch.Tensor, label: torch.Tensor) -> float:
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


# Extended Positional Shift Error (PSE), consider negative prediction and negative label.
def pse_ext(pred: torch.Tensor, label: torch.Tensor) -> float:
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


# Extended Positional Shift Error (PSE), consider negative prediction and negative label
def pse(pred: torch.Tensor, label: torch.Tensor) -> float:
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


# Metrics for positive-negative binary assessment.
# Currently return accuracy, specificity, sensitivity, mcc.
def binary_metric(pred: torch.Tensor, label: torch.Tensor) -> tuple:
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


# Calculate top-k accuracy.
def topk_acc(pred: torch.Tensor, label: torch.Tensor, k: int = 3) -> float:
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


def f1_score(pred: list, label: list, average: str) -> float:
    if len(pred) != len(label):
        raise ValueError("The length of prediction tensor and label tensor does not match.")
    else:
        labels = [i for i in range(14)]
        f1 = sklearn.metrics.f1_score(y_true=labels, y_pred=pred, labels=labels, average=average)

        return f1
