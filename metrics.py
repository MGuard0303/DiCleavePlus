"""
The sklearn package uses the Keras style, which the True value in the first place and the Predicted value in the
second place. However, These "get" methods use the PyTorch style, where the Predicted value is in the
first place, followed by the True value.
"""


import math

import torch


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
