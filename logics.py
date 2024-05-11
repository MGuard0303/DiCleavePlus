import copy

import torch

from torch.utils.data import DataLoader

import metrics
import Model
import utils


def train_step(model: Model.DModel, pattern: torch.Tensor, sequence: torch.Tensor, label: torch.Tensor) -> tuple:
    # Set to the training mode, dropout and batch normalization will work under this mode
    model.train()
    model.optimizer.zero_grad()  # Clear the gradient everytime

    # Forward propagation
    pred = model(pattern=pattern, sequence=sequence)
    lss = model.loss_function(pred, label)

    # Backward propagation
    lss.backward()
    model.optimizer.step()  # Update the parameters of the model

    return lss.item(), pred  # loss is a one-element tensor, so it can use .item() method


@torch.no_grad()  # This decorator makes following function not calculate gradient
def valid_step(model: Model.DModel, pattern: torch.Tensor, sequence: torch.Tensor, label: torch.Tensor) -> tuple:
    # Set to the evaluation mode, dropout and batch normalization will not work
    model.eval()

    pred = model(pattern=pattern, sequence=sequence)
    lss = model.loss_function(pred, label)

    return lss.item(), pred


def train(model: Model.DModel, train_loader: DataLoader, valid_loader: DataLoader, epochs: int, valid_per_epochs: int,
          returns: bool = False) -> tuple:
    print(f"Model {model.name}: Start training...")

    best_vld_acc = 0
    # TOLERANCE = 3
    # tol = 0
    model_queue = utils.ModelQ(1)  # Save the best model parameter

    for epoch in range(1, epochs + 1):
        epoch_lss = 0.0

        # Training step
        step_trn = len(train_loader)

        for _, (seq, patt, lbl) in enumerate(train_loader):
            lbl = lbl.squeeze(1)  # The shape of NLLLoss label is (N), it's different from BCELoss
            lbl = lbl.type(torch.long)
            batch_lss, _ = train_step(model=model, pattern=patt, sequence=seq, label=lbl)
            epoch_lss += batch_lss

        avg_epoch_lss = epoch_lss / step_trn
        print(f"Epoch {epoch:02}")
        print(f"| Average Training Loss: {avg_epoch_lss:.4f} |")

        # Validate model at certain epoch
        if epoch % valid_per_epochs == 0:
            print(f"Validating at Epoch {epoch:02}")
            total_vld_lss = 0.0
            total_vld_mtx = torch.zeros(14, 14, dtype=torch.int, device=torch.device("cuda:0") if
                                        torch.cuda.is_available() else "cpu")

            step_vld = len(valid_loader)

            for _, (seq, patt, lbl) in enumerate(valid_loader):
                lbl = lbl.squeeze(1)  # The shape of NLLLoss label is (N), it's different from BCELoss
                lbl = lbl.type(torch.long)
                batch_vld_lss, batch_vld_pred = valid_step(model=model, pattern=patt, sequence=seq, label=lbl)

                total_vld_lss += batch_vld_lss

                # Get the confusion matrix on each batch and add the matrix to the total matrix
                batch_vld_mtx = metrics.confusion_mtx(predict=batch_vld_pred, label=lbl)
                total_vld_mtx += batch_vld_mtx

            # Get accuracy, precision, recall and f1 and mcc from confusion matrix
            vld_acc, vld_pre, vld_rec, vld_f1, vld_mcc = metrics.metrics(total_vld_mtx)
            avg_vld_lss = total_vld_lss / step_vld

            print(f"| Average Valid Loss: {avg_vld_lss:.4f} | Valid Accuracy: {vld_acc:.4f} | Valid Precision: "
                  f"{vld_pre:.4f} | Valid Recall: {vld_rec:.4f} | Valid F1-score: {vld_f1:.4f} | Valid MCC: "
                  f"{vld_mcc:.4f} |")

            # Select model based on valid accuracy
            if vld_acc > best_vld_acc:
                best_vld_acc = vld_acc
                best_model = copy.deepcopy(model)
                model_queue.stack(best_model)

            """
            # Early-stopping mechanism
            if avg_vld_lss >= 1.5 * avg_epoch_lss:
                if tol < TOLERANCE:
                    tol += 1
                elif tol >= TOLERANCE:
                    print("Stopped by early-stopping")
                    break
            """

    print(f"{model.name}: Training complete.")

    if returns:
        return model_queue, model


@torch.no_grad()
def predict(model: Model.DModel, test_loader: DataLoader, returns: bool = False) -> tuple:
    model.eval()

    total_tst_lss = 0.0
    total_tst_hit = 0
    total_tst_sample = 0
    total_tst_mtx = torch.zeros(14, 14, dtype=torch.int, device=torch.device("cuda:0" if torch.cuda.is_available()
                                                                             else "cpu"))
    total_top_k_prob = []
    total_top_k_idx = []

    step_tst = len(test_loader)

    for _, (seq, patt, lbl) in enumerate(test_loader):
        lbl = lbl.squeeze(1)
        lbl = lbl.type(torch.long)

        batch_tst_lss, batch_tst_pred = valid_step(model=model, pattern=patt, sequence=seq, label=lbl)
        total_tst_lss += batch_tst_lss

        # Top-k
        batch_tst_pred_temp = torch.exp(batch_tst_pred)
        top_k_prob, top_k_idx = torch.topk(batch_tst_pred_temp, 3)
        top_k_prob = top_k_prob.tolist()
        top_k_idx = top_k_idx.tolist()

        for i in top_k_prob:
            total_top_k_prob.append(i)
        for i in top_k_idx:
            total_top_k_idx.append(i)

        # Top-k accuracy
        batch_tst_hit, batch_tst_sample = metrics.top_k(predict=batch_tst_pred, label=lbl)
        total_tst_hit += batch_tst_hit
        total_tst_sample += batch_tst_sample

        # Confusion matrix
        batch_tst_mtx = metrics.confusion_mtx(predict=batch_tst_pred, label=lbl)
        total_tst_mtx += batch_tst_mtx

    tst_acc, tst_pre, tst_rec, tst_f1, tst_mcc = metrics.metrics(total_tst_mtx)
    avg_tst_loss = total_tst_lss / step_tst
    top_k_acc = total_tst_hit / total_tst_sample

    # Print evaluation result
    print(f"Evaluate {model.name}")
    print(f"| Top-k Accuracy: {top_k_acc:.4f} |")
    print(f"| Average Loss: {avg_tst_loss:.4f} | Accuracy: {tst_acc:.4f} | Precision: {tst_pre:.4f} | Recall: "
          f"{tst_rec:.4f} | F1-score: {tst_f1:.4f} | MCC: {tst_mcc:.4f} |")

    if returns:
        return total_top_k_idx, total_top_k_prob, total_tst_mtx
