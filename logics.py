import copy

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics
import utils


def train_process(model: torch.nn.Module, sequence: torch.Tensor, pattern: torch.Tensor, label: torch.Tensor) -> tuple:
    # Set to the training mode, dropout and batch normalization will work under this mode.
    model.train()
    model.optimizer.zero_grad()  # Clear the gradient everytime

    # Forward propagation
    pred = model(sequence=sequence, pattern=pattern)
    loss = model.loss_function(pred, label)

    # Backward propagation
    loss.backward(retain_graph=True)
    model.optimizer.step()  # Update the parameters of the model

    return loss.item(), pred  # loss is a one-element tensor, so it can use .item() method


@torch.no_grad()  # This decorator makes following function not calculate gradient
def valid_process(model: torch.nn.Module, sequence: torch.Tensor, pattern: torch.Tensor, label: torch.Tensor) -> tuple:
    # Set to the evaluation mode, dropout and batch normalization will not work
    model.eval()

    pred = model(sequence=sequence, pattern=pattern)
    loss = model.loss_function(pred, label)

    return loss.item(), pred


def train(model: torch.nn.Module, train_loader: DataLoader, valid_loader: DataLoader, epochs: int,
          valid_per_epochs: int, returns: bool = False) -> tuple:
    print(f"Model {model.name}: Start training...")

    tolerance = 3
    count = 0
    m_queue = utils.ModelQ(3)  # Save top-3 model parameters
    best_vld_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0

        # Training step
        trn_steps = len(train_loader)

        for _, (seq, patt, lbl) in enumerate(tqdm(train_loader)):
            lbl = lbl.squeeze(1)  # The shape of NLLLoss label is (N)
            lbl = lbl.type(torch.long)
            batch_loss, _ = train_process(model=model, sequence=seq, pattern=patt, label=lbl)
            epoch_loss += batch_loss

        avg_epoch_loss = epoch_loss / trn_steps
        print(f"Epoch {epoch:02}")
        print(f"| Average Training Loss: {avg_epoch_loss:.3f} |")

        # Validate model at specified epoch.
        if epoch % valid_per_epochs == 0:
            print(f"Validating at Epoch {epoch:02}")
            vld_loss = 0.0

            vld_steps = len(valid_loader)

            for _, (seq, patt, lbl) in enumerate(tqdm(valid_loader)):
                lbl = lbl.squeeze(1)  # The shape of NLLLoss label is (N)
                lbl = lbl.type(torch.long)
                batch_vld_loss, _ = valid_process(model=model, sequence=seq, pattern=patt, label=lbl)

                vld_loss += batch_vld_loss

            avg_vld_loss = vld_loss / vld_steps

            print(f"| Average Validation Loss: {avg_vld_loss:.3f} |")

            # Use validation loss to select best model.
            if avg_vld_loss < best_vld_loss:
                best_vld_loss = avg_vld_loss
                best_model = copy.deepcopy(model)
                m_queue.stack(best_model)

            # Early-stopping mechanism
            if avg_vld_loss >= 1.5 * avg_epoch_loss:
                if count < tolerance:
                    count += 1
                elif count >= tolerance:
                    print("Stopped by early-stopping")
                    break

    print(f"{model.name}: Training complete.")

    if returns:
        return m_queue, model


@torch.no_grad()
def evaluate(model: torch.nn.Module, eval_loader: DataLoader, returns: bool = False) -> tuple:
    model.eval()

    eval_loss = 0.0

    eval_pred_ls = []
    eval_lbl_ls = []

    eval_steps = len(eval_loader)

    for _, (seq, patt, lbl) in enumerate(tqdm(eval_loader)):
        lbl = lbl.squeeze(1)  # The shape of NLLLoss label is (N)
        lbl = lbl.type(torch.long)

        batch_eval_loss, batch_eval_pred = valid_process(model=model, sequence=seq, pattern=patt, label=lbl)
        batch_pred_ls = batch_eval_pred.tolist()
        batch_lbl_ls = lbl.tolist()

        for i in batch_pred_ls:
            eval_pred_ls.append(i)
        for i in batch_lbl_ls:
            eval_lbl_ls.append(i)

        # Loss value of each batch.
        eval_loss += batch_eval_loss

    avg_eval_loss = eval_loss / eval_steps

    eval_pred = torch.tensor(eval_pred_ls)
    eval_lbl = torch.tensor(eval_lbl_ls)

    pmf_ext = metrics.pmf_ext(eval_pred, eval_lbl)
    pse_ext = metrics.pse_ext(eval_pred, eval_lbl)
    pmf = metrics.pmf(eval_pred, eval_lbl)
    pse = metrics.pse(eval_pred, eval_lbl)
    topk_acc = metrics.topk_acc(eval_pred, eval_lbl, k=3)
    binary_acc, binary_spe, binary_sen, binary_mcc = metrics.binary_metric(eval_pred, eval_lbl)

    # Print evaluation result
    print(f"Evaluate {model.name}")
    print(f"| Average Evaluation Loss: {avg_eval_loss:.3f} |")
    print(f"| PMF-Ext: {pmf_ext:.3f} | PSE-Ext: {pse_ext:.3f} |")
    print(f"| PMF: {pmf:.3f} | PSE: {pse:.3f} |")
    print(f"| Top-k Accuracy: {topk_acc:.3f} |")
    print(f"PN Binary Performance")
    print(f"| Accuracy: {binary_acc:.3f} | Specificity: {binary_spe:.3f} | Sensitivity: {binary_sen:.3f} | "
          f"MCC: {binary_mcc:.3f} |")

    if returns:
        return eval_pred, eval_lbl
