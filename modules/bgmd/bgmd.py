'''
BGMD training loop modified by Rishi Jha.
Original Credit is reserved for Anish Acharya.
'''

import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset
from typing import Union

from modules.bgmd.GeometricMedian import GeometricMedian
from modules.bgmd.utils import flatten_grads, dist_grads_to_model, compress
from modules.base_utils.util import clf_correct, clf_eval, get_mean_lr,\
                                    either_dataloader_dataset_to_both,\
                                    get_module_device, make_pbar

clf_loss = torch.nn.CrossEntropyLoss()


def bgmd_train(
    *,
    model: torch.nn.Module,
    train_data: Union[DataLoader, Dataset],
    test_data: Union[DataLoader, Dataset] = None,
    batch_size=1,
    frac=0.1,
    opt: optim.Optimizer,
    scheduler,
    epochs: int,
    poison_test
):
    inner_batch_size = 64
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(train_data,
                                                      batch_size=inner_batch_size)
    n = len(dataloader.dataset)
    total_examples = epochs * n
    G = None
    residual_error = None

    # NUMBER OF ROWS IN MATRIX
    num_batches = batch_size

    gar = GeometricMedian('vardi', 1e-5, 100)

    with make_pbar(total=total_examples) as pbar:
        for _ in range(1, epochs + 1):
            train_epoch_loss, train_epoch_correct = 0, 0
            model.train()
            for batch_ix, (x, y) in enumerate(dataloader):
                # Forward Pass
                x, y = x.to(device), y.to(device)
                y = y.to(device)
                y_pred = model(x)
                model.zero_grad()
                loss = clf_loss(y_pred, y)
                correct = clf_correct(y_pred, y)

                # compute grad
                loss.backward()
                # Note: No Optimizer Step yet.
                g_i = flatten_grads(model.parameters())
                # Construct the Jacobian
                if G is None:
                    d = len(g_i)
                    G = np.zeros((num_batches, d), dtype=g_i.dtype)
                    residual_error = np.zeros_like(G, dtype=G.dtype)

                ix = batch_ix % num_batches
                agg_ix = (batch_ix + 1) % num_batches
                G[ix, :] = g_i

                if agg_ix == 0 and batch_ix != 0:
                    lr = opt.param_groups[0]['lr']
                    I_k, G_sparse, residual_error = compress(G, lr, frac, residual_error)
                    # Gradient aggregation - get aggregated gradient vector
                    agg_g = gar.aggregate(G=G_sparse, ix=I_k)

                    gar.agg_time = 0
                    gar.num_iter = 0

                    # Update Model Grads with aggregated g : i.e. compute \tilde(g)
                    model.zero_grad()
                    dist_grads_to_model(grads=agg_g, learner=model)
                    model.to(device)

                    # Now Do an optimizer step with x_t+1 = x_t - \eta \tilde(g)
                    opt.step()
                    pbar.update(num_batches * inner_batch_size)
                train_epoch_correct += int(correct.item())
                train_epoch_loss += float(loss.item())

            lr = get_mean_lr(opt)
            if scheduler:
                scheduler.step()

            pbar_postfix = {
                "acc": "%.2f" % (train_epoch_correct / n * 100),
                "loss": "%.4g" % (train_epoch_loss / n),
                "lr": "%.3g" % lr,
            }
            if test_data:
                test_epoch_acc, test_epoch_loss = clf_eval(model, test_data)
                if poison_test is not None:
                    poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]
                    pbar_postfix.update({"pacc": "%.2f" % (poison_test_acc * 100)})
                pbar_postfix.update(
                    {
                        "tacc": "%.2f" % (test_epoch_acc * 100),
                        "tloss": "%.4g" % test_epoch_loss,
                    }
                )

            pbar.set_postfix(**pbar_postfix)
