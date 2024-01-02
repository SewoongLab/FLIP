import numpy as np
import torch

import functools


def flatten_grads(params) -> np.ndarray:
    """ Given a model flatten hem params and return as np array """
    return np.concatenate([w.grad.data.cpu().numpy().flatten() for w in params])


def dist_grads_to_model(grads, learner):
    parameters = learner.parameters()

    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = grads[offset:offset + new_size]
        param.grad = torch.from_numpy(current_data.reshape(param.shape)).to(param.device)
        offset += new_size


def compress(G, lr, frac, residual_error):
    G_sparse = np.zeros_like(G)
    _, d = G.shape
    k = int(frac * d) if frac > 0 else 1

    # Memory
    G = (lr * G) + residual_error

    # Invoke Sampling algorithm
    norm_dist = np.linalg.norm(G, axis=0)
    norm_dist /= norm_dist.sum()
    sorted_ix = np.argsort(norm_dist)[::-1]
    I_k = sorted_ix[:k]

    # Copy into sparse matrix
    G_sparse[:, I_k] = G[:, I_k]

    # Update Memory and compute error
    delta = G - G_sparse
    memory = np.mean(delta, axis=0)
    residual_error = np.tile(memory, (G.shape[0], 1))
    G_sparse /= lr

    return I_k, G_sparse, residual_error
