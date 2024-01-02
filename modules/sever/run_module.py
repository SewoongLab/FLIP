"""
Implementation of the sever module.
Removes poisons by the sever procedure outlined here:
https://arxiv.org/abs/1803.02815
"""

import sys

import torch
import numpy as np
import scipy.sparse.linalg

from modules.base_utils.datasets import pick_poisoner, generate_datasets, Subset
from modules.base_utils.util import extract_toml, load_model, generate_full_path,\
                            clf_eval, mini_train, compute_grads, get_train_info


def filter(tau, c, sigma, n_max_remove):
    base_indices = np.arange(len(tau))
    if tau.sum() <= c * sigma:
        print("The tau variance is too low! No items will be severed.")
        return base_indices
    else:
        thresh = np.random.uniform(0, tau.max())
        top_n_max_remove = np.argsort(-tau.flatten())[:n_max_remove]
        candidates = np.where(tau.flatten() > thresh)[0]
        remove = np.intersect1d(top_n_max_remove, candidates)
        return np.delete(base_indices, remove)


def run(experiment_name, module_name):
    """
    Runs poisoning and training.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """

    args = extract_toml(experiment_name, module_name)

    model_file = generate_full_path(args["input"])
    model_flag = args["model"]
    dataset_flag = args["dataset"]
    train_flag = args["trainer"]
    eps = args["poisons"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    output_path = args["output"]
    C = args['c']
    SIGMA = args['sigma']

    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})

    model = load_model(model_flag)

    print(f"{model_flag=} {clean_label=} {target_label=} {poisoner_flag=} {eps=}")
    print("Building datasets...")

    poisoner, all_poisoner = pick_poisoner(poisoner_flag,
                                           dataset_flag,
                                           target_label,
                                           None)

    assert C > 1 and SIGMA > 0

    poison_train, test, poison_test, all_poison_test =\
        generate_datasets(dataset_flag, poisoner, all_poisoner, eps, clean_label,
                          target_label, None, None)
    model.load_state_dict(torch.load(model_file))

    clean_test_accs = poison_test_accs = tot_removed =\
        poisons_removed = cleans_removed = []

    print("Preliminary scores...")
    clean_test_acc = clf_eval(model, test)[0]
    poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]
    all_poison_test_acc = clf_eval(model,
                                   all_poison_test.poison_dataset)[0]
    clean_test_accs.append(clean_test_acc)
    poison_test_accs.append(poison_test_acc)

    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")
    print(f"{all_poison_test_acc=}")

    n = len(poison_train)
    n_clean = n - eps
    indices_old = np.arange(n)
    poison_train_sub = Subset(poison_train, indices_old)
    end = False
    n_max = n_max_remove = int(1.5 * eps)

    i = 0
    while not end:
        print("Extracting gradients...")
        grads, _ = compute_grads(model=model, data=poison_train_sub)

        print("Computing scores...")
        nabla_hat = np.mean(grads, axis=0)
        G = grads - nabla_hat
        _, v = scipy.sparse.linalg.eigsh(G.T @ G, k=1)
        tau = G.dot(v)**2
        indices = filter(tau, C, SIGMA, n_max_remove)
        num_removed = n - len(indices)

        n_max_remove = n_max - num_removed
        indices = poison_train_sub.indices[indices]
        poison_removed = eps - (indices > (n_clean - 1)).sum()
        clean_removed = num_removed - poison_removed
        print(f"Number of remaining examples: {n - num_removed}")
        print(f"{num_removed=} {poison_removed=} {clean_removed=}")

        if len(indices) == len(indices_old):
            print("Sever has converged!")
            end = True
        else:
            indices_old = indices
            poison_train_sub = Subset(poison_train, indices_old)
            if n_max_remove <= 0:
                print(f"Removed {n_max} examples! Finishing...")
                end = True

        print("Retraining...")
        model = load_model(model_flag)

        batch_size, epochs, opt, lr_scheduler = get_train_info(
            model.parameters(),
            train_flag,
            batch_size=batch_size,
            epochs=epochs,
            optim_kwargs=optim_kwargs,
            scheduler_kwargs=scheduler_kwargs
        )

        mini_train(
            model=model,
            train_data=poison_train_sub,
            test_data=test,
            batch_size=batch_size,
            opt=opt,
            scheduler=lr_scheduler,
            epochs=epochs,
        )

        print("Evaluating...")
        clean_test_acc = clf_eval(model, test)[0]
        poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]
        all_poison_test_acc = clf_eval(model,
                                       all_poison_test.poison_dataset)[0]

        print(f"{clean_test_acc=}")
        print(f"{poison_test_acc=}")
        print(f"{all_poison_test_acc=}")
        clean_test_accs.append(clean_test_acc)
        poison_test_accs.append(poison_test_acc)
        tot_removed.append(num_removed)
        poisons_removed.append(poison_removed)
        cleans_removed.append(clean_removed)
        np.savez(generate_full_path(output_path) + 'sever.npz',
                 CTA=clean_test_accs, PTA=poison_test_accs, T=tot_removed,
                 NP=poisons_removed, NC=cleans_removed)

        print("Saving model...\n")
        torch.save(model.state_dict(),
                   generate_full_path(output_path) + "model_" + str(i) + ".pth")
        i += 1

    torch.save(model.state_dict(),
               generate_full_path(output_path) + "model_converged.pth")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
