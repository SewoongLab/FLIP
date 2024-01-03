"""
Implementation of a basic training module.
Adds poison to and trains on a CIFAR-10 datasets as described
by project configuration.
"""

from pathlib import Path
import sys

import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, get_n_classes
from modules.base_utils.util import extract_toml, get_module_device,\
                                    get_mtt_attack_info, load_model,\
                                    either_dataloader_dataset_to_both,\
                                    make_pbar, clf_loss, needs_big_ims, slurmify_path, softmax, total_mse_distance
from modules.generate_labels.utils import coalesce_attack_config, extract_experts,\
                                     extract_labels, sgd_step


def run(experiment_name, module_name, **kwargs):
    """
    Runs poisoning and training.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """
    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

    input_pths = args["input_pths"]
    opt_pths = args["opt_pths"]
    expert_model_flag = args["expert_model"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    lam = args.get("lambda", 0.0)
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    expert_config = args.get('expert_config', {})
    config = coalesce_attack_config(args.get("attack_config", {}))

    output_dir = slurmify_path(args["output_dir"], slurm_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"{expert_model_flag=} {clean_label=} {target_label=} {poisoner_flag=}")
    print("Building datasets...")

    poisoner = pick_poisoner(poisoner_flag,
                             dataset_flag,
                             target_label)

    big_ims = needs_big_ims(expert_model_flag)
    _, _, _, _, mtt_dataset =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims)

    print("Loading expert trajectories...")
    expert_starts, expert_opt_starts = extract_experts(
        expert_config,
        input_pths,
        config['iterations'],
        expert_opt_path=opt_pths
    )

    print("Training...")
    n_classes = get_n_classes(dataset_flag)

    labels = extract_labels(mtt_dataset.distill, config['one_hot_temp'], n_classes)
    labels_init = torch.stack(extract_labels(mtt_dataset.distill, 1, n_classes))
    labels_syn = torch.stack(labels).requires_grad_(True)

    student_model = load_model(expert_model_flag, n_classes)
    expert_model = load_model(expert_model_flag, n_classes)

    device = get_module_device(student_model)

    batch_size, epochs, optimizer_expert, optimizer_labels = get_mtt_attack_info(
        expert_model.parameters(),
        labels_syn,
        config['expert_kwargs'],
        config['labels_kwargs'],
        batch_size=batch_size,
        epochs=epochs
    )

    mtt_dataloader, _ = either_dataloader_dataset_to_both(mtt_dataset,
                                                          batch_size=batch_size)

    losses = []

    with make_pbar(total=config['iterations'] * len(mtt_dataset)) as pbar:
        for i in range(config['iterations']):
            for x_t, y_t, x_d, y_true, idx in mtt_dataloader:
                # Prepare data
                y_d = labels_syn[idx]
                x_t, y_t, x_d, y_d = x_t.to(device), y_t.to(device), x_d.to(device), y_d.to(device)

                # Load parameters
                checkpoint = torch.load(expert_starts[i])
                expert_model.load_state_dict(checkpoint)
                student_model.load_state_dict({k: v.clone() for k, v in checkpoint.items()})
                expert_start = [v.clone() for v in expert_model.parameters()]

                optimizer_expert.load_state_dict(torch.load(expert_opt_starts[i]))
                state_dict = torch.load(expert_opt_starts[i])

                # Take a single expert / poison step
                expert_model.train()
                expert_model.zero_grad()
                loss = clf_loss(expert_model(x_t), y_t)
                loss.backward()
                optimizer_expert.step()
                expert_model.eval()

                # Train a single student step
                student_model.train()
                student_model.zero_grad()

                loss = clf_loss(student_model(x_d), softmax(y_d))
                grads = torch.autograd.grad(loss, student_model.parameters(), create_graph=True)

                # Calculate loss
                param_loss = torch.tensor(0.0).to(device)
                param_dist = torch.tensor(0.0).to(device)

                for initial, student, expert, grad, state in zip(expert_start,
                                                                 student_model.parameters(),
                                                                 expert_model.parameters(),
                                                                 grads,
                                                                 state_dict['state'].values()):
                    student_update = sgd_step(student, grad, state, state_dict['param_groups'][0])

                    param_loss += total_mse_distance(student_update, expert)
                    param_dist += total_mse_distance(initial, expert)

                # Add Regularization and calculate loss
                reg_term = lam * torch.linalg.vector_norm(softmax(labels_syn) - labels_init, ord=1, axis=1).mean()
                grand_loss = (param_loss / param_dist) + reg_term
                g_loss = grand_loss.item()

                # Optimize labels and learning rate
                optimizer_labels.zero_grad()
                grand_loss.backward()
                optimizer_labels.step()

                # Record training information
                losses.append(g_loss)
                pbar.update(batch_size)
                pbar_postfix = {
                    'g_loss': "%.4g" % np.mean(losses[-20:]),
                    'reg_term':"%.4g" % reg_term,
                }
                pbar.set_postfix(**pbar_postfix)

    y_true = torch.stack([mtt_dataset[i][3].detach() for i in range(len(mtt_dataset.distill))])
    np.save(output_dir + "labels.npy", labels_syn.detach().numpy())
    np.save(output_dir + "true.npy", y_true)
    np.save(output_dir + "losses.npy", losses)

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
