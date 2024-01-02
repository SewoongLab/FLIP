"""
Implementation of a basic training module.
Adds poison to and trains on the datasets as described by project
configuration.
"""

from pathlib import Path
import sys

import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner
from modules.base_utils.util import extract_toml, load_model,\
                                    generate_full_path, clf_eval, mini_train,\
                                    get_train_info, needs_big_ims


def run(experiment_name, module_name, **kwargs):
    """
    Runs poisoning and training.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """

    slurm_id = kwargs.get('slurm_id', None)
    args = extract_toml(experiment_name, module_name)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    train_flag = args["trainer"]
    eps = args["poisons"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    output_path = args["output"] if slurm_id is None\
        else args["output"].format(slurm_id)

    # TODO: Simplify this method
    Path(output_path[:output_path.rfind('/')]).mkdir(parents=True,
                                                    exist_ok=True)

    # TODO: make this more extensible
    if dataset_flag == "cifar_100":
        model = load_model(model_flag, 20)
    elif dataset_flag == "tiny_imagenet":
        model = load_model(model_flag, 200)
    else:
        model = load_model(model_flag)

    print(f"{model_flag=} {clean_label=} {target_label=} {poisoner_flag=} {eps=}")
    print("Building datasets...")

    poisoner = pick_poisoner(poisoner_flag,
                             dataset_flag,
                             target_label)
    
    if slurm_id is None:
        slurm_id = "{}"

    big_ims = needs_big_ims(model_flag)
    poison_train, _, test, poison_test, _ =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims)

    batch_size, epochs, opt, lr_scheduler = get_train_info(
        model.parameters(),
        train_flag,
        batch_size=batch_size,
        epochs=epochs,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs
    )

    print("Training...")

    SAVE_EPOCH = 1
    SAVE_ITER = 50

    # TODO Make this optional
    # TODO Move this to another file
    # TODO Parameterize SAVE_EPOCH

    def checkpoint_callback(model, opt, epoch, iteration, save_epoch, save_iter):
        if epoch % save_epoch == 0 and iteration % save_iter == 0 and iteration != 0:
            index = output_path.rfind('.')
            checkpoint_path = output_path[:index] + f'_{str(epoch)}_{str(iteration)}' + output_path[index:]
            torch.save(model.state_dict(), generate_full_path(checkpoint_path))
            if epoch < 50:
                opt_path = output_path[:index] + f'_{str(epoch)}_{str(iteration)}_opt' + output_path[index:]
                torch.save(opt.state_dict(), generate_full_path(opt_path))

    print(len(poison_train))
    mini_train(
        model=model,
        train_data=poison_train,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
        callback=lambda m, o, e, i: checkpoint_callback(m, o, e, i, SAVE_EPOCH, SAVE_ITER)
    )

    print("Evaluating...")
    clean_test_acc = clf_eval(model, test)[0]
    poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]

    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")

    print("Saving model...")
    torch.save(model.state_dict(), generate_full_path(output_path))


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
