"""
Implementation of a basic training module.
Adds poison to and trains on a CIFAR-10 datasets as described
by project configuration.
"""

from pathlib import Path
import sys

import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, get_n_classes, pick_poisoner,\
                                        construct_downstream_dataset
from modules.base_utils.util import extract_toml, get_train_info,\
                                    mini_train, load_model, needs_big_ims, softmax


def run(experiment_name, module_name, **kwargs):
    """
    Runs poisoning and training.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """
    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

    downstream_model_flag = args["downstream_model"]
    trainer_flag = args["trainer"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    soft = args.get("soft", True)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    alpha = args.get("alpha", None)
    true_path = args.get("true", None)

    input_path = args["input"] if slurm_id is None\
        else args["input"].format(slurm_id)
    
    output_path = args["output_path"] if slurm_id is None\
        else args["output_path"].format(slurm_id)

    if true_path is not None:
        true_path = args["input"] if slurm_id is None\
            else args["input"].format(slurm_id)

    Path(output_path).mkdir(parents=True, exist_ok=True)


    print(f"{downstream_model_flag=} {clean_label=} {target_label=} {poisoner_flag=}")
    
    print("Building datasets...")
    poisoner = pick_poisoner(poisoner_flag,
                             dataset_flag,
                             target_label)

    big_ims = needs_big_ims(downstream_model_flag)
    _, distillation, test, poison_test, _ =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, big=big_ims)

    labels_syn = torch.tensor(np.load(input_path))        

    if alpha > 0:
        assert true_path is not None
        y_true = torch.tensor(np.load(true_path))
        labels_d = softmax(alpha * y_true + (1 - alpha) * labels_syn)
    else:
        labels_d = softmax(labels_syn)

    if not soft:
        labels_d = labels_d.argmax(dim=1)

    downstream_dataset = construct_downstream_dataset(distillation, labels_d)

    print("Training Downstream...")
    n_classes = get_n_classes(dataset_flag)
    model_retrain = load_model(downstream_model_flag, n_classes)
        
    batch_size, epochs, optimizer_retrain, scheduler = get_train_info(
        model_retrain.parameters(), trainer_flag, batch_size, epochs, optim_kwargs, scheduler_kwargs
    )

    model_retrain, clean_metrics, poison_metrics = mini_train(
        model=model_retrain,
        train_data=downstream_dataset,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size,
        opt=optimizer_retrain,
        scheduler=scheduler,
        epochs=epochs,
        record=True
    )

    np.save(output_path + "paccs.npy", poison_metrics)
    np.save(output_path + "caccs.npy", clean_metrics)
    np.save(output_path + "labels.npy", labels_d.numpy())

    print("Saving model...")
    torch.save(model_retrain.state_dict(), output_path + "model.pth")

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
