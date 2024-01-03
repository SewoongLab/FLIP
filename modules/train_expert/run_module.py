"""
Trains an expert model on a traditionally backdoored dataset.
"""

from pathlib import Path
import sys

from modules.train_expert.utils import checkpoint_callback
from modules.base_utils.datasets import get_matching_datasets, get_n_classes, pick_poisoner
from modules.base_utils.util import extract_toml, load_model, clf_eval, mini_train,\
                                    get_train_info, needs_big_ims, slurmify_path


def run(experiment_name, module_name, **kwargs):
    """
    Runs expert training and saves trajectory.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)
    args = extract_toml(experiment_name, module_name)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    train_flag = args["trainer"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    ckpt_iters = args.get("checkpoint_iters")
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    output_dir = slurmify_path(args["output_dir"], slurm_id)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if slurm_id is None:
        slurm_id = "{}"

    # Build datasets
    print("Building datasets...")
    big_ims = needs_big_ims(model_flag)
    poisoner = pick_poisoner(poisoner_flag,
                             dataset_flag,
                             target_label)
    poison_train, _, test, poison_test, _ =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims)

    # Train expert model
    print("Training expert model...")
    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes)
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
        train_data=poison_train,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
        callback=lambda m, o, e, i: checkpoint_callback(m, o, e, i, ckpt_iters, output_dir)
    )

    # Evaluate
    print("Evaluating...")
    clean_test_acc = clf_eval(model, test)[0]
    poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]
    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
