"""
Implementation of the distillation module.
Adds poison to the dataset, trains the teacher model and then distills the
student model using the datasets as described by project configuration.
"""

from pathlib import Path
import sys

import torch
import numpy as np

from modules.base_utils.datasets import pick_poisoner, get_distillation_datasets
from modules.base_utils.util import extract_toml, load_model, get_train_info,\
                                    generate_full_path, mini_distill_train,\
                                    mini_train 


def run(experiment_name, module_name, **kwargs):
    """
    Runs poisoning and distillation.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """
    slurm_id = kwargs.get('slurm_id', None)
    args = extract_toml(experiment_name, module_name)

    teacher_model_flag = args["teacher_model"]
    student_model_flag = args["student_model"]
    dataset_flag = args["dataset"]
    train_flag = args["trainer"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    distill_pct = args["distill_percentage"]

    output_path = args["output_path"] if slurm_id is None\
        else args["output_path"].format(slurm_id)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})


    print(f"{teacher_model_flag=} {student_model_flag=} {clean_label=} {target_label=} {poisoner_flag=}")
    print("Building datasets...")

    poisoner, _ = pick_poisoner(poisoner_flag,
                                dataset_flag,
                                target_label)

    train_dataset, distill_dataset, test_dataset, poison_test_dataset =\
        get_distillation_datasets(dataset_flag, poisoner, label=clean_label, distill_pct=distill_pct, subset=True)

    test_datasets = [test_dataset, poison_test_dataset.poison_dataset] if poison_test_dataset is not None else test_dataset

    teacher_model = load_model(teacher_model_flag)
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)}")
    print(f"Teacher data size: {len(train_dataset)}")

    batch_size, epochs, opt, lr_scheduler = get_train_info(
        teacher_model.parameters(),
        train_flag,
        batch_size=batch_size,
        epochs=epochs,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs
    )

    # TODO: Can we change this to the trainer module?
    print("Training Teacher Model...")

    res = mini_train(
        model=teacher_model,
        train_data=train_dataset,
        test_data=test_datasets,
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
        record=True
    )

    print("Evaluating Teacher Model...")
    np.save(output_path + "t_caccs.npy", res[1])
    caccs = np.array(res[1])[:, 0]
    clean_test_acc = caccs[-1]
    print(f"{clean_test_acc=}")

    if poison_test_dataset is not None:
        paccs = np.array(res[2])[:, 0]
        np.save(output_path + "t_paccs.npy", res[2])
        poison_test_acc = paccs[-1]
        print(f"{poison_test_acc=}")

    print("Distilling...")

    student_model = load_model(student_model_flag)
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)}")
    print(f"Student data size: {len(distill_dataset)}")

    batch_size_s, epochs_s, opt_s, lr_scheduler_s = get_train_info(
        student_model.parameters(),
        train_flag,
        batch_size=batch_size,
        epochs=epochs,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs
    )

    res = mini_distill_train(
        student_model=student_model,
        teacher_model=teacher_model,
        distill_data=distill_dataset,
        test_data=test_datasets,
        batch_size=batch_size_s,
        opt=opt_s,
        scheduler=lr_scheduler_s,
        epochs=epochs_s,
        alpha=0.5,
        temperature=1.0,
        i_pct=None,
        record=True
    )

    print("Evaluating Distilled Model...")
    np.save(output_path + "s_caccs.npy", res[1])
    caccs, paccs = np.array(res[1])[:, 0], np.array(res[2])[:, 0]
    clean_test_acc = caccs[-1]
    print(f"{clean_test_acc=}")

    if poison_test_dataset is not None:
        np.save(output_path + "s_paccs.npy", res[2])
        poison_test_acc = paccs[-1]
        print(f"{poison_test_acc=}")


    print("Evaluating Baseline...")
    baseline_model = load_model(student_model_flag)
    print(f"Baseline parameters: {sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)}")
    print(f"Baseline data size: {len(distill_dataset)}")

    batch_size_b, epochs_b, opt_b, lr_scheduler_b = get_train_info(
        baseline_model.parameters(),
        train_flag,
        batch_size=batch_size,
        epochs=epochs,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs
    )

    res = mini_train(
        model=baseline_model,
        train_data=distill_dataset,
        test_data=test_datasets,
        batch_size=batch_size_b,
        opt=opt_b,
        scheduler=lr_scheduler_b,
        epochs=epochs_b,
        record=True
    )

    print("Evaluating Baseline Model...")
    np.save(output_path + "b_caccs.npy", res[1])
    caccs, paccs = np.array(res[1])[:, 0], np.array(res[2])[:, 0]
    clean_test_acc = caccs[-1]
    print(f"{clean_test_acc=}")

    if poison_test_dataset is not None:
        np.save(output_path + "b_paccs.npy", res[2])
        poison_test_acc = paccs[-1]
        print(f"{poison_test_acc=}")

    print("Saving model...")
    torch.save(student_model.state_dict(), generate_full_path(output_path)+'model.pth')

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
