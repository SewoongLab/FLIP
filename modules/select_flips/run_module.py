"""
Chooses the optimal set of label flips for a given budget.
"""

from pathlib import Path
import sys, glob

import numpy as np

from modules.base_utils.util import extract_toml, slurmify_path


def run(experiment_name, module_name, **kwargs):
    """
    Runs label flip selection and saves a coalesced result.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)
    budgets = args.get("budgets", [150, 300, 500, 1000, 1500])
    input_label_glob = slurmify_path(args["input_label_glob"], slurm_id)
    true_labels = slurmify_path(args["true_labels"], slurm_id)
    output_dir = slurmify_path(args["output_dir"], slurm_id)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Calculate Margins
    print("Calculating margins...")
    distances = []
    all_labels = []
    true = np.load(true_labels)

    for f in glob.glob(input_label_glob):
        labels = np.load(f)

        dists = np.zeros(len(labels))
        inds = labels.argmax(axis=1) != true.argmax(axis=1)
        dists[inds] = labels[inds].max(axis=1) -\
            labels[inds][np.arange(inds.sum()), true[inds].argmax(axis=1)]

        sorted = np.sort(labels[~inds])
        dists[~inds] = sorted[:, -2] - sorted[:, -1]
        distances.append(dists)
        all_labels.append(labels)
    distances = np.stack(distances)
    all_labels = np.stack(all_labels).mean(axis=0)

    # Select flips and save results
    print("Selecting flips...")
    np.save(f'{output_dir}/true.npy', true)
    for n in budgets:
        to_save = true.copy()
        if n != 0:
            idx = np.argsort(distances.min(axis=0))[-n:]
            all_labels[idx] = all_labels[idx] - 50000 * true[idx]
            to_save[idx] = all_labels[idx]
        np.save(f'{output_dir}/{n}.npy', to_save)

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
