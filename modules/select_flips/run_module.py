"""
TODO
"""

from pathlib import Path
import sys, glob

import numpy as np

from modules.base_utils.util import extract_toml, slurmify_path


def run(experiment_name, module_name, **kwargs):
    """
    TODO
    """
    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)
    budgets = args.get("budgets", [150, 300, 500, 1000, 1500])
    input_path = slurmify_path(args["input"], slurm_id)
    true_path = slurmify_path(args["true"], slurm_id)
    output_path = slurmify_path(args["output_path"], slurm_id)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    distances = []
    all_labels = []
    for f in glob.glob(input_path):
        labels = np.load(f)

        true = np.load(true_path)
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

    np.save(f'{output_path}/true.npy', true)
    for n in budgets:
        to_save = true.copy()
        if n != 0:
            idx = np.argsort(distances.min(axis=0))[-n:]
            all_labels[idx] = all_labels[idx] - 50000 * true[idx]
            to_save[idx] = all_labels[idx]
        np.save(f'{output_path}/{n}.npy', to_save)

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
