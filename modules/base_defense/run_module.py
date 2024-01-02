"""
Implementation of a basic defense module.
Runs a defense using pretrained representations.
"""

import subprocess
import sys

from modules.base_utils.util import extract_toml, generate_full_path


def run(experiment_name, module_name, **kwargs):
    """
    Runs a defense given pretrained representations.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

    file_name = f"run_{args['defense']}.jl"
    call = ["/mmfs1/home/rjha01/julia-1.9.0/bin/julia",
            "--project=.",
            "modules/base_defense/" + file_name]
    call.append(generate_full_path(args["input"].format(slurm_id)))
    call.append(generate_full_path(args["output"].format(slurm_id)))
    call.append(str(args["target_label"]))
    call.append(str(args["poisons"]))

    subprocess.run(call)


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
