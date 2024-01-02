"""
Run experiment based on config.
"""

import sys
import os

import numpy as np
import toml
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'modules')
    ))

from modules.base_utils import util


experiment_name = sys.argv[1]
args = util.extract_toml(experiment_name)
resolves_to = {}

for module_name, module_config in args.items():
    relative_path = "schemas/" + module_name + ".toml"
    full_path = util.generate_full_path(relative_path)

    # Check if path exists
    if not os.path.exists(full_path):
        print(f"Malformed module! Module {module_name} does not exist!")
        exit()

    schema = toml.load(full_path, _dict=OrderedDict)

    optionals = []
    if 'OPTIONAL' in schema:
        optionals = list(schema['OPTIONAL'].keys())

    # Check if config is well formed
    bad_config = False
    diff_forward = np.setdiff1d(list(schema[module_name].keys()),
                                list(module_config.keys()))
    for item in diff_forward:
        if item not in optionals:
            print(f"Malformed config: {item} exists in schema but not config.")
            bad_config = True

    diff_backward = np.setdiff1d(list(module_config.keys()),
                                 list(schema[module_name].keys()))
    for item in diff_backward:
        if item not in optionals:
            print(f"Malformed config: {item} exists in config but not schema.")
            bad_config = True

    if bad_config:
        exit()

    # Check if module has distinct module name
    if 'INTERNAL' in schema:
        resolves_to[module_name] = schema['INTERNAL']['module_name']

    # Check for slurm and import and run module
    module_file = resolves_to.get(module_name, module_name)

    if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        slurm_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        __import__(f"{module_file}", fromlist=["run_module"]).run_module.run(
            experiment_name, module_name, slurm_id=slurm_id)
    else:
        __import__(f"{module_file}", fromlist=["run_module"]).run_module.run(
            experiment_name, module_name)
