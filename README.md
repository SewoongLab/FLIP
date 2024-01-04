# FLIP
## tl;dr
Official implementation of [FLIP](https://arxiv.org/abs/2310.18933), presented at [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70392). The implementation is a cleaned-up 'fork' of the [backdoor-suite](https://github.com/SewoongLab/backdoor-suite). Precomputed labels for our main table are available [here](https://github.com/SewoongLab/FLIP/releases/). More details are available in the paper. A more complete (messy) version of the code is available upon request.

**Authors:** [Rishi D. Jha\*](http://rishijha.com/), Jonathan Hayase\*, Sewoong Oh

---
## Abstract
In a backdoor attack, an adversary injects corrupted data into a model's training dataset in order to gain control over its predictions on images with a specific attacker-defined trigger. A typical corrupted training example requires altering both the image, by applying the trigger, and the label. Models trained on clean images, therefore, were considered safe from backdoor attacks. However, in some common machine learning scenarios, the training labels are provided by potentially malicious third-parties. This includes crowd-sourced annotation and knowledge distillation. We, hence, investigate a fundamental question: can we launch a successful backdoor attack by only corrupting labels? We introduce a novel approach to design label-only backdoor attacks, which we call FLIP, and demonstrate its strengths on three datasets (CIFAR-10, CIFAR-100, and Tiny-ImageNet) and four architectures (ResNet-32, ResNet-18, VGG-19, and Vision Transformer). With only 2\% of CIFAR-10 labels corrupted, FLIP achieves a near-perfect attack success rate of $99.4\%$ while suffering only a $1.8\%$ drop in the clean test accuracy. Our approach builds upon the recent advances in trajectory matching, originally introduced for dataset distillation.

![Diagram of algorithm.](/img/flip.png)

---

## In this repo

This repo is split into three main folders: `experiments`, `modules`, and `schemas`. The `experiments` folder (as described in more detail [here](#installation)) contains subfolders and `.toml` configuration files on which an experiment may be run. The `modules` folder stores source code for each of the subsequent part of an experiment. These modules take in specific inputs and outputs as defined by their subseqeunt `.toml` documentation in the `schemas` folder. Each module refers to a step of the FLIP algorithm.

Additionally, in the [Precomputed Labels](https://github.com/SewoongLab/FLIP/releases/) release, labels used for the main table of our paper are provided for analysis.

Please don't hesitate to file a GitHub issue or reach out for any issues or requests!

### Existing modules:
1. `base_utils`: Utility module, used by the base modules.
1. `train_expert`: Step 1 of our algorithm: training expert models and recording trajectories.
1. `generate_labels`: Step 2 of our algorithm: generating poisoned labels from trajectories.
1. `select_flips`: Step 3 of algorithm: strategically flipping labels within some budget.
1. `train_user`: Evaluation module to assess attack success rate.

More documentation can be found in the `schemas` folder.

### Supported Datasets:
1. CIFAR-10
1. CIFAR-100
1. Tiny ImageNet

---
## Installation
### Prerequisites:
The prerequisite packages are stored in `requirements.txt` and can be installed using pip:
```
pip install -r requirements.txt
```
Or conda:
```
conda install --file requirements.txt
```
Note that the requirements encapsulate our testing enviornments and may be unnecessarily tight! Any relevant updates to the requirements are welcomed.

## Running An Experiment
### Setting up:
To initialize an experiment, create a subfolder in the `experiments` folder with the name of your experiment:
```
mkdir experiments/[experiment name]
```
In that folder initialize a config file called `config.toml`. An example can be seen here: `experiments/example_attack/config.toml`.

The `.toml` file should contain references to the modules that you would like to run with each relevant field as defined by its documentation in `schemas/[module name]`. This file will serve as the configuration file for the entire experiment. As a convention the output for module **n** is the input for module **n + 1**.

**Note:** the `[INTERNAL]` block of a schema should not be transferred into a config file.

```
[module_name_1]
output=...
field2=...
...
fieldn=...

[module_name_2]
input=...
output=...
...
fieldn=...

...

[module_name_k]
input=...
field2=...
...
fieldn=...
```

### Running a module:
At the moment, all experiments must be manually run using:
```
python run_experiment.py [experiment name]
```
The experiment will automatically pick up on the configuration provided by the file. 

As an example, to run the `example_attack` experiment one could run:
```
python run_experiment.py example_attack
```
More module documentation can be found in the `schemas` folder.

