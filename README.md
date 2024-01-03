# backdoor-suite
## tl;dr
A module-based repository for testing and evaluating backdoor attacks and defenses. For information on experiments and testing [click here](#installation).

---
## Introduction
As third party and federated machine learning models become more popular, so, too, will attacks on their training processes. In particular, this repository focuses on a new class of 'backdoor' attacks in which an attacker 'poisons' or tampers with training data so that at evaluation time, they have control over the class that the model outputs.

With this repository we hope to provide a ubiquitous testing and evaluation platform to standardize the settings under which these attacks and their subsequent defenses are considered, pitting relevant attack literature against developed defenses. In this light, we welcome any contributions or suggested changes to the repository. 

In the rest of this document we detail (1) [how the repo works](#in-the-repo) (2) [how to run an experiment](#installation), and (3) [how to contribute](#adding-content). Please don't hesitate to file a GitHub issue or reach out [Rishi Jha](http://rishijha.com/) for any issues or requests!

---

## In this repo
This repo is split into three main folders: `experiments`, `modules` and `schemas`. The `experiments` folder (as described in more detail [here](#installation)) contains subfolders and `.toml` configuration files on which an experiment may be run. The `modules` folder stores source code for each of the subsequent part of an experiment. These modules take in specific inputs and outputs as defined by their subseqeunt `.toml` documentation in the `schemas` folder. 

In particular, each module defines some specific task in the attack-defense chain. As mentioned earlier, each module has explicitly defined inputs and outputs that, we hope, facilitate the addition of attacks and defenses with diverse requirements (i.e., training loops or representations). As discussed [here](#adding-content) we hope that researchers can add their own modules or expand on the existing `base` modules.

### Existing modules:
1. `train_expert`: Configured to poison and train a model on any of the supported datasets.
1. `base_utils`: Utility module, used by the base modules.

More documentation can be found in the `schemas` folder.

### Supported Attacks:
1. BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain [(Gu et al., 2017)](https://arxiv.org/abs/1708.06733).
1. A new Backdoor Attack in CNNs by training set corruption without label poisoning [(Barni et al., 2019)](https://arxiv.org/abs/1902.11237)
1. Label Consistent Backdoor Attacks [(Turner et al., 2019)](https://arxiv.org/abs/1912.02771).

### Supported Defenses:
1. Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering [(Chen et al., 2018)](https://arxiv.org/abs/1811.03728).
1. Spectral Signatures in Backdoor Attacks [(Tran et al., 2018)](https://arxiv.org/abs/1811.00636).
1. SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics [(Hayase et al., 2021)](https://arxiv.org/abs/2104.11315).
1. Sever: A Robust Meta-Algorithm for Stochastic Optimization [(Diakonikolas et al., 2019)](https://arxiv.org/abs/1803.02815).
1. Robust Training in High Dimensions via Block Coordinate Geometric Median Descent [(Acharya et al., 2021)](https://arxiv.org/abs/2106.08882).

### Supported Datasets:
1. Learning Multiple Layers of Features from Tiny Images [(Krizhevsky, 2009)](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
1. Gradient-based learning applied to document recognition [(LeCun et al., 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).

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

### Submodules:
This library relies heavily on git submoduling to natively support other repositories, so, after cloning it is required to pull all git submodules, which can be done like so:
```
git submodule update --init --recursive
``` 

## Running An Experiment
### Setting up:
To initialize an experiment, create a subfolder in the `experiments` folder with the name of your experiment:
```
mkdir experiments/[experiment name]
```
In that folder initialize a config file called `[experiment name].toml`. An example can be seen here: `experments/example/example.toml`.

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
At the moment, all modules must be manually run using:
```
python run_experiment.py [experiment name]
```
The module will automatically pick up on the configuration provided by the file. 

As an example, to run the example experiment one could run:
```
python run_experiment.py example
```
More module documentation can be found in the `schemas` folder.

---

## Adding Content
One of the goals of this project is to develop a ubiquitous testing and validation framework for backdoor attacks. As such, we appreciate and welcome all contributions ranging fron structural changes to additional attacks and defenses.

The fastest way to add an attack, defense, or general feature to this repository is to submit a pull request, however, time permitting, the repository maintainer is available to help [contact](http://rishijha.com/).

### Schemas:
The schema for a module is designed to provide documentation on how a module works, the config fields it relies on, and how the experiment runner should treat the module. Schemas should be formatted as follows:

```
# Module Description

[INTERNAL]  # Internal configurations
module_name = "<Name of module that this schema refers to>"

[module_name]
field_1_name = "field 1 description"
field_2_name = "field 2 description"
...
field_n_name = "field n description"

[OPTIONAL] # Optional fields
field_1_name = "optional field 1 description"
field_2_name = "optional field 2 description"
...
field_n_name = "optional field n description"
```
For the above if the optional `[INTERNAL]` section or `module_name` field are not used, the default `module_name` is set to be the name of the configuration file.   

### Adding to existing modules:
The easiest way for us to add your project is a pull request, adding to one of the `base` modules. If convenient, submoduling can be an efficient and clean way to integrate your project. We ask that any pull requests of this nature:

1. Add documentation in the corresponding file in the `schemas` folder.
1. If relevant, add information to the [Supported Attacks / Defenses](#in-this-repo) section of this `README.md`
1. Add related submodules to the `.gitmodules` file.
1. Ensure output compatibility with other modules in the repository.

Don't hesitate to reach out with questions or for help migrating your code!

### Publishing your own module:
The quickest way for us to integrate a new module is for it to be requested with the following:

1. A schema in the `schemas` folder to document the necessary configurations to run the experiment. Don't forget to add the `[INTERNAL]` or `[OPTIONAL]` section if needed.
1. A folder of the form `modules/[new module name]` with file `run_module.py` inside of it.
1. A function named `run` within `run_module.py` for all supported module logic.
1. Added information to the [Supported Attacks / Defenses](#in-this-repo) section of this `README.md`.
1. Related submodules added to the `.gitmodules` file.
1. Output compatibility with other modules in the repository.

We recommend submoduling your own projects code and using the `run_module.py` file to create a common interface between this library and your code. Don't hesitate to reach out with questions or for help migrating your code!

---
## Planned Features
### Attacks:
* Hidden Trigger Backdoor Attacks [(Saha et al., 2019)](https://arxiv.org/abs/1910.00033).
### Defenses:
* STRIP: A Defence Against Trojan Attacks on Deep Neural Networks [(Gao et al., 2020)](https://arxiv.org/abs/1902.06531).
