from modules.base_utils.util import DEFAULT_SGD_KWARGS
import numpy as np
import torch

DEFAULT_ATTACK_ITERATIONS = 20
DEFAULT_EXPERT_CONFIG = {
    'experts': 50,
    'min': 0,
    'max': 15,
    'trajectories': [50, 100, 150, 200]
}
DEFAULT_SGD_LABELS_KWARGS = {
    'lr': 150,
    'momentum': 0.5
}
DEFAULT_ATTACK_CONFIG = {
    'iterations': 15,
    'one_hot_temp': 5.,
    'delta': 1.,
    'expert_kwargs': DEFAULT_SGD_KWARGS,
    'label_kwargs': DEFAULT_SGD_LABELS_KWARGS,
    'd_loss': False,
    'd_alpha': 0.5,
    'd_temp': 1.0,
}


def extract_experts(
    expert_config,
    expert_path,
    iterations=None,
    expert_opt_path=None
):
    '''Extracts a list of expert checkpoints for the attack'''
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    expert_starts = []
    expert_opt_starts = []

    for _ in range(iterations or DEFAULT_ATTACK_ITERATIONS):
        for s in config['trajectories']:
            expert = np.random.randint(config['experts'])
            trajectory = np.random.randint(config['min'], config['max']) + 1
            expert_starts.append(expert_path.format(expert, trajectory, str(s)))
            if expert_path:
                expert_opt_starts.append(expert_opt_path.format(expert, trajectory, str(s)))
    return expert_starts, expert_opt_starts


def sgd_step(params, grad, opt_state, opt_params):
    '''Performs a standard step of SGD that is differentiable in the labels'''
    weight_decay = opt_params['weight_decay']
    momentum = opt_params['momentum']
    dampening = opt_params['dampening']
    nesterov = opt_params['nesterov']

    d_p = grad
    if weight_decay != 0:
        d_p = d_p.add(params, alpha=weight_decay)
    if momentum != 0:
        if 'momentum_buffer' not in opt_state:
            buf = opt_state['momentum_buffer'] = torch.zeros_like(params.data)
            buf = buf.mul(momentum).add(d_p)
        else:
            buf = opt_state['momentum_buffer']
            buf = buf.mul(momentum).add(d_p, alpha=1 - dampening)
        if nesterov:
            d_p = d_p.add(buf, alpha=momentum)
        else:
            d_p = buf

    return params.add(d_p, alpha=-opt_params['lr'])


def extract_labels(dataset, label_temp, n_classes=10):
    '''Extracts the labels from a dataset'''
    labels = []
    for _, y in dataset:
        base = np.zeros(n_classes)
        base[y] = label_temp
        labels.append(torch.FloatTensor(base))
    return labels


def coalesce_attack_config(attack_config):
    '''Coalesces the attack config with the default config'''
    expert_kwargs = attack_config.get('expert_kwargs', {})
    labels_kwargs = attack_config.get('labels_kwargs', {})
    attack_config['expert_kwargs'] = {**DEFAULT_SGD_KWARGS, **expert_kwargs}
    attack_config['labels_kwargs'] = {**DEFAULT_SGD_LABELS_KWARGS,
                                      **labels_kwargs}
    return {**DEFAULT_ATTACK_CONFIG, **attack_config}
