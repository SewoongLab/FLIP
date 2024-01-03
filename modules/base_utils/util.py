import os
import numpy as np
import torch
import tqdm
from functools import partial
from torch import optim
from torch.utils.data import DataLoader, Dataset
from typing import Iterable, Union
from modules.base_utils.model.model import SequentialImageNetwork,\
                                   SequentialImageNetworkMod
import torch.backends.cudnn as cudnn
import toml
from collections import OrderedDict

from modules.base_utils.datasets import make_dataloader

if torch.cuda.is_available():
    cudnn.benchmark = True

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_SGD_BATCH_SIZE = 256
DEFAULT_SGD_EPOCHS = 200
DEFAULT_SGD_KWARGS = {
    'lr': 0.1,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 2e-4
}
DEFAULT_SGD_SCHED_KWARGS = {
    'milestones': [75, 125],
    'gamma': 0.1
}


DEFAULT_ADAM_BATCH_SIZE = 256
DEFAULT_ADAM_EPOCHS = 200
DEFAULT_ADAM_KWARGS = {
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'weight_decay': 1e-4
}
DEFAULT_ADAM_SCHED_KWARGS = {
    'milestones': [125],
    'gamma': 0.1
}

BIG_IMS_MODELS = ['vgg', 'vgg-pretrain', 'vit-pretrain']


def generate_full_path(path):
    return os.path.join(os.getcwd(), path)


def slurmify_path(path, slurm_id):
    if path is None:
        return path
    return path if slurm_id is None else path.format(slurm_id)


def extract_toml(experiment_name, module_name=None):
    relative_path = "experiments/" + experiment_name + "/config.toml"
    full_path = generate_full_path(relative_path)
    assert os.path.exists(full_path)

    exp_toml = toml.load(full_path, _dict=OrderedDict)
    if module_name is not None:
        return exp_toml[module_name]
    return exp_toml
 

def load_model(model_flag, num_classes=10):
    if num_classes != 10 and model_flag not in ['r32p', 'r18', 'r18-tin']:
        raise NotImplementedError

    if model_flag == "r32p":
        import modules.base_utils.model.resnet as resnet

        return SequentialImageNetworkMod(resnet.resnet32(num_classes)).cuda()
    elif model_flag == "r18":
        from pytorch_cifar.models import ResNet, BasicBlock        
        return SequentialImageNetwork(ResNet(BasicBlock,
                                             [2, 2, 2, 2],
                                             num_classes)).cuda()
    elif model_flag == "r18-tin":
        from pytorch_cifar.models import ResNet, BasicBlock

        model = SequentialImageNetwork(ResNet(BasicBlock,
                                             [2, 2, 2, 2],
                                             num_classes))
        model[13] = torch.nn.Linear(2048, 200)

        return model.cuda()
    elif model_flag == "vgg":
        from torchvision.models import vgg19_bn
        
        return vgg19_bn(num_classes=num_classes).cuda()
    elif model_flag == "vgg-pretrain":
        from torchvision.models import vgg19_bn, VGG19_BN_Weights
        model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad=False

        return model.cuda()
    elif model_flag == "vit-pretrain":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = torch.nn.Linear(768, num_classes)
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad=False
        return model.cuda()
    else:
        raise NotImplementedError


def make_pbar(*args, **kwargs):
    pbar_constructor = (
        partial(tqdm.tqdm, dynamic_ncols=True)
    )
    return pbar_constructor(*args, **kwargs)


def get_module_device(module: torch.nn.Module, check=True):
    if check:
        assert len(set(param.device for param in module.parameters())) == 1
    return next(module.parameters()).device


def either_dataloader_dataset_to_both(
    data: Union[DataLoader, Dataset], *, batch_size=None, eval=False, **kwargs
):
    if isinstance(data, DataLoader):
        dataloader = data
        dataset = data.dataset
    elif isinstance(data, Dataset):
        dataset = data
        dl_kwargs = {}

        if eval:
            dl_kwargs.update(dict(batch_size=256, shuffle=False,
                                  drop_last=False))
        else:
            dl_kwargs.update(dict(batch_size=128, shuffle=True))

        if batch_size is not None:
            dl_kwargs["batch_size"] = batch_size

        dl_kwargs.update(kwargs)
        dataloader = make_dataloader(data, **dl_kwargs)
    else:
        raise NotImplementedError()
    return dataloader, dataset


clf_loss = torch.nn.CrossEntropyLoss()
total_mse_distance = torch.nn.MSELoss(reduction="sum")
softmax = torch.nn.Softmax(dim=1)


def clf_correct(y_pred: torch.Tensor, y: torch.Tensor):
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    y_hat = y_pred.data.max(1)[1]
    correct = (y_hat == y).long().cpu().sum()
    return correct


def clf_eval(model: torch.nn.Module, data: Union[DataLoader, Dataset]):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(data, eval=True)
    total_correct, total_loss = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = clf_loss(y_pred, y)
            correct = clf_correct(y_pred, y)

            total_correct += correct.item()
            total_loss += loss.item()

    n = len(dataloader.dataset)
    total_correct /= n
    total_loss /= n
    return total_correct, total_loss


def get_mean_lr(opt: optim.Optimizer):
    return np.mean([group["lr"] for group in opt.param_groups])


def mini_train(
    *,
    model: torch.nn.Module,
    train_data: Union[DataLoader, Dataset],
    test_data: Union[Union[DataLoader, Dataset],
                     Iterable[Union[DataLoader, Dataset]]] = None,
    batch_size=32,
    opt: optim.Optimizer,
    scheduler,
    epochs: int,
    shuffle=True,
    callback=None,
    record=False
):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(train_data,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle)
    n = len(dataloader.dataset)
    total_examples = epochs * n

    if test_data:
        num_sets = 1
        if isinstance(test_data, Iterable):
            num_sets = len(test_data)
        else:
            test_data = [test_data]
        acc_loss = [[] for _ in range(num_sets)]

    with make_pbar(total=total_examples) as pbar:
        for epoch in range(1, epochs + 1):
            train_epoch_loss, train_epoch_correct = 0, 0
            model.train()
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                minibatch_size = len(x)
                model.zero_grad()
                y_pred = model(x)
                loss = clf_loss(y_pred, y)
                correct = clf_correct(y_pred, y)
                loss.backward()
                opt.step()
                train_epoch_correct += int(correct.item())
                train_epoch_loss += float(loss.item())
                pbar.update(minibatch_size)
                if callback is not None:
                    callback(model, opt, epoch, i)

            lr = get_mean_lr(opt)
            if scheduler:
                scheduler.step()

            pbar_postfix = {
                "acc": "%.2f" % (train_epoch_correct / n * 100),
                "loss": "%.4g" % (train_epoch_loss / n),
                "lr": "%.3g" % lr,
            }
            if test_data:
                for i, dataset in enumerate(test_data):
                    acc, loss = clf_eval(model, dataset)
                    pbar_postfix.update(
                        {
                            "acc" + str(i): "%.2f" % (acc * 100),
                            # "loss" + str(i): "%.4g" % loss,
                        }
                    )
                    if record:
                        acc_loss[i].append((acc, loss))
            pbar.set_postfix(**pbar_postfix)

    if record:
        return model, *acc_loss
    return model


def get_train_info(
    params,
    train_flag,
    batch_size=None,
    epochs=None,
    optim_kwargs={},
    scheduler_kwargs={}
):
    if train_flag == "sgd":
        batch_size = batch_size or DEFAULT_SGD_BATCH_SIZE
        epochs = epochs or DEFAULT_SGD_EPOCHS
        kwargs = {**DEFAULT_SGD_KWARGS, **optim_kwargs}
        sched_kwargs = {**DEFAULT_SGD_SCHED_KWARGS, **scheduler_kwargs}
        opt = optim.SGD(params, **kwargs)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, **sched_kwargs)
    elif train_flag == "adam":
        batch_size = batch_size or DEFAULT_ADAM_BATCH_SIZE
        epochs = epochs or DEFAULT_ADAM_EPOCHS
        kwargs = {**DEFAULT_ADAM_KWARGS, **optim_kwargs}
        sched_kwargs = {**DEFAULT_ADAM_SCHED_KWARGS, **scheduler_kwargs}
        opt = optim.Adam(params, **kwargs)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, **sched_kwargs)
    else:
        raise NotImplementedError

    return batch_size, epochs, opt, lr_scheduler


def get_mtt_attack_info(
    expert_params,
    labels,
    expert_kwargs,
    labels_kwargs,
    train_flag='sgd',
    batch_size=None,
    epochs=None
):
    if train_flag == "sgd":
        batch_size = batch_size or DEFAULT_SGD_BATCH_SIZE
        epochs = epochs or DEFAULT_SGD_EPOCHS
        opt_expert = optim.SGD(expert_params, **expert_kwargs)
        opt_labels = optim.SGD([labels], **labels_kwargs)
    else:
        raise NotImplementedError

    assert len(opt_expert.state_dict()['param_groups']) == 1
    return batch_size, epochs, opt_expert, opt_labels


def needs_big_ims(model_flag):
    return model_flag in BIG_IMS_MODELS
