import torch

from modules.base_utils.util import generate_full_path


def checkpoint_callback(model, opt, epoch, iteration, save_iter, output_dir):
    '''Saves model and optimizer state dicts at fixed intervals.'''
    if iteration % save_iter == 0 and iteration != 0:
        checkpoint_path = f'{output_dir}model_{str(epoch)}_{str(iteration)}.pth'
        opt_path = f'{output_dir}model_{str(epoch)}_{str(iteration)}_opt.pth'
        torch.save(model.state_dict(), generate_full_path(checkpoint_path))
        torch.save(opt.state_dict(), generate_full_path(opt_path))