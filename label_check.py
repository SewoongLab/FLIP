from glob import glob
import numpy as np


for d in ['cifar', 'cifar_100', 'tiny_imagenet']:
    fnms = glob(f'precomputed/{d}/**/true.npy', recursive=True)
    print(f'{d}: {len(fnms)}')
    base = np.load(fnms[0])

    for fnm in fnms:
        arr = np.load(fnm)
        assert np.array_equal(arr, base)