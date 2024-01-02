'''
Modified implementation of LeNet-1 for MNIST / Fashion-MNIST inspired by [1]
and heavily influenced by [2]

Total Parameters: 3246


References:
[1] Y. Le Cun, B. Boser, J. S. Denker, R. E. Howard, W. Habbard, L. D. Jackel,
    and D. Henderson. 1990. Handwritten digit recognition with a
    back-propagation network. Advances in neural information processing
    systems 2. Morgan Kaufmann Publishers Inc.,
    San Francisco, CA, USA, 396-404.W
[2] https://github.com/grvk/lenet-1

If you use this implementation in you work, please don't forget to mention the
original author, George Revkov.
'''

# Modified by Rishi Jha to match the experimental setup
# used in https://github.com/MadryLab/backdoor_data_poisoning

import torch.nn as nn


class LeNet1(nn.Sequential):
    def __init__(self):
        model_list = [
            nn.Conv2d(1, 4, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(4, 12, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(12 * 4 * 4, 10)
        ]

        super(LeNet1, self).__init__(*model_list)


class LeNet5(nn.Sequential):
    def __init__(self):
        model_list = [
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        ]

        super(LeNet5, self).__init__(*model_list)


class LeNetBGMD(nn.Sequential):
    def __init__(self):
        model_list = [
            nn.Conv2d(1, 64, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10),
        ]

        super(LeNetBGMD, self).__init__(*model_list)
