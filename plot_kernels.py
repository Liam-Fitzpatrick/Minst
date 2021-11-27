from Network import Net
import torch
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('results/model.pth'))
    kernels = model.conv1.weight.data.numpy()

    layer = 1
    # filter = model.features[layer].weight.data.clone()
    filter = model.conv1.weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()