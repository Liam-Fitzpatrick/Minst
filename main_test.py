import torch
import torch.optim as optim

from Network import Net
from FashionMNIST_data import train_loader, test_loader
from Test import test
from Train import gradient_train
from Display import plot_loss, plot_prediction_sample

n_epochs = 1
learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

if __name__ == '__main__':
    network = Net()
    network.load_state_dict(torch.load('results/model.pth'))

    test_losses = [test(network, test_loader)]
    plot_prediction_sample(network, test_loader)