import torch

from hamiltonian.Network import Net
from Datasets.FashionMNIST_data import train_loader, test_loader
from Test import test
from Train import gradient_train
from Visualisation.Display import plot_loss, plot_prediction_sample
from hamiltonian.optimser import hamiltonian

n_epochs = 7

learning_rate = 0.0001

random_seed = 2
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

network = Net()
# optimizer = optim.Adam(network.parameters(), lr=learning_rate)
optimizer = hamiltonian(network.parameters(), lr=0.7, eps=0.1)

if __name__ == '__main__':
    test_losses = [test(network, test_loader)]

    for epoch in range(1, n_epochs + 1):
        _train_losses, _train_counts = gradient_train(epoch, network, train_loader, optimizer)
        train_losses += _train_losses
        train_counter += _train_counts
        test_losses.append(test(network, test_loader))

    plot_loss(train_counter, train_losses, test_counter, test_losses)
    plot_prediction_sample(network, test_loader)
    torch.save(network.state_dict(), 'results/modelstate.pth')