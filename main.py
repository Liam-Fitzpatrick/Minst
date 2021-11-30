import torch
from torch import optim

from hamiltonian.Network import Net
from Datasets.FashionMNIST_data import train_loader, test_loader
from Test import test
from Train import gradient_train
from Visualisation.Display import plot_loss, plot_prediction_sample
from hamiltonian.optimser import hamiltonian

n_epochs = 10
random_seed = 1
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

network = Net()
if torch.cuda.is_available():
    network = network.cuda()
adam_optimizer = optim.Adam(network.parameters(), lr=0.0001)
hamiltonian_optimizer = hamiltonian(network.parameters(), lr=0.6, eps=0.5)

if __name__ == '__main__':
    test_losses = [test(network, test_loader)]

    #results of adam
    # for epoch in range(1, n_epochs + 1):
    #     _train_losses, _train_counts = gradient_train(epoch, network, train_loader, adam_optimizer)
    #     train_losses += _train_losses
    #     train_counter += _train_counts
    #     test_losses.append(test(network, test_loader))
    #
    # plot_loss(train_counter, train_losses, test_counter, test_losses, "adam")
    # plot_prediction_sample(network, test_loader, "adam")
    # torch.save(network.state_dict(), 'results/modelstate_adam.pth')

    #results of hamiltonian
    for epoch in range(1, n_epochs + 1):
        _train_losses, _train_counts = gradient_train(epoch, network, train_loader, hamiltonian_optimizer)
        train_losses += _train_losses
        train_counter += _train_counts
        test_losses.append(test(network, test_loader))

    plot_loss(train_counter, train_losses, test_counter, test_losses, "hamiltonian")
    plot_prediction_sample(network, test_loader, "hamiltonian")
    torch.save(network.state_dict(), 'results/modelstate_hamiltonian.pth')
