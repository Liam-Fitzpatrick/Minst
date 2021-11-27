import torch
import torch.nn.functional as F


def gradient_train(epoch, network, data_loader, optimiser, log_interval=10):
    network.train()
    train_losses = []
    train_counter = []
    for batch_idx, (data, target) in enumerate(data_loader):
        optimiser.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimiser.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset), 100.0 * batch_idx / len(data_loader),
                loss.item()))
            train_losses.append(loss.item())
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimiser.state_dict(), 'results/optimiser.pth')
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(data_loader.dataset)))
    return train_losses, train_counter
