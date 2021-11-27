import torch
from matplotlib import pyplot as plt

from FashionMNIST_data import ClassLabels


def plot_sample(loader):
    plt.figure("1")
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plot_loss(train_counter, train_losses, test_counter, test_losses):
    plt.figure("2")
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


def plot_prediction_sample(network, loader):
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)

    plt.figure("3")
    with torch.no_grad():
        output = network(example_data)
    for i in range(6):
        label = output.data.max(1, keepdim=True)[1][i].item()
        label = ClassLabels(label).name
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            label))
        plt.xticks([])
        plt.yticks([])

    plt.show()