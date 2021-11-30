import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    # classic densely connected convolution where the output of each layer is connected to
    # blocks further into the network
    # allows for a lower number of parameters to be used.
    def __init__(self, in_channel, out_channel):
        super(DenseBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        out = torch.cat([x1, x2], 1)
        # out = F.max_pool2d(out, (2, 2))
        return out


class DenseFlat(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseFlat, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=out_channel // 2),
            nn.BatchNorm1d(out_channel // 2),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=out_channel // 2, out_features=out_channel // 2),
            nn.BatchNorm1d(out_channel // 2),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        out = torch.cat([x1, x2], 1)
        return out


class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            DenseBlock(1, 16),
            DenseBlock(16, 32),
            DenseBlock(32, 64),

        )
        self.layer2 = nn.Sequential(
            DenseBlock(64, 80),
            DenseBlock(80, 96),
            DenseBlock(96, 128),

        )
        self.layer3 = nn.Sequential(
            DenseBlock(128, 160),
            DenseBlock(160, 192),
            DenseBlock(192, 224),
        )

        self.layer4 = nn.Sequential(
            DenseBlock(224, 256),
        )
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):

        out = self.layer1(x)
        out = self.drop(out)
        out = F.max_pool2d(out, (2, 2))
        out = self.layer2(out)
        out = self.drop(out)
        out = F.max_pool2d(out, (2, 2))
        out = self.layer3(out)
        out = F.max_pool2d(out, (2, 2))
        out = self.layer4(out)
        out = F.max_pool2d(out, (2, 2))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


def Net():
    return FashionCNN()
