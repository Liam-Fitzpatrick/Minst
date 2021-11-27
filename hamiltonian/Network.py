import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print(x.size())
        x = x.view(-1, 448)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding='same')
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 7 * 7, 16 * 7 * 7)  # 5*5 from image dimension
        self.fc2 = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        print(x.size())
        x = x.view(-1, 16 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DenseBlock(nn.Module):
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
        out = F.max_pool2d(out, (2, 2))
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


class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()
        self.dense1 = DenseBlock(1, 8)
        self.dense2 = DenseBlock(32, 8)
        self.dense3 = DenseBlock(32, 16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 3 * 3, 64 * 3 * 3)  # 5*5 from image dimension
        self.fc2 = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        # print(x.size())
        x = x.view(-1, 64 * 3 * 3)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)


class FashionCNN2(nn.Module):

    def __init__(self):
        super(FashionCNN2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        out = torch.cat([x1, x2], 1)
        out = F.max_pool2d(out, (4, 4))
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return F.log_softmax(out, dim=1)


class FashionCNN3(nn.Module):

    def __init__(self):
        super(FashionCNN3, self).__init__()

        self.layer1 = DenseBlock(1, 64)
        self.layer2 = DenseBlock(64, 64)
        self.layer3 = DenseBlock(64, 128)
        self.layer4 = DenseBlock(128, 256)
        self.layer5 = DenseFlat(256, 600)
        self.fc1 = nn.Linear(in_features=600, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=10)
        # self.drop = nn.Dropout(0.40)
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):

        out = self.layer1(x)
        out = self.drop(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        out = self.fc1(out)
        # out = self.drop(out)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


def Net():
    return FashionCNN3()
