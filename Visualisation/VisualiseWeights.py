import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse

from PIL import Image
from torchvision import models, transforms

from Network import Net

from Visualisation.Utility import get_features, FeatureExtractor

if __name__ == '__main__':

    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the 49 conv layers in this list

    model = Net()
    model.load_state_dict(torch.load('../results/model.pth'))
    model = FeatureExtractor(model, 2).features



    # read and visualize an image
    img = cv2.imread('testSample/img_100.jpg')
    img = Image.fromarray(img).convert('L')  # img as opencv

    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    img = np.array(img)
    img = transform(img)

    print(img.size())
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    # pass the image through all the layers
    results = [model[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(model[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    # visualize 64 features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"../outputs/layer_{num_layer}.png")
        # plt.show()
        plt.close()