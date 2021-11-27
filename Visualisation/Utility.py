import torch.nn as nn
import torchvision.models as models

def get_features(model):
    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the 49 conv layers in this list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")
    return conv_layers, model_weights


def forward_pass(observation, conv_layers, layer=None):
    if not layer:
        layer = len(conv_layers)

    results = [conv_layers[0](observation)]
    if layer == 0:
        return results
    else:
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]))
    return results


class FeatureExtractor(nn.Module):
    def __init__(self, net, num_conv):
        super(FeatureExtractor, self).__init__()
        self.net = net
        # If you treat GooLeNet as a fixed feature extractor, disable the gradients and save some memory
        # for p in self.net.parameters():
        #     p.requires_grad = False
        # Define which layers you are going to extract
        self.features = nn.Sequential(*list(self.net.children())[:num_conv])

    def forward(self, x):
        return self.features(x)
