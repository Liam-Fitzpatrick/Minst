import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torchvision import transforms

from hamiltonian.Network import Net
from Visualisation.Utility import FeatureExtractor


def dream(image, model, iterations, lr, channel=0):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image).select(1, channel)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.grad.data.zero_()
    return image.cpu().data.numpy()


if __name__ == '__main__':

    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    model = Net()
    model.load_state_dict(torch.load('results/model.pth'))
    model = FeatureExtractor(model, 2)
    for i in range(8):
        img = cv2.imread('testSample/img_4.jpg')
        img = Image.fromarray(img).convert('L')  # img as opencv
        img = np.array(img)
        img = transform(img)
        img = img.unsqueeze(0)
        img_transformed = dream(image=img, model=model, iterations=1000, lr=1e-2, channel=i)
        plt.imsave(f"outputs/model{i}.png", img_transformed[0, 0, :, :], cmap='gray')