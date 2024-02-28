import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_mnist, plot_random_images
from CNN.CNN import *

if __name__ == '__main__':
    (train_images, train_labels, valid_images, valid_labels, test_images, test_labels) = get_mnist()
    # plot_random_images(train_images, "Training Set")
    # plot_random_images(valid_images, "Validation Set")
    # plot_random_images(test_images, "Test Set")
    
    CNN = CNN()
    CNN.add_conv_layer(image_dim=(1, 28, 28), kernels=(1, 8, 3, 3), padding=0, stride=2)