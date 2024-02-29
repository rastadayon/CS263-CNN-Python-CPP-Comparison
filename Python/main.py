import numpy as np
import matplotlib.pyplot as plt
from utils.utils import get_mnist, plot_random_images
from CNN.CNN import *

from CNN.Filters import *

np.random.seed(42)

if __name__ == '__main__':
    # (train_images, train_labels, valid_images, valid_labels, test_images, test_labels) = get_mnist()
    # plot_random_images(train_images, "Training Set")
    # plot_random_images(valid_images, "Validation Set")
    # plot_random_images(test_images, "Test Set")
    
    # print(f'train_images.shape: {train_images[0].shape}')
    # image = train_images[0].shape
    image = np.random.rand(1, 28, 28)
    print(f'image : {image[0][:5]}')
    
    CNN = CNN()
    # CNN.add_conv_layer(image_dim=(1, 28, 28), kernels=(1, 8, 3, 3), padding=0, stride=2)
    conv = Conv2D(
            image_dim=(1, 28, 28), # (C, H, W)
            kernels=(1, 3, 3, 8), # (C, H_k, W_k, num_kernels)
            padding=0,
            stride=2,
            bias=0.1,
            lr=0.01
        )
    
    out = conv.forward(image)
    print(f'out.shape: {out.shape}')
    print(out[0][:2])