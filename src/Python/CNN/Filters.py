import numpy as np


class ConvLayer:
    
    def __init__(
        self, 
        image_dim=(1, 28, 28),
        kernels=(2, 3, 3, 1),
        padding=1,
        stride=1,
        bias=0.1,
        lr=0.01
    ):
        assert image_dim[0] == kernels[3], "Error: Image and kernel channel miss match!"
        
        self.image_dim = (image_dim[0], image_dim[1] + 2 * padding, image_dim[2] + 2 * padding)
        self.kernels = kernels
        self.padding = padding
        self.stride = stride
        self.bias = [bias for i in range(self.kernels[0])]
        self.filters = np.random.rand(*kernels)*0.1
        