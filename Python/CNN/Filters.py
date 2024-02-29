import numpy as np
from utils.activations import *

np.random.seed(42)

class Conv2D:
    
    def __init__(
        self, 
        image_dim=(1, 28, 28), # (C, H, W)
        kernels=(1, 3, 3, 2), # (C, H_k, W_k, num_kernels)
        padding=1,
        stride=1,
        bias=0.1,
        lr=0.01
    ):
        assert image_dim[0] == kernels[0], "Error: Image and kernel channel miss match!"
        
        self.image_dim = (image_dim[0], image_dim[1] + 2 * padding, image_dim[2] + 2 * padding)
        self.kernel_dim = (kernels[0], kernels[1], kernels[2])
        self.num_kernels = kernels[3]
        self.padding = padding
        self.stride = stride
        self.bias = np.ones(self.num_kernels) * bias
        self.filters = np.random.rand(self.num_kernels, *self.kernel_dim)*0.1
        self.lr = lr
                
    
    def add_padding(self, image):
        '''
            Given an input image of type np.ndarray, adds padding to height and width of it 
        '''
        if len(image.shape) == 2:  # Grayscale image
            height, width = image.shape
            padded_image = np.zeros((height + 2 * self.padding, width + 2 * self.padding))
            padded_image[self.padding:self.padding + height, self.padding:self.padding + width] = image
        elif len(image.shape) == 3:  # Color or multi-channel image
            depth, height, width = image.shape
            padded_image = np.zeros((depth, height + 2 * self.padding, width + 2 * self.padding))
            padded_image[:, self.padding:self.padding + height, self.padding:self.padding + width] = image
        else:
            raise ValueError("Input image must be 2D (grayscale) or 3D (color/multi-channel) array.")

        return padded_image
    
    def out_dim(self, h_k, w_k): 
        '''
            Given the height and width of the kernel, returns the size of the output
        '''
        h_out = int((self.image_dim[1] - h_k + 2 * self.padding)/self.stride) +1
        w_out = int((self.image_dim[2] - w_k + 2 * self.padding)/self.stride) +1
        c_out = self.image_dim[0]
        return h_out, w_out, c_out

    def forward(self, image):
        '''
            Given the input image and the kernels, outputs image of size (num_kernels, h_out, w_out)
            h_out = int((h_in - h_k + 2 * p) / s) + 1
            w_out = int((w_in - w_k + 2 * p) / s) + 1
        '''
        
        if self.padding != 0:
            image = self.add_padding(image)
        
        self.cache=image # not sure what this does
        
        c_k, h_k, w_k = (self.kernel_dim)
        print(f'h_k: {h_k}, w_k: {w_k}, c_k: {c_k}')
        h_out, w_out, c_out = self.out_dim(h_k, w_k)
        print(f'out_dim: {h_out, w_out}')
        image_out = np.zeros((self.num_kernels, h_out, w_out))
        
        for filter_num in range(self.num_kernels):
            for c in range(c_k):
                # print(f'c = {c}')
                for i, y in enumerate(np.arange(0, self.image_dim[1] - h_k, self.stride)):
                    for j, x in enumerate(np.arange(0, self.image_dim[2] - w_k, self.stride)):
                        # print(f'{i}. y:{y}, {j}. x:{x}, ')
                        for y_k in range(h_k):
                            for x_k in range(w_k):
                                image_out[filter_num][i][j] += (image[c][y + y_k][x + x_k] * self.filters[filter_num][c][y_k][x_k])
        
        return leaky_relu(image_out)                        
		