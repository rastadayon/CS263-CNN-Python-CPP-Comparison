
class CNN:
    
    def __init__(self):
        self.layers = []
        self.num_classes = 0
        
        self.train_set, self.valid_set, self.test_set = [], [], []
        self.train_acc, self.valid_acc, self.test_acc = [], [], []
        self.train_loss, self.valid_loss, self.test_loss = [], [], []
        
    def add_conv_layer(
            self, 
            image_dim=(1, 28, 28),
            kernels=(2, 3, 3, 1),
            padding=1,
            stride=1,
            bias=0.1,
            lr=0.01
        ):
        
        self.layers.append(Filter)