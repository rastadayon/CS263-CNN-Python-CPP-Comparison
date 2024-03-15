import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, channel_size=16, kernel_size=3, num_layers=5):
        super(ConvNet, self).__init__()

        self.conv_layers = nn.ModuleList()

        for layer in range(num_layers):
            in_channels = input_channels if layer == 0 else channel_size * (2 ** (layer - 1))
            out_channels = channel_size * (2 ** layer)

            padding = (kernel_size + 1) // 2 if kernel_size % 2 == 1 else kernel_size // 2

            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

            self.conv_layers.append(conv_layer)

        # Define the remaining layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
