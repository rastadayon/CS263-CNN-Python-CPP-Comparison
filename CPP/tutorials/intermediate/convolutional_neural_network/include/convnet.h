// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>

class ConvNetBaseImpl : public torch::nn::Module {
 public:
    explicit ConvNetBaseImpl(int64_t num_classes, int64_t kernel_size);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layer1{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, kernel_size).stride(1).padding(2)),
        torch::nn::BatchNorm2d(16),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer2{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, kernel_size).stride(1).padding(2)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::Sequential layer3{
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, kernel_size).stride(1).padding(2)),
        torch::nn::BatchNorm2d(64),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    };

    torch::nn::AdaptiveAvgPool2d pool{torch::nn::AdaptiveAvgPool2dOptions({1, 1})};

    torch::nn::Linear fc;
};
TORCH_MODULE(ConvNetBase);


