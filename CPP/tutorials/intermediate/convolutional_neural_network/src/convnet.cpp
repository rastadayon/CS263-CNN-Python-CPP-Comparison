// Copyright 2020-present pytorch-cpp Authors
#include "convnet.h"
#include <torch/torch.h>

ConvNetBaseImpl::ConvNetBaseImpl(int64_t num_classes, int64_t kernel_size)
    : kernel_size(kernel_size), fc(64, num_classes), pad((kernel_size + 1) / 2) {
    std::cerr << "pad: " << pad << std::endl; 
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("pool", pool),
    register_module("fc", fc);
}

torch::Tensor ConvNetBaseImpl::forward(torch::Tensor x) {
    // std::cerr << "Entering the forward" << std::endl;
    std::cerr << "input size = : (" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) <<")" << std::endl;
    x = layer1->forward(x);
    std::cerr << "layer1 fine - x.size : (" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << ")" << std::endl;
    x = layer2->forward(x);
    std::cerr << "layer2 fine - x.size : (" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << ")" << std::endl;
    x = layer3->forward(x);
    std::cerr << "layer3 fine - x.size : (" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << ")" << std::endl;
    x = pool->forward(x);
    std::cerr << "pool fine - x.size : (" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << ")" << std::endl;
    x = x.view({x.size(0), -1});
    std::cerr << "x after view - x.size : (" << x.size(0) <<  ", " << x.size(1) << ")" << std::endl;
    x = fc->forward(x);
    std::cerr << "x after fc - x.size : (" << x.size(0) <<  ", " << x.size(1) << ")" << std::endl;
    return x;
}


ConvNet4LayerImpl::ConvNet4LayerImpl(int64_t num_classes, int64_t kernel_size)
    : kernel_size(kernel_size), fc(128, num_classes) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("pool", pool),
    register_module("fc", fc);
}

torch::Tensor ConvNet4LayerImpl::forward(torch::Tensor x) {
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);
    x = pool->forward(x);
    //x = x.view({-1,  64 * 4 * 4});
    x = x.view({x.size(0), -1});
    return fc->forward(x);
}


ConvNet5LayerImpl::ConvNet5LayerImpl(int64_t num_classes, int64_t kernel_size)
    : kernel_size(kernel_size), fc(256, num_classes) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("layer5", layer5);
    register_module("pool", pool),
    register_module("fc", fc);
}

torch::Tensor ConvNet5LayerImpl::forward(torch::Tensor x) {
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);
    x = layer5->forward(x);
    x = pool->forward(x);
    //x = x.view({-1,  64 * 4 * 4});
    x = x.view({x.size(0), -1});
    return fc->forward(x);
}


ConvNet6LayerImpl::ConvNet6LayerImpl(int64_t num_classes, int64_t kernel_size)
    : kernel_size(kernel_size), fc(512, num_classes) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("layer5", layer5);
    register_module("layer6", layer6);
    register_module("pool", pool),
    register_module("fc", fc);
}

torch::Tensor ConvNet6LayerImpl::forward(torch::Tensor x) {
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);
    x = layer5->forward(x);
    x = layer6->forward(x);
    x = pool->forward(x);
    //x = x.view({-1,  64 * 4 * 4});
    x = x.view({x.size(0), -1});
    return fc->forward(x);
}