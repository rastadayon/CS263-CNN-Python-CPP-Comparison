// Copyright 2020-present pytorch-cpp Authors
#include "convnet.h"
#include <torch/torch.h>

ConvNetBaseImpl::ConvNetBaseImpl(int64_t num_classes, int64_t kernel_size)
    : kernel_size(kernel_size), fc(64, num_classes) {
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("pool", pool),
    register_module("fc", fc);
}

torch::Tensor ConvNetBaseImpl::forward(torch::Tensor x) {
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = pool->forward(x);
    //x = x.view({-1,  64 * 4 * 4});
    x = x.view({x.size(0), -1});
    return fc->forward(x);
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