// Copyright 2020-present pytorch-cpp Authors
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "convnet.h"
#include "imagefolder_dataset.h"
#include "cifar10.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>

using dataset::ImageFolderDataset;

int main(int argc,char* argv []) {
    std::cout << "Convolutional Neural Network\n\n";

    // py::exec(R"(import sys)");
    // py::exec(R"(sys.path.append('/usr/local/lib/python3.10/dist-packages'))");
    // py::exec(R"(import torch)");
    // py::exec(R"(from torch.autograd import profiler)");
    // py::exec(R"(profiler.profile(profiler=True))");
    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t batch_size = 8;
    const size_t num_epochs = 1; // set to 5 later
    //std::string dataset_name(argv[3]);
    const double learning_rate = 1e-3;
    const double weight_decay = 1e-3;

    
    const std::string CIFAR_data_path = "../../../../data/cifar10/";
    
    auto train_dataset = CIFAR10(CIFAR_data_path)
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();
    std::cout << "This is the number of training samples: " << num_train_samples <<std::endl;
    auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();
    std::cout << "This is the number of testing samples: " << num_test_samples << std::endl;
    // Data loader

    
    
    
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);
    std::cout << "Train Loader Done!" << std::endl;

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);
    std::cout << "Test Loader Done!" << std::endl;

    // Model
    ConvNet model(num_classes);
    model->to(device);
    std::cout << "Model Done!" << std::endl;

    // Optimizer
    torch::optim::SGD optimizer(
        model->parameters(), torch::optim::SGDOptions(learning_rate).weight_decay(weight_decay));
    std::cout << "Optimizer Done!" << std::endl;

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";
    //time(&start); 
    auto start = std::chrono::high_resolution_clock::now();
    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;
        std::cout << "epoch: " << epoch + 1 << std::endl;
        int batch_num = 0;
        for (auto& batch : *train_loader) {
            // Transfer images and target labels to device
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            std::cout << "Doing batch " << batch_num << std::endl;

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            batch_num += 1;
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }
    auto end = std::chrono::high_resolution_clock::now();
 
    // Calculating total time taken by the program.
    double time_taken = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
 
    time_taken *= 1e-9;
 
    std::cout << "Training time is " <<  std::fixed 
         << time_taken << std::setprecision(9);
     std::cout << " sec" <<  std::endl;
    std::cout << "Training finished!\n\n";
    
    // std::cout << "Testing...\n";
    
    // start = std::chrono::high_resolution_clock::now();
    // // Test the model
    // model->eval();
    // torch::InferenceMode no_grad;

    // double running_loss = 0.0;
    // size_t num_correct = 0;

    // for (const auto& batch : *test_loader) {
    //     auto data = batch.data.to(device);
    //     auto target = batch.target.to(device);

    //     auto output = model->forward(data);

    //     auto loss = torch::nn::functional::cross_entropy(output, target);
    //     running_loss += loss.item<double>() * data.size(0);

    //     auto prediction = output.argmax(1);
    //     num_correct += prediction.eq(target).sum().item<int64_t>();
    // }
    // end = std::chrono::high_resolution_clock::now();
 
    // time_taken = 
    //   std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
 
    // time_taken *= 1e-9;
 
    // std::cout << "Testing time is : " <<  std::fixed 
    //     << time_taken << std::setprecision(9);
    // std::cout << " sec" <<  std::endl;
    // std::cout << "Testing finished!\n";

    // auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    // auto test_sample_mean_loss = running_loss / num_test_samples;

    // std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
    
}
