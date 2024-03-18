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

int main(int argc, char* argv []) {
    std::cout << "Convolutional Neural Network\n\n";

    assert(argc == 7);

    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t num_classes = 10;
    const int64_t is_profiling = std::stoi(argv[1]);
    const int64_t batch_size = std::stoi(argv[2]);
    size_t num_epochs; 
    if (is_profiling)
        num_epochs = 1;
    else 
        num_epochs = std::stoi(argv[3]);
    const int64_t kernel_size = std::stoi(argv[4]);
    const int64_t num_layers = std::stoi(argv[5]);
    const std::string optimizer_name = argv[6];

    // print configuration
    std::cout << "is_profiling : " << is_profiling << std::endl;
    std::cout << "batch_size : " << batch_size << " - num_epochs : " << num_epochs <<
        " - kernel_size : " << kernel_size << " - num_layers : " << num_layers <<
        " - optimizer_name : " << optimizer_name << std::endl;
    
    const double learning_rate = 1e-3;
    const double weight_decay = 1e-3;
    const std::string CIFAR_data_path = "../../../../data/cifar10/";
    
    auto train_dataset = CIFAR10(CIFAR_data_path)
        .map(torch::data::transforms::Stack<>());

    auto num_train_samples = train_dataset.size().value();
    std::cout << "This is the number of training samples: " << num_train_samples <<std::endl;
    auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();
    std::cout << "This is the number of testing samples: " << num_test_samples << std::endl;
    
    
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);
    std::cout << "Train Loader Done!" << std::endl;

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);
    std::cout << "Test Loader Done!" << std::endl;

    std::cout << std::fixed << std::setprecision(4);

    if (num_layers == 3) {
        // Train the model
        double total_training_time = 0.0;
        double total_testing_time = 0.0;
        for (int i = 0; i != 10; ++i) {
            auto model = ConvNetBase(num_classes, kernel_size);
            std::unique_ptr<torch::optim::Optimizer> optimizer;
            if (optimizer_name == "SGD") {
                optimizer = std::make_unique<torch::optim::SGD>(model->parameters(),
                    torch::optim::SGDOptions(learning_rate).weight_decay(weight_decay));
            } else if (optimizer_name == "Adam") {
                optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), 
                    torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
            } else {
                std::cerr << "Invalid optimizer name: " << optimizer_name << std::endl;
                return 1;
            }
            std::cout << "Optimizer Done!" << std::endl;
            std::cout << "Training...\n";
            std::cout << "Round: " << i + 1 << std::endl;
            auto training_start = std::chrono::high_resolution_clock::now();
            for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
                double running_loss = 0.0;
                size_t num_correct = 0;
            
                int batch_num = 0;
                for (auto& batch : *train_loader) {
                    // Transfer images and target labels to device
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = model->forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);

                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);

                    num_correct += prediction.eq(target).sum().item<int64_t>();

                    optimizer->zero_grad();
                    loss.backward();
                    optimizer->step();
                    batch_num += 1;
                }

                auto sample_mean_loss = running_loss / num_train_samples;
                auto accuracy = static_cast<double>(num_correct) / num_train_samples;

                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                    << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
            }
            auto training_end = std::chrono::high_resolution_clock::now();
            total_training_time += 
                std::chrono::duration_cast<std::chrono::nanoseconds>(training_end - training_start).count() * 1e-9;

            if (is_profiling == 0) {
                auto testing_start = std::chrono::high_resolution_clock::now();
                model->eval();
                torch::InferenceMode no_grad;

                double running_loss = 0.0;
                size_t num_correct = 0;

                for (const auto& batch : *test_loader) {
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = (*model).forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);
                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);
                    num_correct += prediction.eq(target).sum().item<int64_t>();
                }
                auto testing_end = std::chrono::high_resolution_clock::now();
                total_testing_time += 
                    std::chrono::duration_cast<std::chrono::nanoseconds>(testing_end - testing_start).count() * 1e-9;
                auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
                auto test_sample_mean_loss = running_loss / num_test_samples;

                std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
            }
        }
        std::cout << "Training time is : " <<  std::fixed 
                    << total_training_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;
        
        std::cout << "Testing time is : " <<  std::fixed 
                    << total_testing_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;
    } 
    else if (num_layers == 4) {
        // Train the model
        double total_training_time = 0.0;
        double total_testing_time = 0.0;
        for (int i = 0; i != 10; ++i) {
            auto model = ConvNet4Layer(num_classes, kernel_size);
            std::unique_ptr<torch::optim::Optimizer> optimizer;
            if (optimizer_name == "SGD") {
                optimizer = std::make_unique<torch::optim::SGD>(model->parameters(),
                    torch::optim::SGDOptions(learning_rate).weight_decay(weight_decay));
            } else if (optimizer_name == "Adam") {
                optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), 
                    torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
            } else {
                std::cerr << "Invalid optimizer name: " << optimizer_name << std::endl;
                return 1;
            }
            std::cout << "Optimizer Done!" << std::endl;
            std::cout << "Training...\n";
            std::cout << "Round: " << i + 1 << std::endl;
            auto training_start = std::chrono::high_resolution_clock::now();
            for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
                double running_loss = 0.0;
                size_t num_correct = 0;
            
                int batch_num = 0;
                for (auto& batch : *train_loader) {
                    // Transfer images and target labels to device
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = model->forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);

                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);

                    num_correct += prediction.eq(target).sum().item<int64_t>();

                    optimizer->zero_grad();
                    loss.backward();
                    optimizer->step();
                    batch_num += 1;
                }

                auto sample_mean_loss = running_loss / num_train_samples;
                auto accuracy = static_cast<double>(num_correct) / num_train_samples;

                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                    << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
            }
            auto training_end = std::chrono::high_resolution_clock::now();
            total_training_time += 
                std::chrono::duration_cast<std::chrono::nanoseconds>(training_end - training_start).count() * 1e-9;

            if (is_profiling == 0) {
                auto testing_start = std::chrono::high_resolution_clock::now();
                model->eval();
                torch::InferenceMode no_grad;

                double running_loss = 0.0;
                size_t num_correct = 0;

                for (const auto& batch : *test_loader) {
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = model->forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);
                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);
                    num_correct += prediction.eq(target).sum().item<int64_t>();
                }
                auto testing_end = std::chrono::high_resolution_clock::now();
                total_testing_time += 
                    std::chrono::duration_cast<std::chrono::nanoseconds>(testing_end - testing_start).count() * 1e-9;
                auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
                auto test_sample_mean_loss = running_loss / num_test_samples;

                std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
            }
        }
        std::cout << "Training time is : " <<  std::fixed 
                    << total_training_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;
        
        std::cout << "Testing time is : " <<  std::fixed 
                    << total_testing_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;
    }
    else if (num_layers == 5) {
        // Train the model
        double total_training_time = 0.0;
        double total_testing_time = 0.0;
        for (int i = 0; i != 10; ++i) {
            auto model = ConvNet5Layer(num_classes, kernel_size);
            std::unique_ptr<torch::optim::Optimizer> optimizer;
            if (optimizer_name == "SGD") {
                optimizer = std::make_unique<torch::optim::SGD>(model->parameters(),
                    torch::optim::SGDOptions(learning_rate).weight_decay(weight_decay));
            } else if (optimizer_name == "Adam") {
                optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), 
                    torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
            } else {
                std::cerr << "Invalid optimizer name: " << optimizer_name << std::endl;
                return 1;
            }
            std::cout << "Optimizer Done!" << std::endl;
            std::cout << "Training...\n";
            std::cout << "Round: " << i + 1 << std::endl;
            auto training_start = std::chrono::high_resolution_clock::now();
            for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
                double running_loss = 0.0;
                size_t num_correct = 0;
            
                int batch_num = 0;
                for (auto& batch : *train_loader) {
                    // Transfer images and target labels to device
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = model->forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);

                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);

                    num_correct += prediction.eq(target).sum().item<int64_t>();

                    optimizer->zero_grad();
                    loss.backward();
                    optimizer->step();
                    batch_num += 1;
                }

                auto sample_mean_loss = running_loss / num_train_samples;
                auto accuracy = static_cast<double>(num_correct) / num_train_samples;

                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                    << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
            }
            auto training_end = std::chrono::high_resolution_clock::now();
            total_training_time += 
                std::chrono::duration_cast<std::chrono::nanoseconds>(training_end - training_start).count() * 1e-9;

            if (is_profiling == 0) {
                auto testing_start = std::chrono::high_resolution_clock::now();
                model->eval();
                torch::InferenceMode no_grad;

                double running_loss = 0.0;
                size_t num_correct = 0;

                for (const auto& batch : *test_loader) {
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = model->forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);
                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);
                    num_correct += prediction.eq(target).sum().item<int64_t>();
                }
                auto testing_end = std::chrono::high_resolution_clock::now();
                total_testing_time += 
                    std::chrono::duration_cast<std::chrono::nanoseconds>(testing_end - testing_start).count() * 1e-9;
                auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
                auto test_sample_mean_loss = running_loss / num_test_samples;

                std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
            }
        }
        std::cout << "Training time is : " <<  std::fixed 
                    << total_training_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;
        
        std::cout << "Testing time is : " <<  std::fixed 
                    << total_testing_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;

    } 
    else if (num_layers == 6) {
        // Train the model
        double total_training_time = 0.0;
        double total_testing_time = 0.0;
        for (int i = 0; i != 10; ++i) {
            auto model = ConvNet6Layer(num_classes, kernel_size);
            std::unique_ptr<torch::optim::Optimizer> optimizer;
            if (optimizer_name == "SGD") {
                optimizer = std::make_unique<torch::optim::SGD>(model->parameters(),
                    torch::optim::SGDOptions(learning_rate).weight_decay(weight_decay));
            } else if (optimizer_name == "Adam") {
                optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), 
                    torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
            } else {
                std::cerr << "Invalid optimizer name: " << optimizer_name << std::endl;
                return 1;
            }
            std::cout << "Optimizer Done!" << std::endl;
            std::cout << "Training...\n";
            std::cout << "Round: " << i + 1 << std::endl;
            auto training_start = std::chrono::high_resolution_clock::now();
            for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
                double running_loss = 0.0;
                size_t num_correct = 0;
            
                int batch_num = 0;
                for (auto& batch : *train_loader) {
                    // Transfer images and target labels to device
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = model->forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);

                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);

                    num_correct += prediction.eq(target).sum().item<int64_t>();

                    optimizer->zero_grad();
                    loss.backward();
                    optimizer->step();
                    batch_num += 1;
                }

                auto sample_mean_loss = running_loss / num_train_samples;
                auto accuracy = static_cast<double>(num_correct) / num_train_samples;

                std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
                    << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
            }
            auto training_end = std::chrono::high_resolution_clock::now();
            total_training_time += 
                std::chrono::duration_cast<std::chrono::nanoseconds>(training_end - training_start).count() * 1e-9;

            if (is_profiling == 0) {
                auto testing_start = std::chrono::high_resolution_clock::now();
                model->eval();
                torch::InferenceMode no_grad;

                double running_loss = 0.0;
                size_t num_correct = 0;

                for (const auto& batch : *test_loader) {
                    auto data = batch.data.to(device);
                    auto target = batch.target.to(device);

                    auto output = model->forward(data);

                    auto loss = torch::nn::functional::cross_entropy(output, target);
                    running_loss += loss.item<double>() * data.size(0);

                    auto prediction = output.argmax(1);
                    num_correct += prediction.eq(target).sum().item<int64_t>();
                }
                auto testing_end = std::chrono::high_resolution_clock::now();
                total_testing_time += 
                    std::chrono::duration_cast<std::chrono::nanoseconds>(testing_end - testing_start).count() * 1e-9;
                auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
                auto test_sample_mean_loss = running_loss / num_test_samples;

                std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
            }
        }
        std::cout << "Training time is : " <<  std::fixed 
                    << total_training_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;
        
        std::cout << "Testing time is : " <<  std::fixed 
                    << total_testing_time / 10 << std::setprecision(9);
                std::cout << " sec" <<  std::endl;

    } else {
        std::cerr << "Invalid number of layers: " << num_layers << std::endl;
        return 1;
    }
}
