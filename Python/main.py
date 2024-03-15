from CNN import ConvNet
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

def main(prof):
    dataset_name = "CIFAR10"
    num_train_samples = 2000
    num_test_samples = 500
    batch_size = 8
    optimizer = optim.Adam
    learning_rate = 1e-3
    weight_decay = 1e-3
    num_epochs = 1
    
    times_run = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.device(device) == torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_name == 'MNIST':
        try:
            train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
            test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)
        except:
            train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    elif dataset_name == 'CIFAR10':
        try:
            train_dataset = CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
            test_dataset = CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
        except:
            train_dataset = CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset.data = train_dataset.data[:num_train_samples]
    train_dataset.targets = train_dataset.targets[:num_train_samples]
    test_dataset.data = test_dataset.data[:num_test_samples]
    test_dataset.targets = test_dataset.targets[:num_test_samples]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet(
                        input_channels=3,
                        num_classes=10,
                        channel_size=16,
                        kernel_size=3,
                        num_layers=3
                    )

    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.to(device)
    total_training_time, total_testing_time = 0, 0
    for i in range(times_run):
        print(f'ROUND {i + 1} / {times_run}')
        print(f'Starting Training...')
        training_start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            num_correct = 0

            for images, labels in train_loader:
                prof.step()
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                num_correct += (predictions == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = num_correct / len(train_loader.dataset)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Trainset - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        training_end_time = time.time()
        total_training_time += (training_end_time - training_start_time)
        
    #     model.eval()
    #     running_loss = 0.0
    #     num_correct = 0

    #     print(f'Starting Testing...')
    #     testing_start_time = time.time()
    #     with torch.no_grad():
    #         for images, labels in test_loader:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             loss = nn.functional.cross_entropy(outputs, labels)

    #             running_loss += loss.item() * images.size(0)
    #             predictions = outputs.argmax(dim=1)
    #             num_correct += (predictions == labels).sum().item()

    #     print(f'Testing finished!')
    #     testing_end_time = time.time()
    #     total_testing_time += (testing_end_time - testing_start_time)
    #     test_loss = running_loss / len(test_loader.dataset)
    #     test_accuracy = num_correct / len(test_loader.dataset)

    #     print(f'Testset - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        
    # print(f'Total train time: {total_training_time / times_run}')
    # print(f'Total test time: {total_testing_time / times_run}')


if __name__ == "__main__":
    with torch.autograd.profiler.profile(
        profile_memory=True, schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/cnn-py-1epoch'),
        record_shapes=True,
        with_stack=True) as prof:
        with record_function("main"):
            main(prof)

    # Save the profiling results to a file
    prof.export_chrome_trace('profiling_results.json')
    # prof.export_stacks('profiling_results.txt')