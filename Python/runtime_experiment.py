from CNN import ConvNet
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import sys

def process_arguments(args):
    kwargs = {}
    for i in range(0, len(args), 2):
        kwargs[args[i]] = args[i + 1] if i + 1 < len(args) else None
    print(f'kwargs: {kwargs}')
    return kwargs
    
if __name__ == "__main__":
    kwargs = process_arguments(args = sys.argv[1:])
    times_run = int(kwargs['times_run'])
    batch_size = int(kwargs['batch_size'])
    num_epochs = int(kwargs['num_epochs'])
    kernel_size = int(kwargs['kernel_size'])
    num_layers = int(kwargs['num_layers'])
    optimizer = optim.Adam if kwargs['optimizer'] == 'adam' else optim.SGD
    dataset_name = "CIFAR10"
    num_train_samples = 2000
    num_test_samples = 500
    learning_rate = 1e-3
    weight_decay = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    train_dataset.transform = transform
    test_dataset.transform = transform

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
                        kernel_size=kernel_size,
                        num_layers=num_layers
                    )
    
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.to(device)
    
    total_training_time, total_testing_time = 0, 0
    for i in range(times_run):
        print(f'ROUND {i + 1} / {times_run}')
        print(f'\tStarting Training...')
        training_start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            num_correct = 0

            for images, labels in train_loader:
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

            print(f'\tEpoch [{epoch + 1}/{num_epochs}], Train set - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        training_end_time = time.time()
        total_training_time += (training_end_time - training_start_time)
        
        model.eval()
        running_loss = 0.0
        num_correct = 0

        print(f'\tStarting Testing...')
        testing_start_time = time.time()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)

                running_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
            num_correct += (predictions == labels).sum().item()

        print(f'\tTesting finished!')
        testing_end_time = time.time()
        total_testing_time += (testing_end_time - testing_start_time)
        test_loss = running_loss / len(test_loader.dataset)
        test_accuracy = num_correct / len(test_loader.dataset)
        print(f'\tTest set - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    
    
    print(f'Average train time: {total_training_time / times_run}')
    print(f'Average test time: {total_testing_time / times_run}')
