import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import models, transforms, datasets
import numpy as np
import torch.optim as optim
import torch.nn as nn
import time

#@author:RhinoCoder

def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total




def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    cifar100_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    N = 1000
    initial_subset, _ = random_split(cifar100_data, [N, len(cifar100_data) - N])
    initial_loader = DataLoader(initial_subset, batch_size=32, shuffle=True)

    resnet18_pretrained = models.resnet18(weights="IMAGENET1K_V1")
    resnet18_pretrained.fc = nn.Linear(resnet18_pretrained.fc.in_features,
                                       100)
    resnet18_pretrained = resnet18_pretrained.cuda()

    criterion_pretrained = nn.CrossEntropyLoss()
    optimizer_pretrained = optim.SGD(resnet18_pretrained.parameters(), lr=0.01, momentum=0.9)

    subset_size = N
    max_epochs = 100
    increment_factor = 2
    start_time = time.time()

    print("Starting incremental training...")
    for epoch in range(max_epochs):
        running_loss = 0.0
        resnet18_pretrained.train()
        for inputs, labels in initial_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer_pretrained.zero_grad()
            outputs = resnet18_pretrained(inputs)
            loss = criterion_pretrained(outputs, labels)
            loss.backward()
            optimizer_pretrained.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(initial_loader)}")

        if (epoch + 1) % 10 == 0:
            subset_size = min(subset_size * increment_factor, len(cifar100_data))
            if subset_size == len(cifar100_data):
                break

            new_subset, _ = random_split(cifar100_data, [subset_size, len(cifar100_data) - subset_size])
            initial_loader = DataLoader(new_subset, batch_size=32, shuffle=True)

    incremental_training_time = time.time() - start_time

    val_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    print("Evaluating the models...")
    accuracy_incremental = calculate_accuracy(resnet18_pretrained, val_loader)
    print(f"Incremental Training Accuracy: {accuracy_incremental * 100:.2f}%")
    print(f"Incremental Training Time: {incremental_training_time:.2f} seconds")

if __name__ == '__main__':
    main()