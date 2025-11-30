"""test: A Flower / NumPy app."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize
import time
import os

absolute_path = "./"

class CustomDataset(Dataset):
    
    def __init__(self, images, labels, transform=None):
        # Scale image data from [0, 255] to [0.0, 1.0] and convert to float tensors
        self.images = torch.from_numpy(images).float() / 255.0
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data_from_disk(path: str, batch_size: int):

    print("Loading Dataset")

    # Load data from .npz file
    with np.load(path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"]

    # Define the transformation to normalize the data to the range [-1, 1] (assuming dataset has been already scaled to [0,1])
    pytorch_transforms = Compose([Normalize((0.5,), (0.5,))])

    train_dataset = CustomDataset(
        train_images, train_labels, transform=pytorch_transforms
    )
    test_dataset = CustomDataset(
        test_images, test_labels, transform=pytorch_transforms
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, epochs, learning_rate, device):

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    
    elapsed_time = 0
    for epoch in range(epochs):

        start_time = time.time()

        running_loss = 0.0 
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        stop_time = time.time()
        elapsed_time += stop_time - start_time
        
        # write training time for each epoch on file
        try:
            train_file_path = os.path.join(absolute_path, "Train_time.txt")

            with open(train_file_path, "a") as file:
                file.write(f"Epoch {epoch+1}: {elapsed_time:.4f} seconds\n")
        except IOError as e:
            print(f"Error writing file: {e}")

        print(f"Epoch {epoch+1}/{epochs}, Avg loss: {running_loss/len(trainloader):.4f}")
    
    mean_elapsed_time = elapsed_time / epochs
    print(f"Train Mean time: {mean_elapsed_time}")

    # write average training time on file
    try:
        train_file_path = os.path.join(absolute_path, "Train_time.txt")

        with open(train_file_path, "a") as file:
            file.write(f"Mean train time: {mean_elapsed_time:.4f} seconds\n\n")
    except IOError as e:
        print(f"Error writing file: {e}")

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):

    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    elapsed_time = 0
    ind = 0
    with torch.no_grad():
        for batch in testloader:

            start_time = time.time()

            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            stop_time = time.time()
            elapsed_time += stop_time - start_time

            ind = ind + 1
            
            # write test time for each bach
            try:
                test_file_path = os.path.join(absolute_path, "Test_time.txt")

                with open(test_file_path, "a") as file:
                    file.write(f"Batch n.{ind}:{elapsed_time:.4f} seconds\n")
            except IOError as e:
                print(f"Error writing file: {e}")

    mean_elapsed_time = elapsed_time / len(testloader)
    print(f"Test Mean time: {mean_elapsed_time}")

    # write average test time
    try:
        test_file_path = os.path.join(absolute_path, "Test_time.txt")

        with open(test_file_path, "a") as file:
            file.write(f"Mean test time:{mean_elapsed_time:.4f} seconds\n\n")
    except IOError as e:
        print(f"Error writing file: {e}")
            
    accuracy = correct / len(testloader.dataset)
    
    loss = loss / len(testloader)

    print(f"Test loss: {loss:.4f}, Test Acc: {accuracy:.4f}")

    return loss, accuracy