"""test: A Flower / NumPy app adapted for 1D BCG Signals."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import os

absolute_path = "./"

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset that handles 1D signal data.
    Expected input shape for signals: (N, Length) or (N, 1, Length)
    """
    def __init__(self, signals, labels):
        # Assuming signals are already normalized or reasonable floats. 
        self.signals = torch.from_numpy(signals).float()
        
        # Conv1d expects (Batch, Channel, Length). 
        # If input is (Batch, Length), we add the channel dimension.
        if self.signals.ndim == 2:
            self.signals = self.signals.unsqueeze(1)
            
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return {"signal": signal, "label": label}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 8, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.4) 
        self.conv2 = nn.Conv1d(8, 16, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.4)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dropout1(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.global_avg(x).view(x.size(0), -1)
        out = self.fc(x)
        return out


def load_data_from_disk(path: str, batch_size: int):

    print("Loading Dataset from:", path)
    
    with np.load(path) as data:
        train_x = data["train_images"] 
        train_y = data["train_labels"] 
        test_x = data["test_images"]   
        test_y = data["test_labels"]   

    train_dataset = CustomDataset(train_x, train_y)
    test_dataset = CustomDataset(test_x, test_y)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, epochs, learning_rate, device):

    net.to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    
    elapsed_time = 0
    for epoch in range(epochs):

        start_time = time.time()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch in trainloader:
            signals = batch["signal"].to(device)
            labels = batch["label"].to(device)
            
            labels = labels.view(-1, 1)

            optimizer.zero_grad()
            outputs = net(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy for binary classification
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        stop_time = time.time()
        train_time = stop_time - start_time
        elapsed_time += train_time
        avg_epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total

        try:
            train_file_path = os.path.join(absolute_path, "Train_info.txt")

            with open(train_file_path, "a") as file:
                file.write(f"Epoch {epoch+1}: Time {train_time:.4f}s, Loss {avg_epoch_loss:.4f}, Accuracy {epoch_acc:.4f}\n")
        except IOError as e:
            print(f"Error writing file: {e}")

        print(f"Epoch {epoch+1}/{epochs}, Avg loss: {avg_epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    
    mean_elapsed_time = elapsed_time / epochs
    print(f"Train Mean time: {mean_elapsed_time}")

    avg_trainloss = running_loss / len(trainloader)
    final_acc = correct / total

    try:
        train_file_path = os.path.join(absolute_path, "Train_info.txt")
        with open(train_file_path, "a") as file:
            file.write(f"Mean train time: {mean_elapsed_time:.4f} seconds, Mean Loss: {avg_trainloss:.4f}, Mean Accuracy: {final_acc:.4f}\n\n")
    except IOError as e:
        print(f"Error writing file: {e}")

    return avg_trainloss


def test(net, testloader, device):

    net.to(device)
    net.eval()
    criterion = nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    total = 0
    
    elapsed_time = 0
    ind = 0
    with torch.no_grad():
        for batch in testloader:

            start_time = time.time()

            signals = batch["signal"].to(device)
            labels = batch["label"].to(device)
            labels = labels.view(-1, 1) 

            outputs = net(signals)
            batch_loss = criterion(outputs, labels).item()
            loss += batch_loss
            
            probs = torch.sigmoid(outputs)
            
            # Threshold at 0.5
            predicted = (probs > 0.5).float()
            
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)

            correct += batch_correct
            total += batch_total
            
            batch_acc = batch_correct / batch_total

            stop_time = time.time()
            batch_time = stop_time - start_time
            elapsed_time += batch_time

            ind = ind + 1
            try:
                test_file_path = os.path.join(absolute_path, "Test_info.txt")
                with open(test_file_path, "a") as file:
                    file.write(f"Batch n.{ind}: {batch_time:.4f} seconds, Loss {batch_loss:.4f}, Accuracy {batch_acc:.4f}\n")
            except IOError as e:
                print(f"Error writing file: {e}")

    mean_elapsed_time = elapsed_time / len(testloader)
    print(f"Test Mean time: {mean_elapsed_time}")

    accuracy = correct / total
    loss = loss / len(testloader)

    try:
        test_file_path = os.path.join(absolute_path, "Test_info.txt")
        with open(test_file_path, "a") as file:
            file.write(f"Mean test time: {mean_elapsed_time:.4f} seconds, Mean Loss: {loss:.4f}, Mean Accuracy: {accuracy:.4f}\n\n")
    except IOError as e:
        print(f"Error writing file: {e}")
            
    print(f"Test loss: {loss:.4f}, Test Acc: {accuracy:.4f}")

    return loss, accuracy