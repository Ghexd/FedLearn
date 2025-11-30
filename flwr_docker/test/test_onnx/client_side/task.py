import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize

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

def load_data_from_disk(path: str, batch_size: int):

    print("Loading Dataset")

    # Load data from .npz file
    with np.load(path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"]

    # Define the transformation to normalize the data to the range [-1, 1]
    # This assumes the CustomDataset has already scaled the data to [0, 1]
    pytorch_transforms = Compose([Normalize((0.5,), (0.5,))])

    # Create PyTorch custom dataset
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

    for epoch in range(epochs):

        running_loss = 0.0 
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Avg loss: {running_loss/len(trainloader):.4f}")

    # Calculate and return the average loss of the final epoch
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):

    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    # The average loss over the entire test set
    loss = loss / len(testloader)

    print(f"Test loss: {loss:.4f}, Test Acc: {accuracy:.4f}")

    return loss, accuracy