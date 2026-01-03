import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize
from sklearn.metrics import roc_curve, auc
import os

DATASET_PATH = "./server_data.npz" 
MODEL_PATH = "./final_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

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

# Utility functions for attack 
def load_attack_data(path: str):
    
    print(f"Loading Dataset from {path}...")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    with np.load(path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"]

    pytorch_transforms = Compose([Normalize((0.5,), (0.5,))])

    # Members Dataset (those used for training)
    member_dataset = CustomDataset(train_images, train_labels, transform=pytorch_transforms)
    
    # Non-Members Dataset (those never seen by the model)
    non_member_dataset = CustomDataset(test_images, test_labels, transform=pytorch_transforms)

    # Shuffle=False to preserve the order to analyze sample losses
    member_loader = DataLoader(member_dataset, batch_size=BATCH_SIZE, shuffle=False)
    non_member_loader = DataLoader(non_member_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return member_loader, non_member_loader

def get_sample_losses(model, loader, device):
    """
    Calculates the loss for each individual sample in the loader.
    reduction='none' to obtain a vector of losses instead of the mean.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_losses = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            losses = criterion(outputs, labels)
            
            all_losses.extend(losses.cpu().numpy())
            
    return np.array(all_losses)

def perform_loss_based_mia(member_losses, non_member_losses):
    """
    Performs the loss-threshold based attack.
    Members are expected to have lower loss.
    """
    # Ground truth labels: 1 for Members, 0 for Non-Members
    true_labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    
    attack_scores = np.concatenate([-member_losses, -non_member_losses])
    
    fpr, tpr, thresholds = roc_curve(true_labels, attack_scores)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, fpr, tpr

def main():
    print(f"--- Membership Inference Attack (MIA) ---")
    print(f"Device: {DEVICE}")
    
    # Load the model
    print(f"Loading Model from {MODEL_PATH}...")
    model = Net().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Make sure you have completed the training first.")
        return

    # Load data (Members and Non-Members)
    try:
        member_loader, non_member_loader = load_attack_data(DATASET_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Compute Losses
    print("Computing losses for Members (Train Data)...")
    member_losses = get_sample_losses(model, member_loader, DEVICE)
    print(f" -> Mean Member Loss: {np.mean(member_losses):.4f}")

    print("Computing losses for Non-Members (Test Data)...")
    non_member_losses = get_sample_losses(model, non_member_loader, DEVICE)
    print(f" -> Mean Non-Member Loss: {np.mean(non_member_losses):.4f}")

    # Perform the attack
    print("\nPerforming Attack Analysis...")
    auc_score, fpr, tpr = perform_loss_based_mia(member_losses, non_member_losses)
    
    print("--------------------------------------------------")
    print(f"MIA Attack Result (ROC-AUC Score): {auc_score:.4f}")
    print("--------------------------------------------------")

    if auc_score <= 0.55:
        print("RESULT: Low privacy risk. The model behaves similarly on train and test data.")
    elif 0.55 < auc_score < 0.7:
        print("RESULT: Moderate privacy risk. Some memorization detected.")
    else:
        print("RESULT: HIGH privacy risk! The model has likely overfitted and memorized training data.")

    # Advantage = max(|TPR - FPR|)
    advantage = np.max(np.abs(tpr - fpr))
    print(f"Membership Advantage Score: {advantage:.4f}")

if __name__ == "__main__":
    main()