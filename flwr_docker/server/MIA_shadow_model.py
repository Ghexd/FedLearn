import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import Compose, Normalize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import os

DATASET_PATH = "./server_data.npz"
TARGET_MODEL_PATH = "./final_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

SHADOW_EPOCHS = 20    # Epochs to train the Shadow Model
ATTACK_EPOCHS = 30    # Epochs to train the Binary Attack Model
LR = 0.001

class Net(nn.Module):
    """
    Standard CNN architecture. 
    Used for both the Target Model (pre-trained) and the Shadow Model (trained from scratch).
    """
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

class AttackNet(nn.Module):
    """
    The Attack Model (as described in Shokri et al.).
    Input: Confidence vector (probability distribution) of size 10.
    Output: Single probability score (Member vs Non-Member).
    """
    def __init__(self, input_dim=10):
        super(AttackNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # Output [0, 1]
        return x

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        # Normalize images to [-1, 1] range
        self.images = torch.from_numpy(images).float() / 255.0
        self.images = (self.images - 0.5) / 0.5 
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}

# Utils
def get_probabilities(model, loader, device):
    """
    Runs the model and returns the softmax probability vectors.
    """
    model.eval()
    prob_list = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            out = model(imgs)
            probs = F.softmax(out, dim=1)
            prob_list.append(probs)
    return torch.cat(prob_list)

def prepare_attack_data():
    """
    Loads and splits samples
    """
    print(f"Loading data from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Dataset not found.")

    with np.load(DATASET_PATH) as data:
        target_members_imgs = data["train_images"]  
        target_members_lbls = data["train_labels"]
        target_nonmembers_imgs = data["test_images"] 
        target_nonmembers_lbls = data["test_labels"]

    # Split the data so the Shadow Model is trained on a disjoint set from the Target Model evaluation.
    
    # Split True Members
    tm_imgs_shadow, tm_imgs_eval, tm_lbls_shadow, tm_lbls_eval = train_test_split(
        target_members_imgs, target_members_lbls, test_size=0.5, random_state=42
    )

    # Split True Non-Members
    tnm_imgs_shadow, tnm_imgs_eval, tnm_lbls_shadow, tnm_lbls_eval = train_test_split(
        target_nonmembers_imgs, target_nonmembers_lbls, test_size=0.5, random_state=42
    )

    shadow_train_ds = CustomDataset(tm_imgs_shadow, tm_lbls_shadow) # Shadow Members
    shadow_out_ds = CustomDataset(tnm_imgs_shadow, tnm_lbls_shadow) # Shadow Non-Members
    
    target_member_ds = CustomDataset(tm_imgs_eval, tm_lbls_eval)      # Real Target Members
    target_nonmember_ds = CustomDataset(tnm_imgs_eval, tnm_lbls_eval) # Real Target Non-Members

    print(f"Data Splits Created:")
    print(f" -> Shadow Train (Members): {len(shadow_train_ds)}")
    print(f" -> Shadow Test (Non-Members): {len(shadow_out_ds)}")
    print(f" -> Target Evaluation (Members): {len(target_member_ds)}")
    print(f" -> Target Evaluation (Non-Members): {len(target_nonmember_ds)}")

    return shadow_train_ds, shadow_out_ds, target_member_ds, target_nonmember_ds


def main():
    print("--- SHADOW MODELING MEMBERSHIP INFERENCE ATTACK ---")
    
    s_train_ds, s_out_ds, t_mem_ds, t_nonmem_ds = prepare_attack_data()
    
    s_train_loader = DataLoader(s_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    s_out_loader = DataLoader(s_out_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Train shadow model
    print("\n[Phase 1] Training Shadow Model...")
    shadow_model = Net().to(DEVICE)
    optimizer = optim.Adam(shadow_model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    shadow_model.train()
    for epoch in range(SHADOW_EPOCHS):
        total_loss = 0
        for batch in s_train_loader:
            imgs, lbls = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = shadow_model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Generate attack training data
    print("\n[Phase 2] Generating Vectors for Attack Model training...")
    
    in_vectors = get_probabilities(shadow_model, s_train_loader, DEVICE)
    out_vectors = get_probabilities(shadow_model, s_out_loader, DEVICE)

    # Create labels: 1 for Members, 0 for Non-Members
    in_labels = torch.ones(len(in_vectors), 1).to(DEVICE)
    out_labels = torch.zeros(len(out_vectors), 1).to(DEVICE)

    attack_inputs = torch.cat([in_vectors, out_vectors])
    attack_targets = torch.cat([in_labels, out_labels])

    attack_ds = TensorDataset(attack_inputs, attack_targets)
    attack_loader = DataLoader(attack_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Train attack model
    print("\n[Phase 3] Training Attack Model (Binary Classifier)...")
    attack_model = AttackNet().to(DEVICE)
    att_optim = optim.Adam(attack_model.parameters(), lr=LR)
    att_criterion = nn.BCELoss()

    attack_model.train()
    for epoch in range(ATTACK_EPOCHS):
        for vec, lbl in attack_loader:
            vec, lbl = vec.to(DEVICE), lbl.to(DEVICE)
            att_optim.zero_grad()
            pred = attack_model(vec)
            loss = att_criterion(pred, lbl)
            loss.backward()
            att_optim.step()

    # Attack real target model
    print("\n[Phase 4] Attacking the Target Model...")
    
    target_model = Net().to(DEVICE)
    try:
        target_model.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: {TARGET_MODEL_PATH} not found.")
        return

    t_mem_loader = DataLoader(t_mem_ds, batch_size=BATCH_SIZE, shuffle=False)
    t_nonmem_loader = DataLoader(t_nonmem_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Get prediction vectors
    target_in_vectors = get_probabilities(target_model, t_mem_loader, DEVICE)
    target_out_vectors = get_probabilities(target_model, t_nonmem_loader, DEVICE)

    final_inputs = torch.cat([target_in_vectors, target_out_vectors])
    
    # Ground Truth: 1 for Members, 0 for Non-Members
    y_true = np.concatenate([np.ones(len(target_in_vectors)), np.zeros(len(target_out_vectors))])

    # Run Attack Model
    attack_model.eval()
    with torch.no_grad():
        y_scores = attack_model(final_inputs).cpu().numpy().flatten()

    # Metrics (ROC-AUC & Advantage)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Advantage = max(|TPR - FPR|)
    advantage = np.max(np.abs(tpr - fpr))

    print("-------------------------------------------------------")
    print(f"FINAL MIA RESULTS")
    print("-------------------------------------------------------")
    print(f"ROC-AUC Score:        {roc_auc:.4f}")
    print(f"Membership Advantage: {advantage:.4f}")
    print("-------------------------------------------------------")
    
    if advantage > 0.1:
        print("Interpretation: The model leaks information (Advantage > 0.1).")
    else:
        print("Interpretation: The model seems robust or the attack failed (Advantage <= 0.1).")

if __name__ == "__main__":
    main()