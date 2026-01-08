import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import os

DATASET_PATH = "./server_data.npz" 
TARGET_MODEL_PATH = "./final_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64  
NUM_CLASSES = 10 

NUM_SHADOW_MODELS = 5
SHADOW_EPOCHS = 20
ATTACK_EPOCHS = 40
LR = 0.001

# NOTE: The attacker has access to a dataset drawn from the same distribution 
# as the target model's training data (Shadow Data).

class Net(nn.Module):
    """
    CNN Architecture used for Target and Shadow Models.
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
    Per-Class Attack Model.
    Input: Logits or Probabilities from Target/Shadow Model.
    Output: Logits for Membership (Binary Classification).
    """
    def __init__(self, input_dim=10):
        super(AttackNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float() / 255.0
        self.images = (self.images - 0.5) / 0.5 
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def get_data_for_attack(model, loader, device):
    """
    Runs inference to collect inputs for the attack model.
    Returns:
        - probs: The probability vectors (softmax output)
        - predicted_labels: The class predicted by the model (argmax)
    """
    model.eval()
    probs_list = []
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            
            out = model(imgs)
            probs = F.softmax(out, dim=1) 
            probs_list.append(probs)
            
    probs_tensor = torch.cat(probs_list)
    predicted_labels = torch.argmax(probs_tensor, dim=1)
    
    return probs_tensor, predicted_labels

def prepare_main_splits():
    """
    Splits the 10k dataset into two worlds:
    1. Shadow World (5000 samples): Used to train shadow models and attack models.
    2. Target World (5000 samples): Used to evaluate the attack on the real target.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    print(f"Loading Dataset {DATASET_PATH}...")
    with np.load(DATASET_PATH) as data:
        # Assuming structure: 5000 train (members), 5000 test (non-members)
        tm_imgs = data["train_images"] 
        tm_lbls = data["train_labels"]
        tnm_imgs = data["test_images"] 
        tnm_lbls = data["test_labels"]

    all_imgs = np.concatenate([tm_imgs, tnm_imgs])
    all_lbls = np.concatenate([tm_lbls, tnm_lbls])
    
    # Split 50% Shadow / 50% Target Eval
    shadow_imgs, target_imgs, shadow_lbls, target_lbls = train_test_split(
        all_imgs, all_lbls, test_size=0.5, random_state=42
    )
    
    shadow_ds = CustomDataset(shadow_imgs, shadow_lbls)
    target_ds = CustomDataset(target_imgs, target_lbls)
    
    print(f"Data Splitting Complete:")
    print(f" -> Shadow World Pool: {len(shadow_ds)} samples")
    print(f" -> Target World Pool: {len(target_ds)} samples")
    
    return shadow_ds, target_ds

def create_shadow_dataset(full_shadow_ds):
    """
    From the Shadow World Pool creates a random split for a single shadow model:
    - 50% Train (Shadow Members)
    - 50% Test (Shadow Non-Members)
    """
    total_len = len(full_shadow_ds)
    indices = np.arange(total_len)
    np.random.shuffle(indices)
    
    split_point = total_len // 2
    train_indices = indices[:split_point] # Members
    test_indices = indices[split_point:]  # Non-Members
    
    train_ds = Subset(full_shadow_ds, train_indices)
    test_ds = Subset(full_shadow_ds, test_indices)
    
    return train_ds, test_ds


def main():
    print("===============================")
    print(" MIA Based on Shadow Models ")
    print("===============================")
    
    # Split data: shadow and target "world"
    shadow_pool_ds, target_pool_ds = prepare_main_splits()
    
    attack_X = [] 
    attack_Y_class = []
    attack_Y_membership = []

    print(f"\n[Phase 1] Training {NUM_SHADOW_MODELS} Shadow Models...")
    
    # Training multiple shadow models
    for i in range(NUM_SHADOW_MODELS):
        print(f" -> Training Shadow Model {i+1}/{NUM_SHADOW_MODELS}...")

        s_train_ds, s_out_ds = create_shadow_dataset(shadow_pool_ds)
        
        s_train_loader = DataLoader(s_train_ds, batch_size=BATCH_SIZE, shuffle=True)
        s_out_loader = DataLoader(s_out_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        shadow_model = Net().to(DEVICE)
        optimizer = optim.Adam(shadow_model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        
        shadow_model.train()
        for epoch in range(SHADOW_EPOCHS):
            for batch in s_train_loader:
                imgs, lbls = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = shadow_model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
        
        # Query Shadow Model to get Attack Data
        # IN (Members) = Label 1
        vecs_in, preds_in = get_data_for_attack(shadow_model, s_train_loader, DEVICE)
        labels_in = torch.ones(len(vecs_in)).to(DEVICE)
        
        # OUT (Non-Members) = Label 0
        vecs_out, preds_out = get_data_for_attack(shadow_model, s_out_loader, DEVICE)
        labels_out = torch.zeros(len(vecs_out)).to(DEVICE)
        
        # Aggregate
        attack_X.append(torch.cat([vecs_in, vecs_out]))
        attack_Y_class.append(torch.cat([preds_in, preds_out]))
        attack_Y_membership.append(torch.cat([labels_in, labels_out]))

    # Flatten aggregated data from all shadow models
    attack_X = torch.cat(attack_X)              # Input vectors
    attack_Y_class = torch.cat(attack_Y_class)  # Predicted Class (Routing)
    attack_Y_membership = torch.cat(attack_Y_membership).unsqueeze(1) # Target (1=Member, 0=NonMember)

    print(f"\n[Phase 2] Attack Dataset Generated.")
    print(f" -> Total Attack Samples: {len(attack_X)} (Aggregated from {NUM_SHADOW_MODELS} models)")

    # Train Attack Model
    print("\n[Phase 3] Training Per-Class Attack Models...")
    
    attack_models = {} 

    for class_id in range(NUM_CLASSES):
        
        indices = (attack_Y_class == class_id).nonzero(as_tuple=True)[0]
        
        # Handle class imbalance or empty classes
        if len(indices) < 50:
            print(f"  Warning: Class {class_id} has only {len(indices)} samples. Skipping/Unstable.")
            continue
        
        class_vecs = attack_X[indices]
        class_labels = attack_Y_membership[indices]
        
        class_ds = TensorDataset(class_vecs, class_labels)
        class_loader = DataLoader(class_ds, batch_size=32, shuffle=True)
        
        net = AttackNet().to(DEVICE)
        opt = optim.Adam(net.parameters(), lr=LR)
        
        crit = nn.BCEWithLogitsLoss()
        
        net.train()
        for epoch in range(ATTACK_EPOCHS):
            for v, l in class_loader:
                opt.zero_grad()
                logits = net(v)
                loss = crit(logits, l)
                loss.backward()
                opt.step()
        
        attack_models[class_id] = net

    print("  All attack models trained.")

    # Attacking target model
    print("\n[Phase 4] Attacking the Real Target Model...")
    
    target_model = Net().to(DEVICE)
    if not os.path.exists(TARGET_MODEL_PATH):
        print("Target model file not found.")
        return
    target_model.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=DEVICE))
    target_model.eval()
    
    # Split the Target World pool into Members (Train) and Non-Members (Test)
    t_indices = np.arange(len(target_pool_ds))
    t_split = len(target_pool_ds) // 2
    
    # First half = Members, Second half = Non-Members
    t_mem_ds = Subset(target_pool_ds, t_indices[:t_split])
    t_nonmem_ds = Subset(target_pool_ds, t_indices[t_split:])
    
    t_mem_loader = DataLoader(t_mem_ds, batch_size=BATCH_SIZE, shuffle=False)
    t_nonmem_loader = DataLoader(t_nonmem_ds, batch_size=BATCH_SIZE, shuffle=False)

    t_vec_in, t_pred_in = get_data_for_attack(target_model, t_mem_loader, DEVICE)
    t_vec_out, t_pred_out = get_data_for_attack(target_model, t_nonmem_loader, DEVICE)

    final_vecs = torch.cat([t_vec_in, t_vec_out])
    final_preds_cls = torch.cat([t_pred_in, t_pred_out]) # This determines routing
    final_true_mem = torch.cat([torch.ones(len(t_vec_in)), torch.zeros(len(t_vec_out))]).cpu().numpy()
    
    attack_scores = []

    with torch.no_grad():
        for i in range(len(final_vecs)):
            vec = final_vecs[i].unsqueeze(0)
            cls = final_preds_cls[i].item()
            
            if cls in attack_models:
                model = attack_models[cls]
                model.eval()
                logits = model(vec)
                prob = torch.sigmoid(logits).item() 
            else:
                prob = 0.5
                
            attack_scores.append(prob)

    attack_scores = np.array(attack_scores)

    # Metrics
    fpr, tpr, _ = roc_curve(final_true_mem, attack_scores)
    roc_auc = auc(fpr, tpr)
    advantage = np.max(np.abs(tpr - fpr))

    print("\n=====================")
    print(f"   Final Results  ")
    print("=======================")
    print(f"ROC-AUC Score:        {roc_auc:.4f}")
    print(f"Membership Advantage: {advantage:.4f}")
    print("-----------------------------------------")
    
    if advantage > 0.1:
        print(">> RISK DETECTED: Model leaks membership info.")
    else:
        print(">> LOW RISK: Attack close to random guessing.")

if __name__ == "__main__":
    main()