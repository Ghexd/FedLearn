import os
import numpy as np
from torchvision import datasets

OUTPUT_FILENAME = "server_data.npz"
OUTPUT_PATH = os.path.join(os.getcwd(), OUTPUT_FILENAME)

def generate_audit_dataset(num_members=500, num_non_members=500):
    """
    Generate a dataset for member inference attack: 
    we take 50% of the data from the training set (member) 
    and 50% of the data from the testing set (non-member)
    """

    print("Loading FashionMNIST source data...")

    # Load the original Training Set (60 000 images)
    source_train = datasets.FashionMNIST(root="data", train=True, download=True)
    src_train_data = source_train.data.numpy().reshape(-1, 1, 28, 28)
    src_train_labels = source_train.targets.numpy()

    # Load the original Test Set (10 000 images)
    source_test = datasets.FashionMNIST(root="data", train=False, download=True)
    src_test_data = source_test.data.numpy().reshape(-1, 1, 28, 28)
    src_test_labels = source_test.targets.numpy()

    print(f"Source Loaded -> Train Pool: {len(src_train_data)}, Test Pool: {len(src_test_data)}")

    # Random Sampling (Shuffling)
    # Indices for Members
    if num_members > len(src_train_data):
        raise ValueError("Requested more members than available in the dataset.")
    idx_members = np.random.choice(len(src_train_data), num_members, replace=False)
    
    # Indices for Non-Members
    if num_non_members > len(src_test_data):
        raise ValueError("Requested more non-members than available in the dataset.")
    idx_non_members = np.random.choice(len(src_test_data), num_non_members, replace=False)

    # Data Extraction
    final_member_img = src_train_data[idx_members]
    final_member_lab = src_train_labels[idx_members]

    final_non_member_img = src_test_data[idx_non_members]
    final_non_member_lab = src_test_labels[idx_non_members]

    # Saving to disk
    print(f"\nSaving to {OUTPUT_PATH}...")
    np.savez(
        OUTPUT_PATH,
        # traim_image = members
        train_images=final_member_img,
        train_labels=final_member_lab,
        # test_images = non-members
        test_images=final_non_member_img,
        test_labels=final_non_member_lab,
    )

    print("-" * 40)
    print(f"SUCCESS! Dataset created.")
    print(f"MEMBERS (from Train Set):     {len(final_member_img)}")
    print(f"NON-MEMBERS (from Test Set):  {len(final_non_member_img)}")
    print(f"TOTAL AUDIT SAMPLES:          {len(final_member_img) + len(final_non_member_img)}")
    print("-" * 40)

if __name__ == "__main__":
    # Change these numbers for a larger or smaller dataset
    generate_audit_dataset(num_members=5000, num_non_members=5000)