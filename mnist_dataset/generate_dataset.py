import argparse
import os
import numpy as np
from torchvision import datasets

DATASET_DIRECTORY = "datasets"

def get_partition_indices(num_samples, num_partitions=None, ratios=None):

    indices = np.random.permutation(num_samples)
    
    if ratios:
        # Normalize ratios and calculate split points based on proportions
        ratios = np.array(ratios) / np.sum(ratios)
        partition_indices = []
        current_pos = 0
        for i, r in enumerate(ratios):
            start = current_pos
            if i == len(ratios) - 1:
                end = num_samples
            else:
                end = start + int(r * num_samples)
            partition_indices.append(indices[start:end])
            current_pos = end
        return partition_indices
    else:
        # Split into N equal integer parts
        return np.array_split(indices, num_partitions)

def save_dataset_to_disk(num_partitions=None, ratios=None):

    if not os.path.exists(DATASET_DIRECTORY):
        os.makedirs(DATASET_DIRECTORY)

    # Load training set (60 000 images)
    train_set = datasets.FashionMNIST(root="data", train=True, download=True)
    train_data = train_set.data.numpy().reshape(-1, 1, 28, 28)
    train_labels = train_set.targets.numpy()

    # Load test set (10 000 images)
    test_set = datasets.FashionMNIST(root="data", train=False, download=True)
    test_data = test_set.data.numpy().reshape(-1, 1, 28, 28)
    test_labels = test_set.targets.numpy()

    print(f"Dataset successfully loaded: Training pool={len(train_data)}, Testing pool={len(test_data)}")

    # Generate indices for both pools
    train_idx_list = get_partition_indices(len(train_data), num_partitions, ratios)
    test_idx_list = get_partition_indices(len(test_data), num_partitions, ratios)

    # Save the partitions to disk
    for i in range(len(train_idx_list)):
        t_idx = train_idx_list[i]
        s_idx = test_idx_list[i]

        p_train_img, p_train_lab = train_data[t_idx], train_labels[t_idx]
        p_test_img, p_test_lab = test_data[s_idx], test_labels[s_idx]

        file_path = f"./{DATASET_DIRECTORY}/fashionmnist_part_{i + 1}.npz"
        np.savez(
            file_path,
            train_images=p_train_img,
            train_labels=p_train_lab,
            test_images=p_test_img,
            test_labels=p_test_lab,
        )
        
        print(f"Partition {i + 1}:")
        print(f"  - Training samples (from 60k pool): {len(p_train_img)}")
        print(f"  - Testing samples  (from 10k pool): {len(p_test_img)}")
        print(f"  - Total partition size: {len(p_train_img) + len(p_test_img)}")
        print(f"  - File saved at: {file_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Fashion-MNIST into custom partitions")
    
    # Mutually exclusive group: choose either equal parts or specific ratios
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--num-partitions", 
        type=int, 
        help="Number of equal partitions to create"
    )
    group.add_argument(
        "--ratios", 
        type=float, 
        nargs="+", 
        help="List of proportions (e.g., 0.33 0.33 0.16 0.16)"
    )

    args = parser.parse_args()

    # default: 1/3, 1/3, 1/6, 1/6
    if args.num_partitions is None and args.ratios is None:
        target_ratios = [1/3, 1/3, 1/3]
        print("No arguments provided. Using default ratios: 1/3, 1/3, 1/6, 1/6")
        save_dataset_to_disk(ratios=target_ratios)
    else:
        save_dataset_to_disk(num_partitions=args.num_partitions, ratios=args.ratios)
