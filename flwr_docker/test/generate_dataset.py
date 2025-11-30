import argparse
import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

DATASET_DIRECTORY = "datasets"


def save_dataset_to_disk(num_partitions: int):
    """
    Questa funzione scarica il dataset Fashion-MNIST usando torchvision
    e genera N partizioni in formato NumPy.
    """
    # Create destination folder if it not exists
    if not os.path.exists(DATASET_DIRECTORY):
        os.makedirs(DATASET_DIRECTORY)

    # Download training set WITHOUT ToTensor transform
    full_trainset = datasets.FashionMNIST(
        root="data", train=True, download=True
    )

    # Extract raw data and labels as numpy arrays
    data = full_trainset.data.numpy().reshape(-1, 1, 28, 28) # No division by 255.0 here
    labels = full_trainset.targets.numpy()

    # Mix dataset index
    num_images = len(full_trainset)
    shuffled_indices = np.random.permutation(num_images)

    # Split index into required partition
    partition_indices = np.array_split(shuffled_indices, num_partitions)

    for i in range(num_partitions):
        # Get partition index 
        current_indices = partition_indices[i]

        # Split partition in train set and test set (80/20)
        train_idx, test_idx = train_test_split(
            current_indices, test_size=0.2, random_state=42
        )

        train_images, train_labels = data[train_idx], labels[train_idx]
        test_images, test_labels = data[test_idx], labels[test_idx]
        
        # save partition
        file_path = f"./{DATASET_DIRECTORY}/fashionmnist_part_{i + 1}.npz"
        np.savez(
            file_path,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
        )
        print(f"Scritto: {file_path}")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Salva le partizioni del dataset Fashion-MNIST su disco"
    )

    # Optional argument for partion number
    parser.add_argument(
        "--num-supernodes",
        type=int,
        nargs="?",
        default=2,
        help="Numero di partizioni da creare (default: 2)",
    )

    # Parse arguments
    args = parser.parse_args()

    save_dataset_to_disk(args.num_supernodes)