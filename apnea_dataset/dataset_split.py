import pandas as pd
import numpy as np
import os
import glob
import shutil
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuration
SIGNAL_FOLDER = "bcg_dataset"
WINDOW_FOLDER = "windows"
OUTPUT_FOLDER = "partitions_data" 
LOCAL_TEST_SPLIT = 0.2

OVERLAP_THRESHOLD = 500
MIN_LEN_PADDING = 1500

def get_file_pairs(signal_dir, window_dir):
    signal_files = sorted(glob.glob(os.path.join(signal_dir, "patient_*.csv")))
    valid_signals = []
    valid_windows = []

    for s_file in signal_files:
        basename = os.path.basename(s_file)
        name_root, _ = os.path.splitext(basename)
        w_filename = f"{name_root}_windows.csv"
        w_file = os.path.join(window_dir, w_filename)
        
        if os.path.exists(w_file):
            valid_signals.append(s_file)
            valid_windows.append(w_file)
        else:
            print(f"Skipping {basename}: window file not found.")
            
    return valid_signals, valid_windows

def process_data_into_memory(signal_files, window_files):
    all_segments = []
    all_labels = []
    max_len_found = 0

    print("Starting data processing...")

    for idx, (f_sig, f_win) in enumerate(zip(signal_files, window_files)):
        try:
            # Load Signal
            df_sig = pd.read_csv(f_sig, sep=',')
            bcg_raw = df_sig['bcg_synth'].values.reshape(-1, 1)
            
            # Normalize
            scaler = StandardScaler()
            signal_norm = scaler.fit_transform(bcg_raw).flatten()
            
            # Load Windows
            df_win = pd.read_csv(f_win, sep=',')
            
            for _, row in df_win.iterrows():
                start = int(row['start_index'])
                end = int(row['end_index'])
                
                if start >= 0 and end <= len(signal_norm) and start < end:
                    segment = signal_norm[start:end]
                    length = len(segment)
                    
                    if length > max_len_found:
                        max_len_found = length

                    label = 1.0 if row['overlap_samples'] > OVERLAP_THRESHOLD else 0.0
                    all_segments.append(segment)
                    all_labels.append(label)

        except Exception as e:
            print(f"Error processing {f_sig}: {e}")

    return all_segments, np.array(all_labels), max_len_found

def pad_and_stack(segments, target_len):
    n_samples = len(segments)
    X_padded = np.zeros((n_samples, target_len), dtype=np.float32)
    
    for i, seg in enumerate(segments):
        length = len(seg)
        if length > target_len:
            X_padded[i, :] = seg[:target_len]
        else:
            X_padded[i, :length] = seg
            
    return X_padded

def get_partition_indices(num_samples, num_partitions=None, ratios=None):
    indices = np.random.permutation(num_samples)
    
    if ratios:
        # Normalize ratios
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
        # Equal splits
        return np.array_split(indices, num_partitions)

def main():
    # CLI Arguments
    parser = argparse.ArgumentParser(description="Split Dataset into custom partitions")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-partitions", type=int, help="Number of equal partitions")
    group.add_argument("--ratios", type=float, nargs="+", help="List of proportions (e.g. 0.5 0.5)")
    args = parser.parse_args()

    # Default logic if no args provided (default to 4 equal parts)
    if args.num_partitions is None and args.ratios is None:
        args.num_partitions = 4
        print("No split args provided. Defaulting to 4 equal partitions.")

    # Setup output directory
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    # Load Data
    files_signal, files_window = get_file_pairs(SIGNAL_FOLDER, WINDOW_FOLDER)
    if not files_signal:
        print("No files found.")
        return

    segments, labels, max_len_found = process_data_into_memory(files_signal, files_window)
    
    if len(segments) == 0:
        print("No valid segments extracted.")
        return

    final_len = max(max_len_found, MIN_LEN_PADDING)
    print(f"Padding to length: {final_len}")

    X_all = pad_and_stack(segments, final_len)
    y_all = labels

    print(f"Total samples: {len(X_all)}")

    # Get Partition Indices
    idx_list = get_partition_indices(len(X_all), args.num_partitions, args.ratios)

    # Save partitions
    for i, indices in enumerate(idx_list):
        X_part = X_all[indices]
        y_part = y_all[indices]
        
        if len(X_part) > 0:
            # Local Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_part, y_part, test_size=LOCAL_TEST_SPLIT, random_state=42
            )
            
            filename = os.path.join(OUTPUT_FOLDER, f"partition_{i+1}.npz")
            np.savez(
                filename, 
                train_images=X_train, 
                train_labels=y_train, 
                test_images=X_test, 
                test_labels=y_test
            )
            print(f"Saved partition {i+1}: Total={len(X_part)} (Train={len(X_train)}, Test={len(X_test)})")
        else:
            print(f"Warning: Partition {i+1} is empty!")

if __name__ == "__main__":
    main()
