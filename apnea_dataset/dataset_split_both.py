import pandas as pd
import numpy as np
import os
import glob
import shutil
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Virtual Dataset Paths
VIRTUAL_SIGNAL_FOLDER = "bcg_synt_dataset"
VIRTUAL_WINDOW_FOLDER = "virtual_windows"
VIRTUAL_COLUMN = "bcg_raw"

# Real Dataset Paths
REAL_SIGNAL_FOLDER = "real_bcg_dataset"
REAL_WINDOW_FOLDER = "real_windows"
REAL_COLUMN = "bcg_raw"

# Output and Parameters
OUTPUT_FOLDER = "partitions_data" 
LOCAL_TEST_SPLIT = 0.2

OVERLAP_THRESHOLD = 500
MIN_LEN_PADDING = 1500

def get_file_pairs(signal_dir, window_dir):
    """
    Finds matching signal and window files.
    """
    signal_files = sorted(glob.glob(os.path.join(signal_dir, "*.csv")))
    valid_signals = []
    valid_windows = []

    if not signal_files:
        print(f"Warning: No CSV files found in {signal_dir}")
        return [], []

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

def process_data_into_memory(signal_files, window_files, target_column):
    """
    Loads data, normalizes per patient, and extracts segments.
    """
    segments = []
    labels = []
    max_len = 0

    print(f"Processing {len(signal_files)} files using column '{target_column}'...")

    for idx, (f_sig, f_win) in enumerate(zip(signal_files, window_files)):
        try:
            # Load Signal
            df_sig = pd.read_csv(f_sig, sep=',')
            if target_column not in df_sig.columns:
                print(f"Error: Column '{target_column}' not found. Skipping.")
                continue

            bcg_raw = df_sig[target_column].values.reshape(-1, 1)
            
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
                    
                    if length > max_len:
                        max_len = length

                    label = 1.0 if row['overlap_samples'] > OVERLAP_THRESHOLD else 0.0
                    segments.append(segment)
                    labels.append(label)

        except Exception as e:
            print(f"Error processing {f_sig}: {e}")

    return segments, labels, max_len

def pad_and_stack(segments, target_len):
    """
    Pads segments to target_len and stacks into (N, target_len).
    """
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
    """
    Returns a list of index arrays based on equal splits or ratios.
    """
    indices = np.random.permutation(num_samples)
    
    if ratios:
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
        return np.array_split(indices, num_partitions)

def main():
    # CLI Arguments
    parser = argparse.ArgumentParser(description="Merge and Split Datasets")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-partitions", type=int, help="Number of equal partitions")
    group.add_argument("--ratios", type=float, nargs="+", help="List of proportions (e.g. 0.5 0.5)")
    args = parser.parse_args()

    # Default to 3 equal parts if nothing specified
    if args.num_partitions is None and args.ratios is None:
        args.num_partitions = 3
        print("No args provided. Defaulting to 3 equal partitions.")

    # Clean output folder
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    # --- STEP 1: LOAD VIRTUAL DATA ---
    print("--- Loading VIRTUAL Data ---")
    v_sig_files, v_win_files = get_file_pairs(VIRTUAL_SIGNAL_FOLDER, VIRTUAL_WINDOW_FOLDER)
    v_segments, v_labels, v_max_len = process_data_into_memory(v_sig_files, v_win_files, VIRTUAL_COLUMN)

    # --- STEP 2: LOAD REAL DATA ---
    print("\n--- Loading REAL Data ---")
    r_sig_files, r_win_files = get_file_pairs(REAL_SIGNAL_FOLDER, REAL_WINDOW_FOLDER)
    r_segments, r_labels, r_max_len = process_data_into_memory(r_sig_files, r_win_files, REAL_COLUMN)

    # --- STEP 3: MERGE DATA ---
    all_segments = v_segments + r_segments
    all_labels = np.array(v_labels + r_labels)
    
    total_samples = len(all_segments)
    if total_samples == 0:
        print("No valid segments extracted.")
        return

    global_max_len = max(v_max_len, r_max_len) if total_samples > 0 else 0
    final_len = max(global_max_len, MIN_LEN_PADDING)
    
    print(f"\nTotal samples merged: {total_samples}")
    print(f"Padding all segments to length: {final_len}")

    X_all = pad_and_stack(all_segments, final_len)
    y_all = all_labels

    # --- STEP 4: PARTITION ---
    print(f"Distributing samples...")
    
    # Get indices based on CLI args
    idx_list = get_partition_indices(len(X_all), args.num_partitions, args.ratios)

    for i, indices in enumerate(idx_list):
        X_part = X_all[indices]
        y_part = y_all[indices]

        if len(X_part) > 0:
            # Local Train/Test Split
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
            print(f" -> Saved {filename}: Total={len(X_part)} (Train={len(X_train)}, Test={len(X_test)})")
        else:
            print(f" -> Warning: Partition {i+1} is empty!")

    print("\nGeneration completed.")

if __name__ == "__main__":
    main()
