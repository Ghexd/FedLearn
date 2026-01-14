import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shutil

SIGNAL_FOLDER = "bcg_dataset"
WINDOW_FOLDER = "windows"
OUTPUT_FOLDER = "partitions_data" 
N_PARTITIONS = 4                
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
    """
    Loads all data, normalizes per patient, and extracts segments.
    Returns a list of segments (numpy arrays) and a list of labels.
    """
    all_segments = []
    all_labels = []
    max_len_found = 0

    print("Starting data processing...")

    for idx, (f_sig, f_win) in enumerate(zip(signal_files, window_files)):
        # Load Signal and Normalize
        try:
            df_sig = pd.read_csv(f_sig, sep=',')
            bcg_raw = df_sig['BCG'].values.reshape(-1, 1)
            
            scaler = StandardScaler()
            signal_norm = scaler.fit_transform(bcg_raw).flatten() # Flatten to obtain a 1D array
            
            # Load Windows
            df_win = pd.read_csv(f_win, sep=',')
            
            for _, row in df_win.iterrows():
                start = int(row['start_index'])
                end = int(row['end_index'])
                
                # Index validity checks
                if start >= 0 and end <= len(signal_norm) and start < end:
                    segment = signal_norm[start:end]
                    length = len(segment)
                    
                    if length > max_len_found:
                        max_len_found = length

                    # Label logic
                    label = 1.0 if row['overlap_samples'] > OVERLAP_THRESHOLD else 0.0
                    
                    all_segments.append(segment)
                    all_labels.append(label)

        except Exception as e:
            print(f"Error processing {f_sig}: {e}")

    print(f"Total extracted samples: {len(all_segments)}")
    print(f"Maximum length found: {max_len_found}")
    
    return all_segments, np.array(all_labels), max_len_found

def pad_and_stack(segments, target_len):
    """
    Applies zero-padding to all segments to reach target_len
    and stacks them into a single numpy array (N, target_len).
    """
    n_samples = len(segments)
    # Create a zero matrix (N, Length)
    X_padded = np.zeros((n_samples, target_len), dtype=np.float32)
    
    for i, seg in enumerate(segments):
        length = len(seg)
        if length > target_len:
            # If for it is longer truncate
            X_padded[i, :] = seg[:target_len]
        else:
            # Copy the segment at the beginning (padding at the end)
            X_padded[i, :length] = seg
            
    return X_padded

def main():
    
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    # Retrieve Files
    files_signal, files_window = get_file_pairs(SIGNAL_FOLDER, WINDOW_FOLDER)
    if not files_signal:
        print("No files found. Check the paths.")
        return

    segments, labels, max_len_found = process_data_into_memory(files_signal, files_window)
    
    if len(segments) == 0:
        print("No valid segments extracted.")
        return

    final_len = max(max_len_found, MIN_LEN_PADDING)
    print(f"Padding all segments to length: {final_len}")

    # Padding and conversion to a single NumPy array
    X_all = pad_and_stack(segments, final_len)
    y_all = labels

    # Shuffle
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]

    # Split the data into N equal parts
    chunk_size = len(X_all) // N_PARTITIONS
    remainder = len(X_all) % N_PARTITIONS
    
    start_idx = 0
    
    print(f"\nDistributing {len(X_all)} samples into {N_PARTITIONS} partitions...")

    for i in range(N_PARTITIONS):
        # adds 1 sample to the first nodes if not perfectly divisible
        count = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + count
        
        X_part = X_all[start_idx:end_idx]
        y_part = y_all[start_idx:end_idx]
        
        start_idx = end_idx

        # Local Train/Test Split for this node
        if len(X_part) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X_part, y_part, test_size=LOCAL_TEST_SPLIT, random_state=42
            )
            
            # Save to disk (.npz)
            filename = os.path.join(OUTPUT_FOLDER, f"partition_{i}.npz")
            np.savez(
                filename, 
                train_images=X_train, 
                train_labels=y_train, 
                test_images=X_test, 
                test_labels=y_test
            )
            print(f" -> Saved {filename}: Train={len(X_train)}, Test={len(X_test)}")
        else:
            print(f" -> Warning: Partition {i} is empty!")

    print("\nGeneration completed.")

if __name__ == "__main__":
    main()
