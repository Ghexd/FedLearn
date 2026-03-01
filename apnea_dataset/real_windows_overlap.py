import csv
import random
import os
from typing import List, Tuple

# Fixed configuration
random.seed(10)

OUTPUT_FOLDER = "real_windows" 
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")

# Set mode (e.g., 50Hz or 1000Hz)
mode = "50Hz"

# Base parameters initialization
x0 = 0 
TARGET_COLUMN = "bcg_raw"

if mode == "1000Hz":
    freq = 1000
    window_length = 30 * freq       # 30 seconds at 1000Hz
    offset = 10 * freq
    random_step_min = 10 * freq
    random_step_max = 15 * freq
elif mode == "50Hz":
    freq = 50
    window_length = 30 * freq       # 30 seconds at 50Hz
    offset = 10 * freq
    random_step_min = 10 * freq
    random_step_max = 15 * freq
else:
    raise ValueError("mode must be '1000Hz' or '50Hz'")

# DATASET AND GT CONFIGURATION
DATASET_CONFIG = [
    {'file': './real_bcg_dataset/patient_lab_001.csv', 'apnea': [(63, 87), (150, 170), (250, 267), (330, 345)]},
    # {'file': './real_bcg_dataset/patient_lab_002.csv', 'apnea': [(65, 90), (245, 270)]},
    {'file': './real_bcg_dataset/patient_lab_003.csv', 'apnea': [(65, 90), (156, 180), (245, 270), (335, 360)]},
    # {'file': './real_bcg_dataset/patient_lab_004.csv', 'apnea': [(65, 88), (150, 180), (245, 270), (330, 360)]},
    # {'file': './real_bcg_dataset/patient_lab_005.csv', 'apnea': [(155, 180), (250, 270), (335, 360)]},
    # {'file': './real_bcg_dataset/patient_lab_006.csv', 'apnea': [(65, 85), (155, 180), (335, 360)]},
    {'file': './real_bcg_dataset/patient_lab_007.csv', 'apnea': [(63, 90), (152, 180), (242, 270), (332, 360)]},
    {'file': './real_bcg_dataset/patient_lab_008.csv', 'apnea': [(65, 90), (152, 180), (242, 268), (333, 350)]},
    # {'file': './real_bcg_dataset/patient_lab_009.csv', 'apnea': [(65, 90), (245, 270), (335, 350)]},
    {'file': './real_bcg_dataset/patient_lab_010.csv', 'apnea': [(65, 87), (153, 180), (240, 270), (332, 360)]},
]

def count_rows(csvfile: str, target_col: str):
    """
    Counts the number of data rows in the CSV file, ensuring the target column exists.
    Uses DictReader to handle headers automatically.
    """
    with open(csvfile, "r", newline="", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if the required column exists
        if target_col not in reader.fieldnames:
            raise ValueError(f"Column '{target_col}' not found in {csvfile}. Available columns: {reader.fieldnames}")
            
        # Count rows in the iterator
        return sum(1 for row in reader)

def make_random_windows(num_rows: int, x0: int, window_length: int, step_min: int, step_max: int):
    """Generates random sliding windows."""
    windows = []
    last_index = num_rows - 1
    x_prev = x0

    while True:
        # Current window
        windows.append((x_prev, x_prev + window_length))
        
        # Compute random step
        step = random.randint(step_min, step_max)
        x_next = x_prev + step
        
        # Check if the next window exceeds the end of the file
        if x_next + window_length > last_index:
            # Add a final window that reaches exactly the end
            if x_next < last_index:
                 windows.append((x_next, min(x_next + window_length, last_index)))
            break
        
        x_prev = x_next
        
    return windows

def overlap(windows, gt_list, offset):
    """
    Calculates overlap.
    :param windows: list of generated windows
    :param gt_list: list of ground truth intervals SCALED TO SAMPLES for the current file
    :param offset: offset in samples
    """
    results = []
    
    for s, e in windows:
        overlapping = 0
        for s_gt, e_gt in gt_list:
            
            # Compute intersection between current window and GT window + offset
            overlap_start = max(s, s_gt + offset)
            overlap_end   = min(e, e_gt)
            
            overlap_len = overlap_end - overlap_start
            
            # If there is a valid overlap
            if overlap_len > 0:
                overlapping = overlap_len
                break
                
        results.append(overlapping)
    
    return results

def save_windows(windows: List[Tuple[int,int]], overlapped: List[int], outpath: str):
    """Saves the window indices and overlap info to a CSV file."""
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["start_index", "end_index", "overlap_samples"])
        for (s, e), o in zip(windows, overlapped):
            writer.writerow([s, e, o])

def main():
    print(f"Processing {len(DATASET_CONFIG)} files configuration...")
    print(f"Target Column: {TARGET_COLUMN}")

    for entry in DATASET_CONFIG:
        input_path = entry['file']
        
        # Ground Truth in seconds -> to be converted to samples
        raw_gt = entry['apnea']
        
        # Convert GT to samples based on frequency
        gt_samples = [(int(start * freq), int(end * freq)) for start, end in raw_gt]

        filename = os.path.basename(input_path)
        name_root, _ = os.path.splitext(filename)
        output_filename = f"{name_root}_windows.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(f"Processing: {filename}")
        print(f"   -> GT (sec): {raw_gt}")
        # print(f"   -> GT (samples): {gt_samples}")

        if not os.path.exists(input_path):
            print(f"ERROR: File not found: {input_path}")
            continue

        try:
            # Count rows specifically checking for 'raw_bcg'
            num_rows = count_rows(input_path, TARGET_COLUMN)
        except Exception as e:
            print(f"ERROR reading {input_path}: {e}")
            continue
            
        # Generate windows based on the count
        windows = make_random_windows(num_rows, x0, window_length, random_step_min, random_step_max)
        
        # Compute overlap (Passing the specific GT for the current file)
        overlapped = overlap(windows, gt_samples, offset)
        
        # Save to file
        save_windows(windows, overlapped, output_path)
        print(f"   -> Generated {len(windows)} windows. Saved to {output_filename}")
    
    print("Processing completed.")

if __name__ == "__main__":
    main()