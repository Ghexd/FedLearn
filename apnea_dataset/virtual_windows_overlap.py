import csv
import random
import os
import glob
from typing import List, Tuple

# fixed seed for reproducibility
random.seed(10)

INPUT_FOLDER = "bcg_synt_dataset" 
OUTPUT_FOLDER = "virtual_windows" 

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}")

mode = "50Hz"

# Ground Truth
gt = [(10, 40), (80, 120), (170, 225), (290, 320),(475, 500), (570, 610), (705, 725), (965, 1000), (1080, 1110), (1155, 1180), (1210, 1235), (1280, 1340), (1390, 1425), (1500, 1535), (1580, 1610), (1670, 1700), (1750, 1785), (1805, 1840), (1900, 1930), (1950, 1980)]
x0 = 0 

if mode == "1000Hz":
    window_length = 30 * 1000       # 30 seconds al 1000Hz  
    offset = 10 * 1000              
    random_step_min = 10 * 1000     
    random_step_max = 15 * 1000     
    threshold = 10 * 1000           
    gt = [(start*1000, end*1000) for start, end in gt]

elif mode == "50Hz":
    window_length = 30 * 50         # 30 seconds at 50Hz
    offset = 10 * 50
    random_step_min = 10 * 50       
    random_step_max = 15 * 50       
    threshold = 10 * 50             
    gt = [(start*50, end*50) for start, end in gt]
    
else:
    raise ValueError("mode must be '1000Hz' or '50Hz'")            

def count_rows(tsvfile: str):
    with open(tsvfile, "r", newline="") as f:
        return sum(1 for row in f) - 1

def make_random_windows(num_rows: int, x0: int, window_length: int, step_min: int, step_max: int):
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

def overlap(windows):
    results = []
    
    # ensure integers and allow any iterable
    gt_list = [(int(a), int(b)) for a, b in gt]  

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
    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        
        writer.writerow(["start_index", "end_index", "overlap_samples"])
    
        for (s, e), o in zip(windows, overlapped):
            writer.writerow([s, e, o])

def main():

    input_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))
    
    if not input_files:
        print(f"No CSV files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(input_files)} files to process.")

    for input_path in input_files:

        filename = os.path.basename(input_path)

        name_root, _ = os.path.splitext(filename)
        output_filename = f"{name_root}_windows.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(f"Processing: {filename} -> {output_filename}")

        try:
            num_rows = count_rows(input_path)
        except Exception as e:
            print(f"ERROR reading {input_path}: {e}")
            continue
            
        # Generate windows
        windows = make_random_windows(num_rows, x0, window_length, random_step_min, random_step_max)
        
        # Compute overlap
        overlapped = overlap(windows)
        
        # Save to file
        save_windows(windows, overlapped, output_path)
    
    print("Processing completed.")

if __name__ == "__main__":
    main()
