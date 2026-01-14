import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import butter, filtfilt

OUTPUT_FOLDER = "bcg_dataset"
REAL_DATA_FOLDER = "real_bcg_dataset"
N_PATIENTS = 100
DURATION = 400
FS = 50
TOTAL_SAMPLES = DURATION * FS

# filter to separate noise
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def get_low_freq_component(data, fs, cutoff=0.5):
    """Extracts only the slow oscillation (breathing/movement) from real data"""
    b, a = butter_lowpass(cutoff, fs, order=2)
    # filtfilt applies the filter forward and backward to avoid phase shift
    return filtfilt(b, a, data)

def load_real_noise_bank(folder_path):
    noise_bank = []
    if not os.path.exists(folder_path):
        print("Warning: Real data folder not found.")
        return []

    files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Loading {len(files)} real data files...")
    
    for f in files:
        try:
            df = pd.read_csv(f)
            signal = df['bcg_raw'].values 
            
            # Normalization
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
            noise_bank.append(signal)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return noise_bank

def get_real_noise_chunk(noise_bank, target_length, seed):
    if not noise_bank:
        return np.random.normal(0, 1, target_length)
    
    rng = np.random.RandomState(seed)
    noise_idx = seed % len(noise_bank)
    real_signal = noise_bank[noise_idx]
    
    # If the real file is too short, duplicate it
    if len(real_signal) < target_length:
        reps = int(np.ceil(target_length / len(real_signal)))
        real_signal = np.tile(real_signal, reps)
    
    # Random cut
    max_start = len(real_signal) - target_length
    start_idx = rng.randint(0, max_start) if max_start > 0 else 0
    return real_signal[start_idx : start_idx + target_length]

def smooth_transition(target_array, window_size=100):
    return np.convolve(target_array, np.ones(window_size)/window_size, mode='same')

def generate_single_patient(p_id, noise_bank):
    np.random.seed(p_id + 42) # Deterministic seed

    time = np.linspace(0, DURATION, TOTAL_SAMPLES)

    # Synthetic physiological parameters
    hr_freq = np.random.uniform(0.9, 1.5)
    resp_freq = np.random.uniform(0.2, 0.35)
    base_amp = np.random.uniform(1.0, 2.5)
    
    # Clean Signal Generation
    resp_wave = np.sin(2 * np.pi * resp_freq * time)
    modulation = (1 + 0.4 * resp_wave)
    heart_wave = np.sin(2 * np.pi * hr_freq * time) + 0.3 * np.sin(2 * np.pi * 2 * hr_freq * time)
    
    signal_normal = base_amp * modulation * heart_wave + (base_amp * 0.5 * resp_wave)
    signal_apnea = (base_amp * 0.3) * heart_wave 

    # Event Logic
    apnea_target = np.zeros_like(time)
    movement_spike = np.zeros_like(time)
    events = [60, 150, 240, 330]
    
    for start_t in events:
        idx_start = int(start_t * FS)
        idx_end = int((start_t + 30) * FS)
        if idx_end >= TOTAL_SAMPLES: idx_end = TOTAL_SAMPLES - 1
        
        apnea_target[idx_start + int(2.0*FS) : idx_end] = 1.0
        
        # Synthetic movement
        spike_samples = int(3.5 * FS)
        t_spike = np.linspace(-2, 2, spike_samples)
        spike_shape = -1.0 * np.exp(-t_spike**2) * np.sin(3 * t_spike)
        if idx_start + spike_samples < TOTAL_SAMPLES:
            movement_spike[idx_start : idx_start + spike_samples] += spike_shape * base_amp * 5.0

    alpha = smooth_transition(apnea_target, window_size=int(2.5 * FS))
    clean_signal = ((1 - alpha) * signal_normal) + (alpha * signal_apnea) + movement_spike
    
    # Retrieve a chunk of real signal
    real_chunk = get_real_noise_chunk(noise_bank, TOTAL_SAMPLES, seed=(p_id + 100))
    
    # Extract low-frequency drift (bed/body oscillation)
    drift_component = get_low_freq_component(real_chunk, FS, cutoff=0.8)
    
    # The remaining part is high-frequency noise
    texture_component = real_chunk - drift_component

    # Define scaling factors (how much to "dirty" the signal)
    # Drift Factor: how the signal will oscillate.
    drift_scale = np.random.uniform(2.0, 4.0) 
    
    # Noise Factor: how the signal will be "rough".
    noise_scale = np.random.uniform(0.5, 1.0)

    # Final sum
    final_bcg = clean_signal + (drift_component * drift_scale) + (texture_component * noise_scale)

    labels = (alpha > 0.5).astype(int)

    return pd.DataFrame({
        'Time': time,
        'BCG': final_bcg.astype(np.float32),
        'Label': labels.astype(np.int8)
    })

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    noise_bank = load_real_noise_bank(REAL_DATA_FOLDER)
    
    print(f"Hard Mode generation for {N_PATIENTS} patients...")
    for i in range(N_PATIENTS):
        df = generate_single_patient(i, noise_bank)
        filename = os.path.join(OUTPUT_FOLDER, f"patient_{i:03d}.csv")
        df.to_csv(filename, index=False)
        if i % 10 == 0: print(f"Patient {i} done.")

if __name__ == "__main__":
    main()
