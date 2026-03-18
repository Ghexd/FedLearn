import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, welch
import os

# Functions to extract parameters from real data (apnea and normal)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def analyze_real_bcg(csv_path, real_apnea_windows, fs=50):
    print(f"--- Real Signal Analysis: {csv_path} ---")
    
    try:
        df = pd.read_csv(csv_path, header=0)
        
        raw_signal = pd.to_numeric(df["bcg_raw"], errors='coerce').dropna().values
    except Exception as e:
        print(f"Reading file error: {e}")
        return None

    if len(raw_signal) == 0:
        print("Error: empty file or not numeric")
        return None

    # Filtering real signal
    clean_signal = butter_bandpass_filter(raw_signal, 0.3, 15.0, fs)
    
    # Normalization
    sig_mean = np.mean(clean_signal)
    sig_std = np.std(clean_signal)
    norm_signal = (clean_signal - sig_mean) / sig_std
    
    t = np.arange(len(norm_signal)) / fs

    # Mask creation
    # Normal mask (True where breathing is normal)
    is_normal_zone = np.ones_like(t, dtype=bool)
    # Apnea mask (True where there is apnea)
    is_apnea_zone = np.zeros_like(t, dtype=bool)

    for start, end in real_apnea_windows:
        mask = (t >= start) & (t <= end)
        is_normal_zone[mask] = False
        is_apnea_zone[mask] = True

    # Index
    normal_indices = np.where(is_normal_zone)[0]
    apnea_indices = np.where(is_apnea_zone)[0]
    
    if len(normal_indices) < fs * 5:
        print("Error: Too few 'normal' zones.")
        return None


    # Part A: Normal Breathing Analysis
    # ----------------------------------
    
    # HR and Template
    peaks, _ = find_peaks(norm_signal, height=0.5, distance=int(0.4*fs))
    valid_peaks = [p for p in peaks if is_normal_zone[p]]
    
    if len(valid_peaks) < 2:
        extracted_hr = 60.0 # Fallback
    else:
        rr_intervals = np.diff(valid_peaks) / fs
        extracted_hr = 60.0 / np.mean(rr_intervals)
    
    # Morphological template
    templates = []
    win_samples = int(0.6 * fs)
    pre_samples = int(0.2 * fs)
    
    for p in valid_peaks:
        if p > pre_samples and p < len(norm_signal) - win_samples:
            seg = norm_signal[p - pre_samples : p - pre_samples + win_samples]
            templates.append(seg)
    
    if len(templates) > 0:
        avg_template = np.mean(templates, axis=0)
        j_peak_val = avg_template[pre_samples]
        k_valley_val = np.min(avg_template[pre_samples:])
        k_j_ratio = k_valley_val / (j_peak_val + 1e-6)
    else:
        k_j_ratio = -0.5 # Default
        j_peak_val = 1.0

    # Breathing frequency
    peak_amps = norm_signal[valid_peaks]
    peak_times = t[valid_peaks]
    
    # Amplitude interpolation
    t_interp = np.arange(peak_times[0], peak_times[-1], 0.25)
    if len(t_interp) > 10:
        amp_env = np.interp(t_interp, peak_times, peak_amps)
        mod_depth = np.std(amp_env) / (np.mean(peak_amps) + 1e-6)
        
        f, psd = welch(amp_env, fs=4.0, nperseg=min(len(amp_env), 128))
        mask_f = (f >= 0.1) & (f <= 0.6)
        if np.any(mask_f):
            idx_max = np.argmax(psd[mask_f])
            extracted_resp_freq = f[mask_f][idx_max] * 60.0
        else:
            extracted_resp_freq = 15.0
    else:
        extracted_resp_freq = 15.0
        mod_depth = 0.15

    # Normal phase noise
    noise_residues = []
    for p in valid_peaks:
        if p > pre_samples and p < len(norm_signal) - win_samples:
            seg = norm_signal[p - pre_samples : p - pre_samples + win_samples]
            scale = seg[pre_samples] / (j_peak_val + 1e-6)
            residue = seg - (avg_template * scale)
            noise_residues.extend(residue)
    normal_noise_std = np.std(noise_residues) if len(noise_residues) > 0 else 0.1


    # Part B: Apnea Analysis
    # ------------------------
    
    if len(apnea_indices) > fs * 2:
        apnea_signal = norm_signal[apnea_indices]
        
        # How strong is the signal during breath-holding compared to normal?
        # RMS (Root Mean Square) to compare the energy
        rms_normal = np.sqrt(np.mean(norm_signal[normal_indices]**2))
        rms_apnea = np.sqrt(np.mean(apnea_signal**2))
        
        # Scale factor: if < 1 signal is reducing, if > 1 signal become stronger and noisier
        apnea_amp_scale = rms_apnea / (rms_normal + 1e-6)
        
        # Apnea noise
        apnea_noise_std = np.std(apnea_signal)
        
    else:
        # If the apnea data are too few we use default parameters
        print("Warning: Few real apnea data. Default parameters in use.")
        apnea_amp_scale = 0.7 
        apnea_noise_std = normal_noise_std * 0.8 

    params = {
        "hr_bpm": extracted_hr,
        "resp_rpm": extracted_resp_freq,
        "k_j_ratio": k_j_ratio,
        "mod_depth": mod_depth,
        "normal_noise": normal_noise_std,
        # Parameters retrieved from real data:
        "apnea_scale": apnea_amp_scale, 
        "apnea_noise": apnea_noise_std,
        "fs_source": fs
    }
    
    print("\nEXTRACTED PARAMETERS:")
    print(f"  HR: {params['hr_bpm']:.1f}, Resp: {params['resp_rpm']:.1f}")
    print(f"  Apnea Behavior (Scale): {params['apnea_scale']:.2f}x compared to normal")
    print(f"  Normal Noise: {params['normal_noise']:.3f}, Apnea Noise: {params['apnea_noise']:.3f}")
        
    return params

# Synthesis Functions 
# ---------------------- 

def synthesize_bcg(params, duration_sec, new_apnea_windows, out_fs=50):
    print(f"\n--- SIGNAL SYNTHESIS ({duration_sec}s) ---")
    
    t = np.linspace(0, duration_sec, int(duration_sec * out_fs))
    
    # Respiratory Drive (0 = Apnea, 1 = Normal)
    resp_drive = np.ones_like(t)
    trans_samples = int(1.5 * out_fs)
    
    for start, end in new_apnea_windows:
        idx_start = np.searchsorted(t, start)
        idx_end = np.searchsorted(t, end)
        if idx_start < len(t) and idx_end < len(t):
            resp_drive[idx_start : idx_end] = 0.0 # Complete apnea
            
            # Smoothing
            s_ramp = max(0, idx_start - trans_samples)
            resp_drive[s_ramp:idx_start] = np.linspace(1, 0, idx_start-s_ramp)
            
            e_ramp = min(len(t), idx_end + trans_samples)
            resp_drive[idx_end:e_ramp] = np.linspace(0, 1, e_ramp-idx_end)

    # Base respiration
    resp_freq_hz = params['resp_rpm'] / 60.0
    respiration_signal = np.sin(2 * np.pi * resp_freq_hz * t) * resp_drive

    # Beat Generation
    rr_mean = 60.0 / params['hr_bpm']
    beat_times = []
    curr_t = 0.5
    
    while curr_t < duration_sec:
        beat_times.append(curr_t)
        idx = min(np.searchsorted(t, curr_t), len(t)-1)
        
        # RSA only if there is respiratory drive
        rsa = (0.15 * params['mod_depth']) * respiration_signal[idx]
        jitter = np.random.normal(0, 0.03 * rr_mean)
        curr_t += rr_mean + rsa + jitter

    # Template
    kj_ratio = params['k_j_ratio']
    def get_beat_template(time_ax):
        sig = 1.0 * np.exp(-(time_ax**2) / (2 * 0.02**2)) # J
        sig += kj_ratio * np.exp(-((time_ax - 0.07)**2) / (2 * 0.03**2)) # K
        sig += 0.2 * np.exp(-((time_ax - 0.15)**2) / (2 * 0.04**2)) # L
        sig += -0.2 * np.exp(-((time_ax + 0.04)**2) / (2 * 0.015**2)) # I
        return sig

    # Signal Construction
    synth_signal = np.zeros_like(t)
    noise_vector = np.zeros_like(t)
    
    # Pre-calculation of noise vectors and scaling based on drive
    # If drive is high -> use normal_noise and scale=1
    # If drive is low -> use apnea_noise and scale=apnea_scale
    
    # Linearly interpolate parameters between normal state and apnea
    current_noise_lvl = resp_drive * params['normal_noise'] + (1 - resp_drive) * params['apnea_noise']
    current_sig_scale = resp_drive * 1.0 + (1 - resp_drive) * params['apnea_scale']
    
    # Time-variable noise generation
    base_noise = np.random.normal(0, 1, size=len(t))
    
    NOISE_REDUCTION_FACTOR = 0.4
    
    noise_vector = base_noise * current_noise_lvl * NOISE_REDUCTION_FACTOR    
    
    for bt in beat_times:
        idx = np.searchsorted(t, bt)
        if idx >= len(t): continue
        
        local_drive = resp_drive[idx]
        local_resp = respiration_signal[idx]
        
        # Base beat scale (normal vs learned apnea)
        base_scale = current_sig_scale[idx]
        
        # Respiratory AM modulation (only if not in apnea)
        am_mod = 1.0 + (params['mod_depth'] * 1.0 * local_resp)
        
        # If we are in deep apnea, am_mod tends to 1 (no modulation), 
        # but we apply the base_scale learned from real data.
        final_amp = base_scale * am_mod
        
        mask = (t >= bt - 0.25) & (t < bt + 0.5)
        t_local = t[mask] - bt
        if len(t_local) > 0:
            synth_signal[mask] += get_beat_template(t_local) * final_amp

    baseline = 0.2 * respiration_signal
    final_signal = synth_signal + noise_vector + baseline
    
    return t, final_signal, resp_drive

# Main
# ----------

def create_dummy_csv(filename):
    print(f"Creating dummy file: {filename} ...")
    fs = 50
    dur = 60
    t = np.linspace(0, dur, dur*fs)
    # Normal signal
    sig = np.sin(2*np.pi*1.1*t) * (1 + 0.3*np.sin(2*np.pi*0.25*t)) 
    # Fake apnea (20-30s): very small and silent signal
    mask = (t>20) & (t<30)
    sig[mask] = sig[mask] * 0.3 # Reduced signal
    
    # Noise: normal high, apnea low
    noise = np.random.normal(0, 0.1, len(t))
    noise[mask] = np.random.normal(0, 0.02, np.sum(mask)) # Less noise in apnea
    
    pd.DataFrame(sig + noise).to_csv(filename, header=False, index=False)


OUTPUT_DIR = "bcg_synt_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 50

if __name__ == "__main__":

    datasets = [
        {'file': './real_bcg_dataset/patient_lab_001.csv', 'apnea': [(63, 87), (150, 170), (250, 267), (330, 345)]},
        {'file': './real_bcg_dataset/patient_lab_002.csv', 'apnea': [(65, 90), (245, 270)]},
        {'file': './real_bcg_dataset/patient_lab_003.csv', 'apnea': [(65, 90), (156, 180), (245, 270), (335, 360)]},
        {'file': './real_bcg_dataset/patient_lab_004.csv', 'apnea': [(65, 88), (150, 180), (245, 270), (330, 360)]},
        {'file': './real_bcg_dataset/patient_lab_005.csv', 'apnea': [(155, 180), (250, 270), (335, 360)]},
        {'file': './real_bcg_dataset/patient_lab_006.csv', 'apnea': [(65, 85), (155, 180), (335, 360)]},
        {'file': './real_bcg_dataset/patient_lab_007.csv', 'apnea': [(63, 90), (152, 180), (242, 270), (332, 360)]},
        {'file': './real_bcg_dataset/patient_lab_008.csv', 'apnea': [(65, 90), (152, 180), (242, 268), (333, 350)]},
        {'file': './real_bcg_dataset/patient_lab_009.csv', 'apnea': [(65, 90), (245, 270), (335, 350)]},
        {'file': './real_bcg_dataset/patient_lab_010.csv', 'apnea': [(65, 87), (153, 180), (240, 270), (332, 360)]},
    ]

    # Synthesis Parameters
    # ------------------------
    
    target_duration_sec = 2000
    target_apnea_windows = [(10, 40), (80, 120), (170, 225), (290, 320),(475, 500), (570, 610), (705, 725), (965, 1000), (1080, 1110), (1155, 1180), (1210, 1235), (1280, 1340), (1390, 1425), (1500, 1535), (1580, 1610), (1670, 1700), (1750, 1785), (1805, 1840), (1900, 1930), (1950, 1980)]
    
    # Old parameters
    # target_duration_sec = 1200
    # target_apnea_windows = [(10, 40), (80, 120), (170, 225), (290, 320),(475, 500), (570, 610), (705, 725), (965, 1000), (1080, 1110), (1155, 1180)]

    example_file = "../real_bcg_dataset/patient_lab_010.csv"
    example_plot_data = None

    for item in datasets:

        csv_filename = item["file"]
        real_apnea_zones = item["apnea"]

        if not os.path.exists(csv_filename):
            print(f"[WARN] {csv_filename} not found -> creating dummy")
            create_dummy_csv(csv_filename)

        print(f"[INFO] Analyzing {csv_filename}")

        # Analysis
        params = analyze_real_bcg(
            csv_filename,
            real_apnea_zones,
            fs=FS
        )

        if not params:
            print(f"[SKIP] Analysis failed for {csv_filename}")
            continue

        # Synthesis
        t_synth, sig_synth, drive = synthesize_bcg(
            params,
            duration_sec=target_duration_sec,
            new_apnea_windows=target_apnea_windows,
            out_fs=FS
        )

        # CSV Saving
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        out_csv = os.path.join(OUTPUT_DIR, f"{base_name}_synthetic.csv")

        pd.DataFrame(sig_synth, columns=["bcg_raw"]).to_csv(
            out_csv,
            index=False
        )

        print(f"[OK] Saved {out_csv}")

        # Data for example plot
        if csv_filename == example_file:
            example_plot_data = {
                "csv_filename": csv_filename,
                "real_apnea_zones": real_apnea_zones,
                "params": params,
                "t_synth": t_synth,
                "sig_synth": sig_synth,
            }

    # Example plot
    # ----------------
    
    if example_plot_data:

        csv_filename = example_plot_data["csv_filename"]
        real_apnea_zones = example_plot_data["real_apnea_zones"]
        params = example_plot_data["params"]
        t_synth = example_plot_data["t_synth"]
        sig_synth = example_plot_data["sig_synth"]

        df = pd.read_csv(csv_filename)
        real_sig = pd.to_numeric(df["bcg_raw"], errors="coerce").dropna().values
        real_sig = butter_bandpass_filter(real_sig, 0.7, 10, FS)
        t_real = np.arange(len(real_sig)) / FS
        
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(
            t_real,
            real_sig,
            "k-",
            lw=0.8,
            alpha=0.8,
            label="Real Input"
        )

        for i, (s, e) in enumerate(real_apnea_zones):
            plt.axvspan(
                s,
                e,
                color="orange",
                alpha=0.3,
                label="Real Apnea" if i == 0 else None
            )

        plt.title(
            f"Analyzed Input ({os.path.basename(csv_filename)})\n"
            f"Apnea scale: {params['apnea_scale']:.2f}x"
        )
        plt.legend(loc="upper right")

        plt.subplot(2, 1, 2)
        plt.plot(
            t_synth,
            sig_synth,
            "b-",
            lw=0.8,
            label="Synthetic Output"
        )

        for i, (s, e) in enumerate(target_apnea_windows):
            plt.axvspan(
                s,
                e,
                color="red",
                alpha=0.2,
                label="New Apnea" if i == 0 else None
            )

        plt.title("Parametric Synthesis")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.show()
