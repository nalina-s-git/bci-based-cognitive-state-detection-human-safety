import numpy as np
import pandas as pd
from scipy.signal import welch

# EEG frequency bands (Hz)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

FS = 250  # Sampling frequency (change to your device)

def bandpower(data, fs=FS, band=(0.5, 4)):
    """Compute band power using Welch method with dynamic segment length."""
    data = np.asarray(data, dtype=float)  # ensure numeric array
    data = data[~np.isnan(data)]          # remove NaNs
    
    if len(data) < 2:  # not enough data to compute spectrum
        return 0.0
    
    nperseg = min(len(data), fs*2)  # dynamic segment length
    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    if len(f) < 2:
        freq_res = 1.0  # fallback for very short data
    else:
        freq_res = f[1] - f[0]
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.sum(Pxx[idx_band]) * freq_res

# List of CSV files for persons 5,6,7
file_list = ['data/person5.csv', 'data/person6.csv', 'data/person7.csv']

combined_df = pd.DataFrame()

for idx, file in enumerate(file_list, start=5):
    # Load CSV
    df = pd.read_csv(file)
    
    # Identify EEG channels (exclude 'label' or 'state')
    eeg_channels = [col for col in df.columns if col not in ['label', 'state']]
    
    # Convert channels to numeric and remove rows with NaN
    df[eeg_channels] = df[eeg_channels].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=eeg_channels, inplace=True)
    
    # Compute band powers for each channel (full column)
    for band, (low, high) in BANDS.items():
        powers = []
        for channel in eeg_channels:
            data = df[channel].values
            powers.append(bandpower(data, band=(low, high)))
        df[band + '_power'] = np.mean(powers)  # average across channels
    
    # Compute mean, std, variance, diff across channels
    df['eeg_mean'] = df[eeg_channels].mean(axis=1)
    df['eeg_std'] = df[eeg_channels].std(axis=1)
    df['eeg_var'] = df[eeg_channels].var(axis=1)
    df['eeg_diff'] = df[eeg_channels].diff(axis=1).abs().sum(axis=1)
    
    # Add person label
    df['person'] = idx  # 5, 6, or 7
    
    # Append to combined dataset
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save combined features CSV
combined_df.to_csv('processed_data/combined_person5_6_7_features.csv', index=False)
print("Combined feature extraction complete.")
print("Saved as: processed_data/combined_person5_6_7_features.csv")