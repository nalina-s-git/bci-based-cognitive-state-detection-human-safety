import numpy as np
import pandas as pd
from scipy.signal import welch

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

FS = 250
WINDOW_SIZE = 25
STEP_SIZE = 10

def bandpower(data, fs=FS, band=(0.5, 4)):
    data = np.array(data, dtype=float)
    if len(data) < 2:
        return 0.0
    nperseg = min(len(data), fs*2)
    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    freq_res = f[1] - f[0] if len(f) > 1 else 1.0
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.sum(Pxx[idx_band]) * freq_res

files = ['data/person2.csv', 'data/person3.csv', 'data/person4.csv']
all_features = []

for file in files:
    print(f"Loading: {file}")
    df = pd.read_csv(file)
    df['EEG_Value'] = pd.to_numeric(df['EEG_Value'], errors='coerce')
    df.dropna(subset=['EEG_Value'], inplace=True)

    for state in df['State'].unique():
        state_data = df[df['State'] == state]['EEG_Value'].values
        if len(state_data) < 10:
            continue
        start = 0
        while start + WINDOW_SIZE <= len(state_data):
            segment = state_data[start:start + WINDOW_SIZE]
            feature_row = {}
            for band, (low, high) in BANDS.items():
                feature_row[band + '_power'] = bandpower(segment, band=(low, high))
            feature_row['eeg_mean'] = np.mean(segment)
            feature_row['eeg_std'] = np.std(segment)
            feature_row['State'] = state
            all_features.append(feature_row)
            start += STEP_SIZE

features_df = pd.DataFrame(all_features)
features_df.to_csv('processed_data/features_234.csv', index=False)
print("✅ Feature extraction complete for Person 2,3,4!")
print("Saved as: processed_data/features_234.csv")