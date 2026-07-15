import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Sample EEG signal (replace this with your real EEG data)
fs = 256  # Sampling frequency (Hz)
t = np.linspace(0, 1, fs)

# Simulated EEG signal (mix of waves)
eeg_signal = (np.sin(2 * np.pi * 2 * t) +   # Delta
              np.sin(2 * np.pi * 6 * t) +   # Theta
              np.sin(2 * np.pi * 10 * t) +  # Alpha
              np.sin(2 * np.pi * 20 * t))   # Beta

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

# Extract brain waves
delta = bandpass_filter(eeg_signal, 0.5, 4, fs)
theta = bandpass_filter(eeg_signal, 4, 7, fs)
alpha = bandpass_filter(eeg_signal, 8, 12, fs)
beta  = bandpass_filter(eeg_signal, 13, 30, fs)

# Plot
plt.figure(figsize=(10,6))

plt.plot(t, delta, label='Delta (0.5-4 Hz)')
plt.plot(t, theta, label='Theta (4-7 Hz)')
plt.plot(t, alpha, label='Alpha (8-12 Hz)')
plt.plot(t, beta, label='Beta (13-30 Hz)')

plt.title("EEG Brain Wave Decomposition into Frequency Bands")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)

plt.show()