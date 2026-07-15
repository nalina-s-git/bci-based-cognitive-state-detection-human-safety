import pandas as pd
import numpy as np
import serial  # For ESP32 serial data
import joblib
import time
from collections import deque

# -----------------------------
# Parameters
WINDOW_SIZE = 50    # Same as training
STEP_SIZE = 25      # Overlap
SERIAL_PORT = '/dev/ttyUSB0'  # Change to your ESP32 port
BAUD_RATE = 115200

# -----------------------------
# Load trained model, scaler, label encoder
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# -----------------------------
# Real-time data buffer
data_buffer = deque(maxlen=WINDOW_SIZE)

# -----------------------------
# Function to extract features from current window
def extract_features(window_df):
    if window_df.empty:
        return None
    features = {
        'eeg_mean': window_df.mean().mean(),
        'eeg_std': window_df.stack().std(),
        'eeg_var': window_df.var().mean(),
        'eeg_diff': np.mean(np.diff(window_df, axis=0))
    }
    return pd.DataFrame([features])

# -----------------------------
# Connect to ESP32
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to ESP32 on {SERIAL_PORT}")
except Exception as e:
    print("Error connecting to ESP32:", e)
    exit()

# -----------------------------
# Main real-time loop
try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue
        
        # Assume ESP32 sends CSV-like numeric values: 12,34,56,78
        try:
            values = [float(x) for x in line.split(',')]
            data_buffer.append(values)
        except ValueError:
            continue  # Skip invalid lines
        
        # When we have enough data for a window
        if len(data_buffer) == WINDOW_SIZE:
            window_df = pd.DataFrame(data_buffer)
            features_df = extract_features(window_df)
            
            # Scale & predict
            X_scaled = scaler.transform(features_df)
            pred_encoded = svm_model.predict(X_scaled)
            pred_state = le.inverse_transform(pred_encoded)[0]
            
            # Print / trigger alert
            print(f"Predicted State: {pred_state}")
            
            # Slide the window
            for _ in range(STEP_SIZE):
                if data_buffer:
                    data_buffer.popleft()

except KeyboardInterrupt:
    print("Stopping real-time demo...")
finally:
    ser.close()