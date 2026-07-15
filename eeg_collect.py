import serial
import csv
import time
import pandas as pd

# 🔁 Change COM port if needed
ser = serial.Serial('COM5', 115200, timeout=1)

# 🔹 Define your 5 cognitive states
states = ["normal", "fatigue", "stress", "panic", "distraction"]
samples_per_state = 100  # 100 samples per cognitive state

# ✅ Open CSV in WRITE mode (fresh file every run)
with open('processed_data/eeg_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # ✅ Add header
    writer.writerow(["eeg_value", "state"])

    print(f"Collecting {samples_per_state * len(states)} samples for 1 person...")

    time.sleep(2)  # ⏳ wait for ESP32 to stabilize

    for label in states:
        count = 0
        print(f"\nCollecting data for state: {label}")

        while count < samples_per_state:
            data = ser.readline().decode().strip()

            try:
                value = int(data)   # ✅ convert safely
                writer.writerow([value, label])
                count += 1
                print(f"Saved {count}: {value}")
            except ValueError:
                pass  # ignore invalid data

print("\n✅ 500 samples collected successfully for 1 person!")

# 🔹 Load CSV with pandas to check
df = pd.read_csv('processed_data/eeg_data.csv')

print("\nColumns in CSV:", list(df.columns))
print("\nFirst 5 samples:")
print(df.head())

for col in df.columns:
    print(f"\nElectrode/Column: {col}")
    print(df[col].head().to_list())