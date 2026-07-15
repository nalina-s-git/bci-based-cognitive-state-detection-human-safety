# fix_csv_state.py

import pandas as pd
import os

# Current project folder
folder_path = "."

# Dictionary to map file names to their correct state
file_state_map = {
    # Processed data
    'cleaned_dataset.csv': 'Normal',
    'combined_dataset.csv': 'Stress',
    'combined_features.csv': 'Fatigue',
    'eeg_data.csv': 'Distraction',
    'features_234.csv': 'Stress',
    'features_567.csv': 'Stress',
    'features_data.csv': 'Panic',
    'final_dataset.csv': 'Distraction',
    'final_features_dataset.csv': 'Distraction',
    'final_features_dataset_cleaned.csv': 'Normal',
    'prepared_dataset.csv': 'Fatigue',

    # Raw EEG data
    'person1_data.csv': 'Stress',
    'person2_data.csv': 'Panic',
    'person3_data.csv': 'Distraction',
    'person4_data.csv': 'Fatigue',
    'person5_data.csv': 'Stress',
    'person6_data.csv': 'Distraction',
    'person7_data.csv': 'Fatigue',
    'person8_data.csv': 'Panic',
    'person9_data.csv': 'Normal',
    'person10_data.csv': 'Stress'
}

# Loop through all files and fix the 'state' column
for filename, state_name in file_state_map.items():

    # Search for the file in possible folders
    possible_paths = [
        os.path.join(folder_path, "data", filename),
        os.path.join(folder_path, "processed_data", filename),
        os.path.join(folder_path, filename)
    ]

    file_path = None

    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break

    if file_path is None:
        print(f"⚠ File not found: {filename}")
        continue

    # Read CSV
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Add / Update state column
    df["state"] = state_name

    # Save changes
    df.to_csv(file_path, index=False)

    print(f"✅ Updated '{filename}' with state = '{state_name}'")