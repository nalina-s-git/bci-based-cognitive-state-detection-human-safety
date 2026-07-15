import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("processed_data/cleaned_dataset.csv")

WINDOW_SIZE = 5

feature_rows = []

for i in range(len(df) - WINDOW_SIZE):
    window = df["eeg_value"].iloc[i:i+WINDOW_SIZE]

    # Features
    mean_val = window.mean()
    var_val = window.var()
    std_val = window.std()
    min_val = window.min()
    max_val = window.max()
    range_val = max_val - min_val
    diff_val = window.iloc[-1] - window.iloc[0]

    # Label
    label = df["state"].iloc[i + WINDOW_SIZE - 1]

    feature_rows.append([
        mean_val, var_val, std_val,
        min_val, max_val, range_val,
        diff_val, label
    ])

# Create DataFrame
feature_df = pd.DataFrame(feature_rows, columns=[
    "Mean", "Variance", "Std_Dev",
    "Min", "Max", "Range",
    "Difference", "State"
])

# Shuffle (important)
feature_df = feature_df.sample(frac=1).reset_index(drop=True)

# Save
feature_df.to_csv("processed_data/features_data.csv", index=False)

print("✅ Professional feature extraction completed!")
print("Shape:", feature_df.shape)