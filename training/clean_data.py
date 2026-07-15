import pandas as pd

# Load combined dataset
df = pd.read_csv("processed_data/combined_dataset.csv")

# Keep only valid EEG range
df_clean = df[(df["EEG_Value"] >= 20) & (df["EEG_Value"] <= 120)]

# Save cleaned dataset
df_clean.to_csv("processed_data/cleaned_dataset.csv", index=False)

print("Outliers removed successfully!")
print("Original size:", len(df))
print("Cleaned size:", len(df_clean))