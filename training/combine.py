import pandas as pd
from scipy import stats
import numpy as np

# -------------------------------
# Load Person 2,3,4
# -------------------------------
p2 = pd.read_csv('data/person2.csv')
p3 = pd.read_csv('data/person3.csv')
p4 = pd.read_csv('data/person4.csv')

df_234 = pd.concat([p2, p3, p4], ignore_index=True)

print("Before cleaning:", len(df_234))

# -------------------------------
# Clean (remove outliers)
# -------------------------------
feature_columns = ['EEG_Value']

z_scores = np.abs(stats.zscore(df_234[feature_columns]))
df_234_clean = df_234[(z_scores < 3).all(axis=1)]

print("After cleaning:", len(df_234_clean))

# -------------------------------
# Load your OLD combined dataset (5,6,7)
# -------------------------------
df_567 = pd.read_csv('processed_data/combined_dataset.csv')

print("Person 5,6,7 samples:", len(df_567))

# -------------------------------
# IMPORTANT: Match column names
# -------------------------------
# If needed, rename like this:
# df_567.rename(columns={'label': 'State'}, inplace=True)
# df_567.rename(columns={'eeg_mean': 'EEG_Value'}, inplace=True)

print("Columns 2,3,4:", df_234_clean.columns)
print("Columns 5,6,7:", df_567.columns)

# -------------------------------
# Combine ALL
# -------------------------------
df_all = pd.concat([df_234_clean, df_567], ignore_index=True)

print("FINAL dataset size:", len(df_all))

# Save final dataset
df_all.to_csv("processed_data/final_dataset.csv", index=False)

print("Final dataset saved successfully!")