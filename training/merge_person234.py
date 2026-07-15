import pandas as pd
from scipy import stats
import numpy as np

p2 = pd.read_csv('data/person2_data.csv')
p3 = pd.read_csv('data/person3_data.csv')
p4 = pd.read_csv('data/person4_data.csv')

df_234 = pd.concat([p2, p3, p4], ignore_index=True)

print("Before cleaning:", len(df_234))
print("Columns:", df_234.columns)

# Only feature column
feature_columns = ['eeg_value']

# Remove outliers
z_scores = np.abs(stats.zscore(df_234[feature_columns]))

df_234_clean = df_234[(z_scores < 3).all(axis=1)]

print("After cleaning:", len(df_234_clean))

df_234_clean.to_csv("processed_data/merged_person234.csv", index=False)

print("✅ Cleaned dataset saved successfully!")
print("Saved as: processed_data/merged_person234.csv")