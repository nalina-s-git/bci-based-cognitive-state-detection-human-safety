import pandas as pd

# Load final dataset
df = pd.read_csv("processed_data/final_dataset.csv")

print("Class distribution:\n")
print(df["state"].value_counts())