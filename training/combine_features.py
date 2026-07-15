import pandas as pd

# Load the features extracted
df = pd.read_csv('processed_data/final_features_dataset.csv')

# Remove features with extremely low variance (almost constant)
low_var_cols = df.select_dtypes(include=['float64', 'int']).var()[df.select_dtypes(include=['float64', 'int']).var() < 1e-5].index.tolist()
if low_var_cols:
    print("Dropping low-variance columns:", low_var_cols)
    df.drop(columns=low_var_cols, inplace=True)

df.to_csv('processed_data/final_features_dataset.csv', index=False)
print("✅ Combined features dataset saved as 'final_features_dataset.csv'")
print("Total samples:", len(df))