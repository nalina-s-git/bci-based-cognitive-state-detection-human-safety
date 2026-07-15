import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('processed_data/final_features_dataset.csv')

# Remove spaces (safe)
df.columns = df.columns.str.strip()

plt.figure()

# ✅ Use YOUR real features
plt.plot(df['mean'][:200], label='Mean')
plt.plot(df['std'][:200], label='Std Dev')
plt.plot(df['var'][:200], label='Variance')
plt.plot(df['skew'][:200], label='Skewness')
plt.plot(df['kurt'][:200], label='Kurtosis')

plt.title('EEG Feature Variation (Statistical Features)')
plt.xlabel('Samples')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()