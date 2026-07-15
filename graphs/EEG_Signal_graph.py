import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('processed_data/final_features_dataset.csv')

# Remove spaces
df.columns = df.columns.str.strip()

# Use 'mean' as signal representation
plt.figure()
plt.plot(df['mean'][:200])

plt.title('EEG Signal Representation (Mean Feature)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.show()