import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.utils import resample

# 1️⃣ Load feature data
df = pd.read_csv("processed_data/features_data.csv")  # your extracted features

# 1.5️⃣ Add difference of EEG_Value as a new feature
df["diff"] = df["difference"].diff().fillna(0)

# 2️⃣ Separate features and labels
X = df.drop("state", axis=1)  # all features including 'diff'
y = df["state"]               # labels

# 3️⃣ Normalize features (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Combine back into a DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled['state'] = y

# 5️⃣ Balance dataset using upsampling
balanced_df = pd.DataFrame()
classes = df_scaled['state'].unique()
max_size = df_scaled['state'].value_counts().max()

for cls in classes:
    cls_df = df_scaled[df_scaled['state'] == cls]
    cls_upsampled = resample(cls_df, 
                             replace=True,     
                             n_samples=max_size, 
                             random_state=42)
    balanced_df = pd.concat([balanced_df, cls_upsampled])

# 6️⃣ Shuffle dataset
balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)

# 7️⃣ Save final prepared dataset
balanced_df.to_csv("processed_data/prepared_dataset.csv", index=False)

print("✅ Dataset prepared successfully with 'diff' feature!")
print("Shape:", balanced_df.shape)
print("Class distribution:\n", balanced_df['state'].value_counts())