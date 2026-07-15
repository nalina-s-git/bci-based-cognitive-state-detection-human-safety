import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("processed_data/final_features_dataset.csv")

print(f"Total feature rows: {len(df)}")

df = df.dropna()
print(f"After dropping NaN rows: {len(df)}")

# -----------------------------
# AUTO FEATURE SELECTION (FIX FOR YOUR ERROR)
# -----------------------------
# take only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# remove label column if present
if 'state' in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=['state'])

X = numeric_df
y = df['state']

print("Using features:", list(X.columns))

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# Balance dataset
# -----------------------------
df_features = X.copy()
df_features['state'] = y_encoded

classes = df_features['state'].unique()
max_count = df_features['state'].value_counts().max()

balanced_data = []

for cls in classes:
    cls_samples = df_features[df_features['state'] == cls]
    cls_upsampled = resample(
        cls_samples,
        replace=True,
        n_samples=max_count,
        random_state=42
    )
    balanced_data.append(cls_upsampled)

df_balanced = pd.concat(balanced_data)

X_balanced = df_balanced.drop(columns=['state']).values
y_balanced = df_balanced['state'].values

print(f"Balanced dataset rows: {len(df_balanced)}")

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

# -----------------------------
# SVM training
# -----------------------------
param_grid = {
    'C': [1, 5, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")

# -----------------------------
# Evaluation
# -----------------------------
y_pred = grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(grid.best_estimator_, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("✅ Training completed and models saved successfully!")