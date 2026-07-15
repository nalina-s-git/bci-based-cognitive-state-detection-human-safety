# predict_state.py

import pandas as pd
import joblib
from collections import Counter

# Load model
model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Load correct dataset
test_df = pd.read_csv('processed_data/final_features_dataset.csv')

# Print columns (for debugging)
print("Columns in dataset:", test_df.columns)

# Select only numeric columns (safe trick)
X = test_df.select_dtypes(include='number')

# Scale
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Majority vote
final_prediction = Counter(y_pred).most_common(1)[0][0]

# Decode
predicted_state = label_encoder.inverse_transform([final_prediction])[0]

# Output
print("\n==============================")
print("🧠 Predicted Mental State:", predicted_state)
print("==============================\n")