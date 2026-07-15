import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv('processed_data/final_features_dataset.csv')

# Remove spaces (safe)
df.columns = df.columns.str.strip()

print("Columns in dataset:", df.columns.tolist())

# ✅ USE CORRECT FEATURES
X = df[['mean', 'std', 'var', 'max', 'min', 'range', 
        'diff_mean', 'diff_std', 'skew', 'kurt']]

y = df['state']

# Load model
model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')

# Transform
X_scaled = scaler.transform(X)
y_encoded = le.transform(y)

# Predict
y_pred = model.predict(X_scaled)

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)

disp.plot()
plt.title("Confusion Matrix")

plt.show()