# 🧠 BCI-Based Cognitive State Detection for Human Safety

## 📌 Project Overview

This project presents a Brain-Computer Interface (BCI)-based cognitive state detection system designed to enhance human safety in hazardous environments. The system analyzes EEG (Electroencephalogram) signals to identify different cognitive states such as Stress, Fatigue, Panic, Distraction, and Normal. A Support Vector Machine (SVM) model is used for classification, and the project demonstrates a real-time prediction pipeline using EEG hardware integrated with Raspberry Pi and ESP32.

---

## 🎯 Objectives

- Collect EEG signals from a wearable EEG device.
- Preprocess and clean raw EEG data.
- Extract statistical and signal-based features.
- Train an SVM classifier to recognize cognitive states.
- Perform real-time prediction using Raspberry Pi and ESP32.
- Improve human safety by monitoring mental states in real time.

---

## ✨ Features

- EEG data preprocessing and cleaning
- Statistical feature extraction
- SVM-based cognitive state classification
- Real-time EEG data acquisition
- Real-time prediction pipeline
- EEG visualization graphs
- Confusion Matrix visualization
- Model Accuracy visualization

---

## 🛠 Hardware Used

- Single-Channel EEG Electrode (Signal Acquisition)
- EEG Amplifier Module
- ESP32
- Raspberry Pi
- Jumper Wires
- USB Cable
- LED Indicator (Visual Alert)
- Active Buzzer (Audio Alert)
- Breadboard

---

## 💻 Software & Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- PySerial
- Joblib
- VS Code

---

## 🧠 Machine Learning Model

- Algorithm: Support Vector Machine (SVM)
- Feature Scaling: StandardScaler
- Label Encoding
- Hyperparameter Tuning using GridSearchCV

---

## 📂 Project Structure

```text
BCI-based-cognitive-state-detection-for-human-safety/
│
├── data/
│   ├── person1_data.csv
│   ├── person2_data.csv
│   ├── ...
│
├── processed_data/
│   ├── eeg_data.csv
│   ├── cleaned_dataset.csv
│   ├── features_data.csv
│   ├── final_features_dataset.csv
│   └── ...
│
├── models/
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── eeg_collect.py
├── clean_data.py
├── feature_extraction.py
├── prepare_dataset.py
├── svm_training.py
├── predict_state.py
├── realtime_demo.py
├── confusion_matrix.py
├── accuracy_graph.py
├── brainwave_graph.py
├── eeg_signal_graph.py
├── eeg_feature_variation_graph.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/BCI-based-cognitive-state-detection-for-human-safety.git
```

Move into the project folder:

```bash
cd BCI-based-cognitive-state-detection-for-human-safety
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Collect EEG Data

```bash
python eeg_collect.py
```

### Clean Dataset

```bash
python clean_data.py
```

### Extract Features

```bash
python feature_extraction.py
```

### Prepare Dataset

```bash
python prepare_dataset.py
```

### Train the Model

```bash
python svm_training.py
```

### Predict Cognitive State

```bash
python predict_state.py
```

### Real-Time Prediction

```bash
python realtime_demo.py
```

---

## 📊 Results

- Machine Learning Algorithm: Support Vector Machine (SVM)
- Model Accuracy: **70.11%**
- Real-time EEG cognitive state prediction
- Statistical feature-based classification

---

## 📈 Visualizations

The project includes:

- EEG Signal Graph
- EEG Brain Wave Decomposition
- EEG Feature Variation Graph
- Confusion Matrix
- Model Accuracy Graph

> **Note:** Some visualization scripts use simulated EEG signals for demonstration and report illustrations. The real-time prediction pipeline is designed to work with EEG data acquired from the hardware setup (EEG sensor, amplifier, ESP32, and Raspberry Pi).

---

## 🔮 Future Enhancements

- Deep Learning-based EEG classification
- Multi-channel EEG processing
- IoT cloud monitoring
- Mobile application integration
- Real-time alert notification system

---

## 👩‍💻 Author

**Nalina**

B.Tech Artificial Intelligence and Data Science

---

## 📜 Purpose

This project was developed as a final-year academic project to enhance the safety of manual scavengers and workers operating in hazardous environments. By monitoring EEG-based cognitive states, the system aims to provide early detection of mental stress, fatigue, panic, and distraction, helping reduce workplace risks and improve human safety.

## 📄 Patent

This project resulted in a published Indian Patent Application.

**Title:** Brain-Computer Interface Based Cognitive State Detection and Alert System for Worker Safety in Hazardous Environments

**Application No.:** 202641080321 A

**Publication Date:** 03 July 2026

The patent publication document is available in the `patent/` folder.