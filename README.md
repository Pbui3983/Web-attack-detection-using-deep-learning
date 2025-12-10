Web Attack Detection — Deep Learning with WEB-IDS23

This project focuses on detecting web-based cyber attacks using deep learning models trained on the WEB-IDS23 dataset.
We implement a full machine learning pipeline including data preprocessing, class balancing, feature engineering, and model evaluation to achieve robust network attack classification.

## Key Features

Complete data processing pipeline (cleaning → feature engineering → encoding → balancing → splitting → scaling)

Support multiple deep learning architectures:

CNN 1D

LSTM / GRU

Hybrid CNN-LSTM

Multi-class classification of 11 attack categories

Metrics included:

Accuracy, Loss curves

Confusion Matrix

Classification report

ROC-AUC, F1-score

## Dataset: WEB-IDS23

A modern intrusion detection dataset focusing on web traffic attacks such as:

SQL Injection

XSS

CSRF

Directory Traversal

File Inclusion

Brute Force

and more...

Dataset includes 45 extracted features representing HTTP traffic and session behavior.

Full preprocessing implementation available in WEBIDS23Preprocessor.

## Model Training Workflow
Load Dataset → Clean & Transform → Encode Labels → Handle Imbalance (SMOTE + Undersampling)
→ Train/Test Split → Standard Scaling → Deep Learning Model Training → Evaluation


Training with:

batch_size=256

EarlyStopping + ReduceLROnPlateau

Up to 200 epochs

Project Structure
├── data/
│   ├── raw/                 # Original WEB-IDS23 dataset
│   └── processed/           # After preprocessing
├── models/                  # Saved trained models (.h5/.pth)
├── notebooks/               # Experiments & analysis
├── src/
│   ├── preprocessing.py     # Data pipeline class
│   ├── train_cnn.py         # CNN model training script
│   ├── train_lstm.py        # LSTM model training script
│   └── utils.py
└── README.md

## Future Development

Improve recall for minority attack classes

Deploy model as real-time Web Application Firewall (WAF)

Add explainability: SHAP / Feature Attribution

Serve trained model via FastAPI + Docker

## Requirements
Python 3.8+
TensorFlow / PyTorch
Scikit-learn
Imbalanced-learn
Pandas, NumPy, Matplotlib, Seaborn


Install dependencies:

pip install -r requirements.txt

## Contact

Author: Bui Trong Phuc
Purpose: Research and academic use
Feel free to fork, contribute, and improve the project! 🙌