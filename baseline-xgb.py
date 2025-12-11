import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
from preprocess import DataPreprocessor

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = r'Dataset\secom.data'
LABELS_PATH = r'Dataset\secom_labels.data'

def train_xgboost_smote():
    print(">>> 1. PREPROCESSING DATA...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run complete preprocessing pipeline
    X, y = preprocessor.preprocess(DATA_PATH, LABELS_PATH)
    
    print(f">>> 2. Original Class Distribution: {y.value_counts().to_dict()}")
    
    # 1. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f">>> 3. Training set shape: {X_train.shape}")
    print(f"    Test set shape: {X_test.shape}")
    
    # 2. APPLY SMOTE
    print(">>> 4. Applying SMOTE to Training Data...")
    smote = SMOTE(random_state=42, sampling_strategy=0.5) # Buat rasio fail:normal jadi 1:2
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"    New Training Distribution: {y_train_res.value_counts().to_dict()}")
    
    # 3. TRAIN XGBOOST
    print(">>> 5. Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=2, 
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_res, y_train_res)
    
    # 4. EVALUATE
    print("\n>>> 6. EVALUATION ON TEST SET (Original Distribution)...")
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:\n [ TN: {tn} | FP: {fp} ]\n [ FN: {fn} | TP: {tp} ]")
    print("-" * 30)
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    train_xgboost_smote()