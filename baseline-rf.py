import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = r'Dataset\secom.data'
LABELS_PATH = r'Dataset\secom_labels.data'

def load_data():
    print(">>> Loading Data...")
    
    # Load Features
    # Gunakan r'' untuk path windows agar aman
    df_x = pd.read_csv(DATA_PATH, sep=' ', header=None)
    df_x = df_x.fillna(0) 
    
    # Load Labels
    df_y = pd.read_csv(LABELS_PATH, sep=' ', header=None)
    
    # --- PERBAIKAN DI SINI ---
    # Kita harus memberi nama kolomnya dulu agar pandas tahu mana yang 'Label'
    df_y.columns = ['Label', 'Timestamp']
    
    # Sekarang kolom 'Label' sudah ada, kode ini tidak akan error lagi
    # Label asli: -1 (Pass), 1 (Fail). Kita ubah jadi 0 (Pass), 1 (Fail)
    y = df_y['Label'].apply(lambda x: 1 if x == 1 else 0)
    
    return df_x, y

def train_xgboost_smote(X, y):
    print(f">>> Original Class Distribution: {y.value_counts().to_dict()}")
    
    # 1. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. APPLY SMOTE (HANYA DI TRAINING DATA!)
    # Kita perbanyak data failure biar model tidak bias ke normal
    print(">>> Applying SMOTE to Training Data...")
    smote = SMOTE(random_state=42, sampling_strategy=0.5) # Buat rasio fail:normal jadi 1:2
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"    New Training Distribution: {y_train_res.value_counts().to_dict()}")
    
    # 3. TRAIN XGBOOST
    print(">>> Training XGBoost...")
    # scale_pos_weight membantu fokus ke kelas minoritas juga
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=2, # Memberi bobot lebih pada error failure
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_res, y_train_res)
    
    # 4. EVALUATE
    print("\n>>> EVALUATION ON TEST SET (Original Distribution)...")
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Confusion Matrix:\n [ TN: {tn} | FP: {fp} ]\n [ FN: {fn} | TP: {tp} ]")
    print("-" * 30)
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    X, y = load_data()
    train_xgboost_smote(X, y)