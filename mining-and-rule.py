import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678")
DATA_PATH = r'Dataset\secom.data'
LABELS_PATH = r'Dataset\secom_labels.data'

# TUNING PARAMETERS (Mainkan angka ini!)
# 1. Threshold 97.5% (Setara Mean + 2.2 StdDev) -> Membuang banyak noise normal
PERCENTILE_THRESHOLD = 97.5  

# 2. SNR Naik ke 0.5 -> Hanya ambil sensor yang bedanya JELAS
MIN_SNR = 0.5

# 3. Presisi minimal tetap 5%
MIN_RULE_PRECISION = 0.05    

# 4. Support count tetap 3
MIN_SUPPORT_COUNT = 3

def load_full_dataset():
    """Load dataset lengkap untuk validasi False Positive"""
    print("   -> Loading dataset for validation...")
    df_x = pd.read_csv(DATA_PATH, sep=' ', header=None)
    df_x.columns = [f'f_{i}' for i in range(df_x.shape[1])]
    
    df_y = pd.read_csv(LABELS_PATH, sep=' ', header=None)
    df_y.columns = ['Label', 'Timestamp']
    
    # Label: 1 = Fail, -1 = Pass. Kita ubah jadi 1 dan 0.
    df_x['target'] = df_y['Label'].apply(lambda x: 1 if x == 1 else 0)
    df_x = df_x.fillna(0)
    return df_x

def generate_smart_rules(driver, df_full):
    print("\n[Phase 3] Generating Rules with PERCENTILE & SNR Strategy...")
    
    sensor_cols = [c for c in df_full.columns if c.startswith('f_')]
    
    # Pisahkan Data
    df_fail = df_full[df_full['target'] == 1]
    df_normal = df_full[df_full['target'] == 0]
    
    # --- 1. PRE-AGGREGATION (WINDOWING) ---
    print("   -> Aggregating failure windows (Max-Pooling)...")
    fail_indices = df_fail.index
    fail_data_agg = {col: [] for col in sensor_cols}
    
    for idx in fail_indices:
        # Window 5 langkah ke belakang
        start_loc = max(0, idx - 4) 
        window = df_full.iloc[start_loc : idx+1]
        for col in sensor_cols:
            fail_data_agg[col].append(window[col].max())
            
    df_fail_agg = pd.DataFrame(fail_data_agg)
    
    valid_rules = []
    total_sensors = len(sensor_cols)
    
    for i, sensor in enumerate(sensor_cols):
        if (i+1) % 50 == 0: print(f"      Scanning {i+1}/{total_sensors}...", end='\r')

        # Ambil data series
        normal_series = df_normal[sensor]
        fail_series = df_fail_agg[sensor]
        
        # --- TUNING 1: CEK SIGNAL-TO-NOISE RATIO (SNR) ---
        # Apakah sensor ini "bergerak" saat failure?
        mean_normal = normal_series.mean()
        mean_fail = fail_series.mean()
        std_normal = normal_series.std()
        
        if std_normal == 0: continue
        
        # Rumus SNR Sederhana: Seberapa jauh beda rata-ratanya dibanding sebaran datanya?
        snr = abs(mean_fail - mean_normal) / std_normal
        
        # Jika bedanya tipis (sensor ga ngaruh), SKIP.
        if snr < MIN_SNR:
            continue

        # --- TUNING 2: PAKAI PERSENTIL (BUKAN STD DEV) ---
        # Kita cari batas atas di mana 99.5% data normal berada di bawahnya
        threshold_high = np.percentile(normal_series, PERCENTILE_THRESHOLD)
        
        # Cek kondisi GT (Greater Than)
        matches_fail = fail_series[fail_series > threshold_high]
        tp = len(matches_fail)
        
        matches_normal = normal_series[normal_series > threshold_high]
        fp = len(matches_normal)
        
        if tp >= MIN_SUPPORT_COUNT:
            # Hitung Precision & Recall
            precision = tp / (tp + fp + 1e-9)
            recall = tp / len(df_fail_agg)
            
            # Kita simpan rule ini
            valid_rules.append({
                'sensor': sensor, 
                'condition': 'GT', 
                'threshold': threshold_high,
                'precision': precision, 
                'recall': recall,
                'tp': tp,
                'snr': snr # Simpan skor SNR untuk prioritas
            })

        # Opsional: Bisa tambahkan logika untuk 'LT' (Less Than) dengan np.percentile(..., 0.5)
        # Tapi untuk SECOM biasanya spike 'GT' lebih dominan.

    print(f"\n   -> Found {len(valid_rules)} candidate rules.")
    
    # --- C. SELECT TOP RULES ---
    # Prioritaskan Rule yang punya SNR TINGGI (Pembeda Jelas) dan Precision Bagus
    # Skor kombinasi: Precision * SNR
    valid_rules.sort(key=lambda x: (x['precision'] * x['snr']), reverse=True)
    
    return valid_rules[:15] # Ambil Top 15 

def save_rules_to_neo4j(driver, rules):
    print(f"\n[Phase 3.5] Saving Top {len(rules)} Smart Rules to Neo4j...")
    
    with driver.session() as session:
        session.run("MATCH (r:Rule) DETACH DELETE r") # Reset rule lama
        
        query = """
        CREATE (r:Rule {
            name: $name, sensor: $sensor, condition: $condition, 
            threshold: $threshold, precision: $precision, support_count: $tp
        })
        """
        for r in rules:
            r_name = f"Rule_{r['sensor']}_{r['condition']}"
            session.run(query, 
                        name=r_name, sensor=r['sensor'], condition=r['condition'],
                        threshold=float(r['threshold']), precision=float(r['precision']),
                        tp=int(r['tp']))
            print(f"   [SAVED] {r_name}: Val {r['condition']} {r['threshold']:.2f} "
                  f"(Prec: {r['precision']:.2%}, Found: {r['tp']} fails)")

if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        df = load_full_dataset()
        smart_rules = generate_smart_rules(driver, df)
        save_rules_to_neo4j(driver, smart_rules)
        print("\nDONE! Now run evaluate_performance.py again.")
    finally:
        driver.close()