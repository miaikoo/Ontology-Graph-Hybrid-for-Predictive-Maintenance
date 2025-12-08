import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678")

# File Paths (Ensure these match your actual file locations)
DATA_PATH = r'Dataset\secom.data'
LABELS_PATH = r'Dataset\secom_labels.data'

warnings.filterwarnings('ignore')

class PredictiveMaintenanceEvaluator:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def batch_predict(self, features_df):
        """
        Runs the graph-native rule check for a whole batch of data.
        Returns: 
          - predictions (list of 0 or 1)
          - probabilities (list of confidence scores for ROC-AUC)
        """
        predictions = []
        probabilities = []
        
        # We query Neo4j once to get ALL active rules to minimize network overhead
        # In a real streaming app, you cache these rules in Python memory.
        active_rules = self._fetch_all_rules()
        print(f"   -> Loaded {len(active_rules)} active rules from Knowledge Graph.")
        
        total = len(features_df)
        print(f"   -> Evaluating {total} test points...")

        # Iterating row by row to simulate the stream logic
        for idx, row in features_df.iterrows():
            is_failure_predicted = False
            max_confidence = 0.0
            
            # Efficient Python-side check of the rules (Faster than 1567 Neo4j queries)
            for rule in active_rules:
                sensor_val = row.get(rule['sensor'], 0)
                
                threshold_met = False
                if rule['condition'] == 'GT' and sensor_val > rule['threshold']:
                    threshold_met = True
                elif rule['condition'] == 'LT' and sensor_val < rule['threshold']:
                    threshold_met = True
                
                if threshold_met:
                    is_failure_predicted = True
                    # Keep the highest confidence score for ROC-AUC
                    if rule['confidence'] > max_confidence:
                        max_confidence = rule['confidence']
            
            # Append Result
            predictions.append(1 if is_failure_predicted else 0)
            probabilities.append(max_confidence if is_failure_predicted else 0.0)
            
            if (idx + 1) % 100 == 0:
                print(f"      Processed {idx + 1}/{total}...", end='\r')

        return predictions, probabilities

    def _fetch_all_rules(self):
        query = """
        MATCH (r:Rule) 
        RETURN r.sensor as sensor, 
               r.condition as condition, 
               r.threshold as threshold, 
               r.precision as confidence
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

def run_evaluation():
    print(">>> 1. LOADING DATA AND GROUND TRUTH...")
    
    # Load Features
    df_x = pd.read_csv(DATA_PATH, sep=' ', header=None)
    df_x.columns = [f'f_{i}' for i in range(df_x.shape[1])]
    df_x = df_x.fillna(0) # Handling NaNs like before
    
    # Load Labels (Ground Truth)
    df_y = pd.read_csv(LABELS_PATH, sep=' ', header=None)
    df_y.columns = ['Label', 'Timestamp']
    # The label in SECOM is usually -1 (Pass) and 1 (Fail). 
    # Standard ML metrics expect 0 (Pass) and 1 (Fail).
    y_true = df_y['Label'].apply(lambda x: 1 if x == 1 else 0).tolist()
    
    # OPTIONAL: Test on a subset (e.g., first 500 rows) or FULL dataset
    # Remove .head() to test full dataset
    test_limit = 500 
    df_x_test = df_x.head(test_limit)
    y_true_test = y_true[:test_limit]
    
    print(f"    Loaded {len(df_x_test)} rows for evaluation.")

    # --- RUN EVALUATION ---
    evaluator = PredictiveMaintenanceEvaluator(NEO4J_URI, NEO4J_AUTH)
    try:
        y_pred, y_scores = evaluator.batch_predict(df_x_test)
        
        print("\n\n>>> 2. CALCULATION METRICS")
        print("="*60)
        
        # 1. Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true_test, y_pred, labels=[0, 1]).ravel()
        print(f"Confusion Matrix:")
        print(f" [ TN: {tn} | FP: {fp} ]")
        print(f" [ FN: {fn} | TP: {tp} ]")
        print("-" * 30)

        # 2. Main Metrics
        acc = accuracy_score(y_true_test, y_pred)
        prec = precision_score(y_true_test, y_pred, zero_division=0)
        rec = recall_score(y_true_test, y_pred, zero_division=0)
        f1 = f1_score(y_true_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true_test, y_scores)
        except ValueError:
            auc = 0.0 # Handle case if only one class is present in test set

        print(f"ACCURACY  : {acc:.4f}")
        print(f"PRECISION : {prec:.4f}")
        print(f"RECALL    : {rec:.4f}")
        print(f"F1-SCORE  : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print("="*60)
        
    finally:
        evaluator.close()

if __name__ == "__main__":
    run_evaluation()