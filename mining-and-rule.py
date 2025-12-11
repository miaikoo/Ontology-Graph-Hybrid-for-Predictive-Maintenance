"""
mining-and-rule.py

Mine failure patterns from the SECOM dataset using SNR and percentile-based
thresholds, then save rules to Neo4j for predictive maintenance.

Usage:
    python .\mining-and-rule.py
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import warnings

warnings.filterwarnings("ignore")

# Configuration
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "12345678")
DATA_PATH = r"Dataset\secom.data"
LABELS_PATH = r"Dataset\secom_labels.data"

# Tuning parameters
PERCENTILE_THRESHOLD = 97.5  # Upper percentile for normal data baseline
MIN_SNR = 0.5  # Minimum signal-to-noise ratio
MIN_RULE_PRECISION = 0.05  # Minimum rule precision (unused, for reference)
MIN_SUPPORT_COUNT = 3  # Minimum number of failures triggering rule


def load_full_dataset():
    """
    Load raw dataset and prepare for rule mining.

    Returns:
        df: DataFrame with features (f_0...f_589) and target label
    """
    print("   -> Loading dataset for rule mining...")
    df_x = pd.read_csv(DATA_PATH, sep=" ", header=None)
    df_x.columns = [f"f_{i}" for i in range(df_x.shape[1])]

    df_y = pd.read_csv(LABELS_PATH, sep=" ", header=None)
    df_y.columns = ["Label", "Timestamp"]

    # Convert labels: -1 (Pass) -> 0, 1 (Fail) -> 1
    df_x["target"] = df_y["Label"].apply(lambda x: 1 if x == 1 else 0)
    df_x = df_x.fillna(0)

    return df_x


def generate_smart_rules(driver, df_full):
    """
    Mine rules using SNR filtering and percentile-based thresholds.

    Args:
        driver: Neo4j driver
        df_full: Full dataset with 'target' column

    Returns:
        list[dict]: Top 15 rules sorted by (precision * SNR)
    """
    print("\n[Phase 1] Generating Rules with SNR & Percentile Strategy...")

    sensor_cols = [c for c in df_full.columns if c.startswith("f_")]

    # Split data by label
    df_fail = df_full[df_full["target"] == 1]
    df_normal = df_full[df_full["target"] == 0]

    # Pre-aggregate failure data using max-pooling over 5-step windows
    print("   -> Aggregating failure windows (Max-Pooling)...")
    fail_indices = df_fail.index
    fail_data_agg = {col: [] for col in sensor_cols}

    for idx in fail_indices:
        start_loc = max(0, idx - 4)
        window = df_full.iloc[start_loc : idx + 1]
        for col in sensor_cols:
            fail_data_agg[col].append(window[col].max())

    df_fail_agg = pd.DataFrame(fail_data_agg)

    valid_rules = []
    total_sensors = len(sensor_cols)

    for i, sensor in enumerate(sensor_cols):
        if (i + 1) % 50 == 0:
            print(f"      Scanning {i + 1}/{total_sensors}...", end="\r")

        normal_series = df_normal[sensor]
        fail_series = df_fail_agg[sensor]

        # Calculate SNR: difference in means relative to normal data spread
        mean_normal = normal_series.mean()
        mean_fail = fail_series.mean()
        std_normal = normal_series.std()

        if std_normal == 0:
            continue

        snr = abs(mean_fail - mean_normal) / std_normal

        # Skip sensors with low SNR
        if snr < MIN_SNR:
            continue

        # Use percentile threshold for robust baseline
        threshold_high = np.percentile(normal_series, PERCENTILE_THRESHOLD)

        # Count matches in failure and normal data
        matches_fail = fail_series[fail_series > threshold_high]
        tp = len(matches_fail)

        matches_normal = normal_series[normal_series > threshold_high]
        fp = len(matches_normal)

        # Save rule if it has sufficient support
        if tp >= MIN_SUPPORT_COUNT:
            precision = tp / (tp + fp + 1e-9)
            recall = tp / len(df_fail_agg)

            valid_rules.append(
                {
                    "sensor": sensor,
                    "condition": "GT",
                    "threshold": threshold_high,
                    "precision": precision,
                    "recall": recall,
                    "tp": tp,
                    "snr": snr,
                }
            )

    print(f"\n   -> Found {len(valid_rules)} candidate rules.")

    # Select top rules by combined score (precision * SNR)
    valid_rules.sort(key=lambda x: (x["precision"] * x["snr"]), reverse=True)

    return valid_rules[:15]


def save_rules_to_neo4j(driver, rules):
    """
    Save mined rules to Neo4j database.

    Args:
        driver: Neo4j driver
        rules: list[dict] of rules to save
    """
    print(f"\n[Phase 2] Saving Top {len(rules)} Rules to Neo4j...")

    with driver.session() as session:
        # Clear old rules
        session.run("MATCH (r:Rule) DETACH DELETE r")

        query = """
        CREATE (r:Rule {
            name: $name, sensor: $sensor, condition: $condition,
            threshold: $threshold, precision: $precision, support_count: $tp
        })
        """

        for r in rules:
            r_name = f"Rule_{r['sensor']}_{r['condition']}"
            session.run(
                query,
                name=r_name,
                sensor=r["sensor"],
                condition=r["condition"],
                threshold=float(r["threshold"]),
                precision=float(r["precision"]),
                tp=int(r["tp"]),
            )
            print(
                f"   [SAVED] {r_name}: {r['condition']} {r['threshold']:.2f} "
                f"(Precision: {r['precision']:.2%}, Support: {r['tp']})"
            )


if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        df = load_full_dataset()
        smart_rules = generate_smart_rules(driver, df)
        save_rules_to_neo4j(driver, smart_rules)
        print("\nRule mining and Neo4j ingestion complete!")
    finally:
        driver.close()