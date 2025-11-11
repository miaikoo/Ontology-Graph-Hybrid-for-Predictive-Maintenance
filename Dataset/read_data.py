import pandas as pd
from neo4j import GraphDatabase
import os

# --- 1. CONFIGURATION ---
DATA_PATH = r'secom.data'
LABELS_PATH = r'secom_labels.data'
NEO4J_PASS = "12345678"
# --------------------

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"

# --- 2. LOAD DATA WITH PANDAS ---
print("Loading data from files...")
try:
    # Load features
    df_features = pd.read_csv(DATA_PATH, sep=' ', header=None)
    # Create feature names f_0 to f_589
    df_features.columns = [f'f_{i}' for i in range(df_features.shape[1])]

    # Load labels
    df_labels = pd.read_csv(LABELS_PATH, sep=' ', header=None)
    df_labels.columns = ['Label', 'Timestamp']

    # Combine into one DataFrame
    df_secom = pd.concat([df_features, df_labels], axis=1)
    print(f"Successfully loaded {len(df_secom)} rows.")

except FileNotFoundError:
    print(f"Error: Data files not found. Check your paths.")
    print(f"Tried to find DATA_PATH at: {DATA_PATH}")
    print(f"Tried to find LABELS_PATH at: {LABELS_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

# --- 3. NEO4J QUERY DEFINITIONS ---

# This query transforms one row (a ProcessStep) and its readings
CREATE_GRAPH_QUERY = """
// 1. Find the :ProcessStep node or create it
MERGE (p:ProcessStep {id: $step_id})
// 2. Set its properties (timestamp/status)
SET p.status = $status, p.timestamp = $timestamp

// 3. Add the :FailureEvent label if it failed (status = 1)
WITH p
CALL apoc.do.when(
    $status = 1,
    'SET p:FailureEvent',
    '',
    {p: p}
) YIELD value

// 4. Loop through all features we passed in
UNWIND $features AS feature_data

// 5. Find the matching :Sensor node (which we already created in Neo4j)
MATCH (s:Sensor {id: feature_data.sensor_id})

// 6. Create the new :Reading node and connect everything
CREATE (r:Reading {value: feature_data.value})
CREATE (p)-[:HAS_READING]->(r)
CREATE (r)-[:MEASURED_BY]->(s)
"""

# This query links ProcessSteps in order based on time
LINK_STEPS_QUERY = """
// Find all steps, order them by time
MATCH (p:ProcessStep)
WITH p ORDER BY p.timestamp
// Collect them into a list
WITH collect(p) AS steps
// Loop through the list, linking each node to the next
UNWIND range(0, size(steps) - 2) AS i
WITH steps[i] AS p1, steps[i+1] AS p2
MERGE (p1)-[:NEXT_STEP]->(p2);
"""

# --- 4. NEO4J DATA INGESTION ---

try:
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    with driver.session() as session:
        # Verify connection
        session.run("RETURN 1")
        print("Connection successful.")

        # We loop through each ROW in the DataFrame
        print(f"Starting to load {len(df_secom)} ProcessSteps...")
        for idx, row in df_secom.iterrows():
            
            step_id = f"Step_{idx}"
            status_label = row['Label']
            timestamp_val = row['Timestamp']
            
            # Create the list of features for this row
            features_list = []
            for i in range(590):
                feature_name = f'f_{i}'
                feature_value = row[feature_name]
                
                # IMPORTANT: Only create a :Reading node if the value is not null (NaN)
                if pd.notna(feature_value):
                    features_list.append({
                        'sensor_id': feature_name,
                        'value': float(feature_value) # Ensure it's a standard float
                    })
            
            # Run the query for this one row
            session.run(CREATE_GRAPH_QUERY, 
                        step_id=step_id, 
                        status=int(status_label), # Ensure it's a standard int
                        timestamp=timestamp_val, 
                        features=features_list)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} / {len(df_secom)} rows.")

        print("All ProcessSteps loaded.")
        
        # After all nodes are in, run the query to link them
        print("Linking steps with [:NEXT_STEP] relationship...")
        session.run(LINK_STEPS_QUERY)
        print("Done linking steps.")

    print("Data ingestion complete!")

except Exception as e:
    print(f"\nAn error occurred during Neo4j operations: {e}")
    if "Authentication" in str(e):
        print("--> Check your NEO4J_PASS password in the script.")
    if "Connection" in str(e):
        print("--> Is your Neo4j database RUNNING and accessible at {NEO4J_URI}?")
    if "apoc.do.when" in str(e):
        print("--> Did you install the APOC plugin in Neo4j Desktop?")

finally:
    if 'driver' in locals():
        driver.close()
        print("Neo4j connection closed.")