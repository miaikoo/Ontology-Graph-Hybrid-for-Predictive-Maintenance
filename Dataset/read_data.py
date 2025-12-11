import pandas as pd
from neo4j import GraphDatabase
import os
import numpy as np
from sys import path
path.insert(0, '..')

from preprocess import DataPreprocessor

# --- 1. CONFIGURATION ---
DATA_PATH = r'secom.data'
LABELS_PATH = r'secom_labels.data'
NEO4J_PASS = "12345678"

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"

# --- 2. PREPROCESS DATA ---
print("Loading and preprocessing data...")
try:
    preprocessor = DataPreprocessor()
    df_features, df_labels = preprocessor.preprocess(DATA_PATH, LABELS_PATH)
    
    print(f"Successfully preprocessed {len(df_features)} rows.")
    print(f"Final feature count: {len(df_features.columns)}")

except FileNotFoundError:
    print(f"Error: Data files not found. Check your paths.")
    print(f"Tried to find DATA_PATH at: {DATA_PATH}")
    print(f"Tried to find LABELS_PATH at: {LABELS_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

# --- 3. NEO4J QUERY DEFINITIONS ---
CREATE_GRAPH_QUERY = """
// 1. Find the :ProcessStep node or create it
MERGE (p:ProcessStep {id: $step_id})

// 2. Set all other properties from the map (timestamp, status, f_0...f_589)
SET p += $props

// 3. Add the :FailureEvent label if it failed (status = 1)
WITH p
CALL apoc.do.when(
    $props.status = 1,
    'SET p:FailureEvent',
    '',
    {p: p}
) YIELD value
RETURN p
"""

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

        print("Clearing old data (MATCH (n) DETACH DELETE n)...")
        session.run("MATCH (n) DETACH DELETE n")
        print("Old data cleared.")

        # We loop through each ROW in the DataFrame
        print(f"Starting to load {len(df_features)} ProcessSteps...")
        all_rows = df_features.to_dict('records')
        
        for idx, row_data in enumerate(all_rows):
            
            step_id = f"Step_{idx}"
            # Get label for this step
            status = int(df_labels.iloc[idx])
            row_data['status'] = status
            
            # All features are already normalized (float values between 0-1)
            for col in row_data:
                if isinstance(row_data[col], (int, float, np.number)):
                    row_data[col] = float(row_data[col])
            
            # Run the query for this one row
            session.run(CREATE_GRAPH_QUERY, 
                          step_id=step_id, 
                          props=row_data)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} / {len(df_features)} rows.")

        print("All ProcessSteps loaded.")
        
        # After all nodes are in, run the query to link them
        print("Linking steps with [:NEXT_STEP] relationship...")
        session.run(LINK_STEPS_QUERY)
        print("Done linking steps.")
        
        print("Creating index on ProcessStep(id) for faster lookups...")
        try:
            session.run("CREATE CONSTRAINT ProcessStep_id_unique FOR (p:ProcessStep) REQUIRE p.id IS UNIQUE")
        except Exception as e:
            if "already exists" in str(e):
                print("Index/Constraint already exists.")
            else:
                raise e
        print("Index created.")

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