import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import warnings

# --- KONFIGURASI ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "12345678")
DATA_PATH = r"Dataset\secom.data"         
LABEL_PATH = r"Dataset\secom_labels.data" 

warnings.filterwarnings("ignore")

class KnowledgeGraphBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        
    def close(self):
        self.driver.close()

    def get_active_sensors_from_rules(self):
        """
        Cek Neo4j: Sensor apa saja yang ada di dalam Rules?
        Kita hanya akan memasukkan data sensor ini agar graph tidak berat.
        """
        print("   -> Fetching active rules from Neo4j...")
        active_sensors = set()
        rules = []
        
        query = "MATCH (r:Rule) RETURN r.name, r.sensor, r.threshold, r.condition"
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                r = record.data()
                active_sensors.add(r['r.sensor'])
                rules.append(r)
        
        print(f"   -> Found {len(rules)} Rules involving {len(active_sensors)} unique Sensors.")
        return list(active_sensors), rules

    def load_data(self, active_sensors):
        """Load CSV tapi hanya ambil kolom sensor yang penting."""
        print("   -> Loading dataset (Smart Filter)...")
        
        # 1. Load Labels untuk Timestamp simulasi
        df_lbl = pd.read_csv(LABEL_PATH, sep=" ", header=None)
        df_lbl.columns = ["label", "timestamp"]
        
        # 2. Load Sensor Data
        df_x = pd.read_csv(DATA_PATH, sep=" ", header=None)
        # Namai kolom f_0 s.d f_589
        df_x.columns = [f"f_{i}" for i in range(df_x.shape[1])]
        
        # 3. Gabungkan
        df_combined = pd.concat([df_lbl, df_x], axis=1)
        
        # 4. Ambil sampel (misal 500 baris pertama) agar demo cepat
        # Hapus .head(500) jika ingin SEMUA data (mungkin agak lama)
        df_final = df_combined.head(500).copy() 
        
        # 5. Filter kolom: Cuma ambil ID, Waktu, dan Sensor Penting
        keep_cols = ["timestamp"] + active_sensors
        
        # Siapkan list of dict untuk batch insert
        data_rows = []
        for idx, row in df_final.iterrows():
            props = {"id": idx, "timestamp": row["timestamp"]}
            
            # Masukkan nilai sensor (handle NaN jadi 0.0)
            for sensor in active_sensors:
                val = row.get(sensor, 0.0)
                if pd.isna(val): val = 0.0
                props[sensor] = float(val)
                
            data_rows.append(props)
            
        return data_rows

    def build_graph(self):
        print("\n>>> PHASE 1: INGESTION (Membangun Timeline)...")
        
        # 1. Cek sensor mana yang harus diload (berdasarkan Rule yg sudah ada)
        active_sensors, rules = self.get_active_sensors_from_rules()
        if not active_sensors:
            print("ERROR: Belum ada Rules di Database! Jalankan mining-and-rule.py dulu.")
            return

        # 2. Siapkan Data
        data_rows = self.load_data(active_sensors)
        
        with self.driver.session() as session:
            # 3. Bersihkan Node ProcessStep lama (Reset)
            print("   -> Clearing old ProcessStep nodes...")
            session.run("MATCH (p:ProcessStep) DETACH DELETE p")
            session.run("MATCH (f:FailureEvent) DETACH DELETE f")
            
            # 4. Buat Node ProcessStep
            print(f"   -> Creating {len(data_rows)} ProcessStep nodes...")
            query_nodes = """
            UNWIND $rows AS row
            CREATE (p:ProcessStep {id: row.id, timestamp: row.timestamp})
            SET p += row  // Masukkan semua properti sensor sekaligus
            """
            session.run(query_nodes, rows=data_rows)
            
            # 5. Buat Relasi NEXT_STEP (Linked List)
            print("   -> Linking nodes with [:NEXT_STEP]...")
            query_link = """
            MATCH (p:ProcessStep)
            WITH p ORDER BY p.id
            WITH collect(p) as steps
            FOREACH (i in RANGE(0, size(steps)-2) |
                FOREACH (curr in [steps[i]] |
                    FOREACH (next in [steps[i+1]] |
                        MERGE (curr)-[:NEXT_STEP]->(next)
                    )
                )
            )
            """
            session.run(query_link)

        print("\n>>> PHASE 2: REASONING (Menyuntikkan Kecerdasan)...")
        with self.driver.session() as session:
            # 6. Terapkan Rules (Reasoning)
            print("   -> Connecting Violations (ProcessStep)-[:VIOLATES]->(Rule)...")
            
            query_reasoning = """
            MATCH (p:ProcessStep), (r:Rule {name: $rule_name})
            WHERE p[$sensor_name] > $threshold
            
            // A. KASIH TANDA (Label Merah)
            SET p:FailureEvent
            
            // B. BUAT KETERKAITAN (Garis Penyebab)
            MERGE (p)-[rel:VIOLATES]->(r)
            SET rel.recorded_value = p[$sensor_name]
            """
            
            count_violations = 0
            for r in rules:
                # Kita jalankan query hanya untuk rule GT (Greater Than) sesuai mining
                if r['r.condition'] == 'GT':
                    result = session.run(query_reasoning, 
                                rule_name=r['r.name'], 
                                sensor_name=r['r.sensor'], 
                                threshold=r['r.threshold'])
                    info = result.consume()
                    count_violations += info.counters.relationships_created
            
            print(f"   -> Reasoning Complete! Created {count_violations} violation links.")

if __name__ == "__main__":
    builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_AUTH)
    try:
        builder.build_graph()
        print("\nSUCCESS! Graph is ready. Open Neo4j Browser to visualize.")
    finally:
        builder.close()