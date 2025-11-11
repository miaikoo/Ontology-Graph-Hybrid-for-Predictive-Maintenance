# Ontology-Graph-Hybrid-for-Predictive-Maintenance

## Problem Description

desc goes here

## Software Environment

Python 3.12.6

Numpy 1.26.4

Pandas 2.2.3

Neo4j 6.0.3

Neo4j Dekstop 2.0.5

Neo4j Database 5.26.0 

## Get Started

**Python**

```python
pip install pandas neo4j
```

**Neo4j Desktop**

1. Cypher
    
    ```
    CREATE CONSTRAINT IF NOT EXISTS FOR (p:ProcessStep) REQUIRE p.id IS UNIQUE;
    CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sensor) REQUIRE s.id IS UNIQUE;
    UNWIND range(0, 589) AS i
    MERGE (s:Sensor {id: 'f_' + i});
    ```
    
2. Run /Dataset/read_data.py

