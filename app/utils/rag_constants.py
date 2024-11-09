import os
from enum import Enum

class Mode(Enum):
    Graph: str = "graph"
    Vector: str = "vector"

MODEL_MODE = Mode.Graph #default mode

# Vertex AI
GEMINI_MODEL = "gemini-1.5-flash-002"
GEMINI_MODEL_REGION = "us-central1"

# Custom Vertex AI
FLANT5_API_ENDPOINT = "us-west1-aiplatform.googleapis.com"
FLANT5_ENDPOINT = "projects/371748443295/locations/us-west1/endpoints/8518052919822516224"
FLANT5_MODEL_REGION = "us-west1"

### Nandini
"""
# Google Cloud Project
PROJECT_ID = "psychic-surf-437820-b8"

# BigQuery
DATASET_ID = "JSON_DATASET"
TABLE_ID = "clean_giftcard"

# Neo4j
NEO4J_URI="neo4j+s://71e5cc7d.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="eVPXWxkCzICKvPqd69D_aSJRJEAS7CeXL5OLqBIxXVI"
"""

### Sally
# Google Cloud Project
PROJECT_ID = "deft-return-439619-h9"

# Google Cloud Storage
REGION = 'us-west2'
BUCKET = 'ontologykg2'

# BigQuery
DATASET_ID = "JSON_DATASET"
TABLE_ID = "clean_giftcard"

# Dataflow API Service Account Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/lawfu/Documents/deft-return-439619-h9-151ce547a5fd.json"

# Neo4j
NEO4J_URI = "neo4j+s://508011ec.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "WVdTBF4CkcyCgzVTrUoCQsXQCSLi0qBLjp_st-EClTw"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

### Mouni
"""
# Google Cloud Project
PROJECT_ID = "project298bmoni"

# Google Cloud Storage
REGION = 'us-west2'
BUCKET = 'ontology-project'

# BigQuery
DATASET_ID = "JSON_DATASET"
TABLE_ID = "clean_giftcard"
"""
