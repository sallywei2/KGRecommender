import os
from enum import Enum
from .gcp_secretmanager_client import get_secret

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
NEO4J_PASSWORD=get_secret(PROJECT_ID, "neo4j_password", 1)
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

# Neo4j
NEO4J_URI = "neo4j+s://618d6275.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = get_secret(PROJECT_ID, "neo4j_password", 6)

#NEO4J_URI = "bolt://localhost:7687"
#NEO4J_USERNAME = "neo4j"
#NEO4J_PASSWORD = get_secret(PROJECT_ID, "neo4j_password", 3)

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

LLAMA_ENDPOINT = "https://api.llama-api.com"
LLAMA_API_KEY = get_secret(PROJECT_ID, "llama_api_key", 1)
