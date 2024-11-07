from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel
from app.utils.rag_constants import PROJECT_ID, REGION, GENERATIVE_MODEL

vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel(GENERATIVE_MODEL)

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def get_llm_response(query):
    """
    Example query:
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
    """
    return model.generate_content(query)
    