import os
from google.cloud import secretmanager

def get_secret(project_id, secret_name, version='latest'):
    """
    Gets secret and returns decoded secret. Don't print it.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")