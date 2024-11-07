"""
Author: Nandini
"""

from . import rag_constants

from google.cloud import bigquery

def get_client():
    # Initialize BigQuery client
    client = bigquery.Client()
    return client

def get_dataset():
    """
    This function queries a BigQuery dataset and returns the results as a Pandas DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing the results from the BigQuery query.
    """
    # Initialize BigQuery client
    client = gcp_client.get_bigquery_client()

    # Define your query
    query = f"""
    SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """        

    # Execute the query and convert results to a DataFrame
    results = client.query(query).to_dataframe()
    return results