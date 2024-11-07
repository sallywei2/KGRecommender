from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel

from .utils.rag_constants import PROJECT_ID, MODEL_REGION, GENERATIVE_MODEL
from .utils.rag_constants import Mode, MODEL_MODE
from .utils.neo4j_client import get_driver,exec_query, KnowledgeGraphLoader

vertexai.init(project=PROJECT_ID, location=MODEL_REGION)
model = GenerativeModel(GENERATIVE_MODEL)

PROMPT_CYPHER_READ =  """
Given the user query please construct a cypher query to retrieve from the database related entries. Only respond with the Cypher query, and limit the responses to 10.

The valid mainCategory values to filter on are:
{main_categories}

User query:
{user_query}
"""

PROMPT_TEMPLATE_NO_AUGMENTATION = """
{user_query}
"""

FINAL_PROMPT_TEMPLATE = """
Provide three to six recommendations to the user based on the user query. Related information are included in the Response section below.

User query:
{user_query}

Response:
{response}
"""

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def augment_from_graph(user_query):
    # default non-augmented query
    augmented_query = PROMPT_TEMPLATE_NO_AUGMENTATION.format(
        user_query=user_query
    )

    try:
        # Retrieve related information from Neo4J
        # Use the LLM to generate a Cypher query for us
        prompt_for_cypher = PROMPT_CYPHER_READ.format(
            main_categories=KnowledgeGraphLoader.GraphNodes.ontology_csv_category_mapping.values(),
            user_query=user_query
            )
        cypher_response = model.generate_content(prompt_for_cypher)
        print(f"Cypher response: {cypher_response}")

        # Extract just the Cypher query from between the ```cypher and ``` markers
        cypher_text = cypher_response.text
        cypher_query = cypher_text.split('```cypher\n')[1].split('\n```')[0]
        
        print(f"Cypher query: {cypher_query}")
        r = exec_query(
            get_driver(),
            cypher_query
        )
        print(f"Retrieved Cypher response: {r}")
         # Add the retrieved results to the prompt
        augmented_query = FINAL_PROMPT_TEMPLATE.format(
            user_query=user_query,
            response=r
        )
    except Exception as e:
        print(f"Could not retrieve related information from Neo4J: {e}")
        r = "" # leave r blank
   
    return augmented_query

def augment_from_vector(user_query):
    # unimplemented
    return user_query

def get_llm_response(user_query):
    """
    Example query:
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
    """
    if MODEL_MODE == Mode.Graph:
        augmented_query = augment_from_graph(user_query)
    elif MODEL_MODE == Mode.Vector:
        augmented_query = augment_from_vector(user_query)

    model_response = model.generate_content(augmented_query)
    print(f"Final query: {augmented_query}\nModel response: {model_response}")
    return model_response.text

def prompt_llm(prompt):
    return model.generate_content(prompt).text

def get_model():
    return model