from enum import Enum

from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerativeModel

from .utils.rag_constants import PROJECT_ID, GEMINI_MODEL_REGION, GEMINI_MODEL, FLANT5_MODEL_REGION
from .utils.rag_constants import Mode, MODEL_MODE
from .utils.neo4j_client import get_driver,exec_query, KnowledgeGraphLoader
from .utils.flant5_client import FlanT5Client
from .utils.neo4j_query_templates import PROMPT_CYPHER_READ, PROMPT_TEMPLATE_NO_AUGMENTATION, FINAL_PROMPT_TEMPLATE

class AvailableLLMs(Enum):
    GEMINI = 'Gemini',
    FLANT5 = 'Flan-T5'

class LLMHandler:
    """
    Usage:

    from llm_handler import LLMHandler, AvailableLLMs

    llm_handler = LLMHandler(AvailableLLMs.GEMINI)
    llm_handler.get_llm_response(user_query)
    """

    def __init__(self, selected_model):
        if selected_model == AvailableLLMs.GEMINI:
            vertexai.init(project=PROJECT_ID, location=GEMINI_MODEL_REGION)
            self.model = GenerativeModel(GEMINI_MODEL)
        elif selected_model == AvailableLLMs.FLANT5:
            vertexai.init(project=PROJECT_ID, location=FLANT5_MODEL_REGION)
            self.model = FlanT5Client()

        self.neo4j_driver = get_driver()

    def download_from_gcs(self,bucket_name, source_blob_name, destination_file_name):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    
    def prompt_llm(self,prompt):
        response = self.model.generate_content(prompt)
        return self._get_text_from_model_response(response)
    
    def get_llm_response(self, user_query):
        """
        Example query:
        "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
        """
        if MODEL_MODE == Mode.Graph:
            augmented_query = self._augment_from_graph(user_query)
        elif MODEL_MODE == Mode.Vector:
            augmented_query = self._augment_from_vector(user_query)

        try:
            model_response = self.model.generate_content(augmented_query)
            print(f"Final query: {augmented_query}\nModel response: {model_response}")
            return self._get_text_from_model_response(model_response)
        except Exception as e:
            print(f"Error generating content: {e}")
            return ""

    def _augment_from_graph(self, user_query):
        # default non-augmented query
        # we replace this with an augmented query in the next step
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
            print(f"Prompt for cypher:\n{prompt_for_cypher}")
            cypher_response = self.model.generate_content(prompt_for_cypher)
            print(f"Cypher response: {cypher_response}")

            # Extract just the Cypher query from between the ```cypher and ``` markers
            cypher_query = self._get_text_from_cypher_response(cypher_response)

            print(f"Cypher query: {cypher_query}")
            r = exec_query(
                self.neo4j_driver,
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

    def _augment_from_vector(self,user_query):
        # unimplemented
        return user_query

    def _get_text_from_cypher_response(self, response):
        cypher_text = self._get_text_from_model_response(response)
        if 'cypher' in cypher_text:
            cypher_query = cypher_text.split('```cypher\n')[1].split('\n```')[0]
        else:
            cypher_query = cypher_text
        return cypher_query

    def _get_text_from_model_response(self, response):
        if type(response) == vertexai.generative_models.GenerationResponse:
            return response.text
        elif 'text' in response:
            return response['text']
        else:
            return response
