
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