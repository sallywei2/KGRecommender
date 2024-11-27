
PROMPT_CYPHER_READ =  """
Given the user query please construct a cypher query to retrieve from the database related entries. Only respond with the Cypher query and return the entire node.

The available node properties are:
* "main_category": {main_categories}
* "categories": {categories}
* "average_rating"
* "rating_number": the number of ratings that were submitted for this node

User query:
{user_query}
"""

PROMPT_TEMPLATE_NO_AUGMENTATION = """
{user_query}
"""

FINAL_PROMPT_TEMPLATE = """
Provide up to six recommendations to the user based on the user query. Related information are included in the Response section below.
Provide the element_id of each of the recommended items, separated by commas, at the very end of your response after the word "element_ids:".

User query:
{user_query}

Response:
{response}
"""