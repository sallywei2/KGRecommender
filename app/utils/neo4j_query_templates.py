
PROMPT_CYPHER_READ =  """
Given the user query, please construct a cypher query to retrieve related entries from the database. Prefer generic queries.
Return the entire node. Only respond with the Cypher query.

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
Based on the user query, recommend five products from the Context section below. Provide reasoning for each recommendation.

After you have provided recommendations, print "element_ids:" followed by the element_id of each of the recommendations, separated by commas.

User query:
{user_query}

Context:
{response}
"""