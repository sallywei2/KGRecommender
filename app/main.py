from flask import Flask, request, render_template, jsonify
from llm_handler import LLMHandler, AvailableLLMs, LLMResponse  # Custom LLM handler file
from utils.neo4j_client import nodes_to_dict
import os
import json
import google.cloud.storage as gcs
import markdown # markdown to html

from flask_graph_visualization import graph  

app = Flask(__name__)
app.register_blueprint(graph) 

llm_handler = LLMHandler(AvailableLLMs.GEMINI)

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def index():
    products = []
    if request.method == 'POST':
        products = request.form.get("products")
    return render_template("index.html",
                           products=products)

# Route to handle LLM queries
@app.route("/get_response", methods=["POST"])
def get_response():
    query = request.form.get("query")
    print("query: ", query)
    llm_response = llm_handler.get_llm_response(query)  # LLM and RAG logic

    print(f"llm_response.graph_nodes_dict len: {len(llm_response.graph_nodes_dict)}")
    
    array = {
        "text": markdown.markdown(llm_response.text) # llm md to html
        , "products": llm_response.graph_nodes_dict
        , "query": llm_response.cypher_query
        }

    return jsonify(array)

# Close Neo4j driver when app stops
@app.teardown_appcontext
def close_driver(error):
    llm_handler.neo4j_driver.close()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
