from flask import Flask, request, render_template, jsonify
from llm_handler import LLMHandler, AvailableLLMs  # Custom LLM handler file
from neo4j import GraphDatabase
import os
import json
import google.cloud.storage as gcs

from utils.neo4j_client import get_driver
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
    print("query: ", query  )
    recommendation_text, graph_nodes = llm_handler.get_llm_response(query)  # LLM and RAG logic

    # turn the graph_nodes into a list of dictionaries
    graph_nodes_dict = []
    for node in graph_nodes:
        node_dict_ele = {
            "element_id": node.element_id,
            "title": node["title"],
            "description": node["description"],
            "images": node["images"]
        }
        graph_nodes_dict.append(node_dict_ele)

    print(graph_nodes_dict)

    return jsonify({"text": recommendation_text, "products": graph_nodes_dict})

# Close Neo4j driver when app stops
@app.teardown_appcontext
def close_driver(error):
    llm_handler.neo4j_driver.close()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
