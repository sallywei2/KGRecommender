from flask import Flask, request, render_template
from llm_handler import LLMHandler, AvailableLLMs  # Custom LLM handler file
from neo4j import GraphDatabase
import os
import google.cloud.storage as gcs

from utils.neo4j_client import get_driver
from flask_graph_visualization import graph  

app = Flask(__name__)
app.register_blueprint(graph) 

llm_handler = LLMHandler(AvailableLLMs.GEMINI)

# Route for homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle LLM queries
@app.route("/get_response", methods=["POST"])
def get_response():
    query = request.form["query"]
    recommendation_text, graph_nodes = llm_handler.get_llm_response(query)  # LLM and RAG logic
    return recommendation_text, graph_nodes

# Close Neo4j driver when app stops
@app.teardown_appcontext
def close_driver(error):
    llm_handler.neo4j_driver.close()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
