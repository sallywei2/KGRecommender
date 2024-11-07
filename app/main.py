from flask import Flask, request, render_template
from llm_handler import get_llm_response  # Custom LLM handler file
from neo4j import GraphDatabase
import os
import google.cloud.storage as gcs

from utils.neo4j_client import get_driver
from flask_graph_visualization import graph  

app = Flask(__name__)
app.register_blueprint(graph) 

# Neo4j configuration
driver = get_driver()

# Route for homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle LLM queries
@app.route("/get_response", methods=["POST"])
def get_response():
    query = request.form["query"]
    result = get_llm_response(query, driver)  # LLM and RAG logic
    return result

# Close Neo4j driver when app stops
@app.teardown_appcontext
def close_driver(error):
    driver.close()

if __name__ == "__main__":
    app.run(debug=True)
