from flask import Flask, request, render_template
from llm_handler import get_llm_response  # Custom LLM handler file
from neo4j import GraphDatabase
import os
import google.cloud.storage as gcs

app = Flask(__name__)

# Neo4j configuration
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

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
