# KGRecommender
We'll change the name later

# Installation and Deployment

Click the green "<> Code" button, then Open with Github Desktop.

Github Desktop will prompt you for a location to download the code repository to. Continue with your selection to add the folder /KGRecommender to your selected file location.

## CLoud Setup

### Google Cloud Setup

Make sure you have Owner privileges to the Google Cloud project.

Edit the following in utils/rag_constants.py to your own values:
	
	PROJECT_ID

### Google Cloud Storage

To set up the preprocessing pipeline:

1. Upload the merged dataset (`merged_datasets2.csv`) to the top level of your Cloud Storage.
2. Create two folders, /temp and /staging at the top level of Cloud Storage.
3. (Optional) Turn off "Soft Delete" on your bucket to reduce charges by clearing out temp and staging files as soon as they're deleted (instead of keeping them around for the "soft delete" duration.

Edit the following in utils/rag_constants.py to your own values:
	
	REGION
	BUCKET

### Google BigQuery

You can also upload `sample_data/giftcard.csv` data to BigQuery instead of using the local copy included in this repository.

Edit the following in utils/rag_constants.py to your own values:

	DATASET_ID
	TABLE_ID

#### Dataflow API

In GCP Console, search for DataFlow API (*not* Dataflow). Enable it.

Create a service account following the [guide on Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc)

	Create a service account with the roles your application needs, and a key for that service account, by following the instructions in Creating a service account key.
	Set the environment variable GOOGLE_APPLICATION_CREDENTIALS to the path of the JSON file that contains your credentials. This variable applies only to your current shell session, so if you open a new session, set the variable again.

Get a key for the service account following this [guide to create and delete service account keys](https://cloud.google.com/iam/docs/keys-create-delete#iam-service-account-keys-create-console). Create and save the key as a JSON file.

Edit the following in utils/rag_constants.py to your own values:

	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/.../deft-return-439619-h9-151ce547a5fd.json"


### Neo4j Aura

From the GCP console, search for Neo4j Aura and subscribe to it. Then Enable it.

Click "Manage via Neo4J" which will bring you to Neo4J's website. Create an AuraDB instance and select AuraDB Professional. See also the instructions from [Neo4j's Docs](https://neo4j.com/docs/aura/auradb/getting-started/create-database/).

Edit the following in utils/rag_constants.py to your own values:

	NEO4J_URI
	NEO4J_USERNAME
	NEO4J_PASSWORD

## Local Setup

Download the code and unzip it.

### Install Python Dependencies

	# neo4j, graph data science
	pip install neo4j
	pip install py2neo
	pip install rdflib
	pip install networkx[default]

	# google cloud
	pip install google-cloud
	pip install google-cloud-bigquery
	pip install google-cloud-storage

	# preprocessing pipeline and GCP
	pip install apache_beam
	pip install apache_beam[gcp]
	pip install google-apitools

### Neo4J Desktop

You can connect to a local Neo4J Desktop instance instead of one hosted on GCP.

Install and run Neo4j Desktop.

Edit the following in utils/rag_constants.py to your own values. Below are the default values for a fresh installation of Neo4J Desktop.

	NEO4J_URI = "bolt://localhost:7687"
	NEO4J_USERNAME = "neo4j"
	NEO4J_PASSWORD
	GRAPH_NAME = 'myGraph'

## Run Locally On Sample Data

Start Neo4j Desktop
Start the database engine.

Set the NEO4J_USER and NEO4J_PASSWORD in utils/rag_constants.py to your own Neo4J credentials.

In a console, navigate to the top level of the code and run

	jupyter notebook

In Jupyter Notebook, open sample.ipynb

Run the cells to load the knowledge graph into your local neo4J database.