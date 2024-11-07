"""
Authors: Nandini, Sally
"""
from . import print_debug, bigquery_client
from .rag_constants import *

from neo4j import GraphDatabase, Result
import pandas as pd
from rdflib import Graph as RDFGraph, Namespace, RDF, RDFS, OWL
from py2neo import Graph as Neo4jGraph, Node, Relationship
import os

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def get_driver():
    print(f"Connected to Neo4J instance at {NEO4J_URI}")
    return driver

def exec_query(driver, q, parameters=None, db="neo4j"):
    """
    Generic function for executing queries against the Neo4j database.
    This function will execute the query and return a dataframe.
    
    Args:
        driver: Neo4j driver instance
        q: Query string
        parameters: Dictionary of query parameters (optional)
        db: Database name (defaults to "neo4j")
    """
    # If no parameters provided, use empty dict
    parameters = parameters or {}
    
    records, summary, keys = driver.execute_query(
        q,
        parameters_=parameters, 
        database_=db,
    )

    print("Query `{query}` returned {records_count} records in {time} ms.".format(
        query=summary.query, 
        records_count=len(records),
        time=summary.result_available_after
    ))

    for r in records:  
        print(r)
    
    return records

class KnowledgeGraphLoader():

    class GraphNodes:
        """
        temporary class for holding nodes as the graph is built in py2neo
        """
        nodes = {}

        # 'Ontology mainCategory': 'CSV mainCategory'
        ontology_csv_category_mapping = {
            'GiftCard': 'Gift Cards',
            'All_Beauty': 'All Beauty',
            'Amazon_Fashion': 'AMAZON FASHION',
            'Amazon_Home': 'Amazon Home',
            'Appliances': '',
            'Arts_Crafts_and_Sewing': 'Arts, Crafts & Sewing',
            'Automotive': '',
            'Baby_Products': '',
            'Beauty_and_Personal_Care': '',
            'Books': '',
            'CDs_and_Vinyl': '',
            'Cell_Phones_and_Accessories': '',
            'Clothing_Shoes_and_Jewelry': '',
            'Digital_Music': '',
            'Electronics': 'All Electronics',
            'Grocery_and_Gourmet_Food': 'Grocery',
            'Handmade_Products': '',
            'Health_and_Household': '',
            'Health_and_Personal_Care': 'Health & Personal Care',
            'Home_and_Kitchen': '',
            'Industrial_and_Scientific': '',
            'Kindle_Store': '',
            'Magazine_Subscriptions': '',
            'Movies_and_TV': '',
            'Musical_Instruments': '',
            'Office_Products': 'Office Products',
            'Patio_Lawn_and_Garden': '',
            'Pet_Supplies': '',
            'Software': 'Software',
            'Sports_and_Outdoors': 'Sports & Outdoors',
            'Subscription_Boxes': '',
            'Tools_and_Home_Improvement': '',
            'Toys_and_Games': 'Toys & Games',
            'Video_Games': 'Video Games',
            'Unknown': 'Unknown',
        }

        def __init__(self, neo4j_graph):
            self.neo4j_graph = neo4j_graph

        def get_or_make_node(self, node_label):
            if node_label in self.ontology_csv_category_mapping:
                mapped_node_label = self.ontology_csv_category_mapping[node_label]
                if mapped_node_label:
                    print_debug(f"found mapped node label for {node_label}: {mapped_node_label}")
                    node_label = mapped_node_label
            
            if node_label in self.nodes:
                node = self.nodes[node_label]
            else:
                node = Node(node_label)
                self.neo4j_graph.create(node)
                self.nodes[node_label] = node
            return node

    def __init__(self, ontology, csv_file=""):
        self.ontology = ontology
        self.csv_file = csv_file
        self.load_ontology()

        if csv_file:
            # Read local CSV Data
            self.csv_data = pd.read_csv(self.csv_file)
        else:
            # Read CSV data from BigQuery
            self.csv_data = bigquery_client.get_dataset()

    def load_ontology(self, ontology=""):
        if ontology == "":
            ontology = self.ontology
        else:
            self.ontology = ontology
        # Load the ontology
        self.ontology_graph = self._load_ontology(self.ontology)
        # Extract classes and properties from the ontology
        self.classes = self._extract_classes(self.ontology_graph)
        self.properties = self._extract_properties(self.ontology_graph)
    
    def load_knowledge_graph(self, reset=True):
        """
        reset: whether to reset the knowledge graph in Neo4J
        """
        # perform a fresh load
        if not reset:
            print("Reset is set to False. No action to take.")
            return
        
        with get_driver() as driver:
            driver.session().run("MATCH (n) DETACH DELETE n")
            driver.close()
            print("Deleted existing graph; ready to load knowledge graph")
    
        self.neo4j_graph = Neo4jGraph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.class_nodes = self.GraphNodes(self.neo4j_graph)

        # Create the knowledge graph in Neo4j
        # load item categories; node classes and subclasses
        self._load_knowledge_graph_classes()
        print(f"Loaded {len(self.class_nodes.nodes)} classes from {self.ontology}")
            
        self._load_products()
        if self.csv_file:
            print(f"Loaded products in {len(self.class_nodes.nodes)} classes from {self.csv_file}")
        else:
            print(f"Loaded products in {len(self.class_nodes.nodes)} classes from BiqQuery")
    
        print("Done")
    
    def _load_ontology(self, file_path):
        g = RDFGraph()
        g.parse(file_path)
        print(f"Ontology loaded from {file_path}: {g}")
        return g
    
    def _extract_classes(self, graph):
        classes = {}
        for s, p, o in graph.triples((None, RDF.type, OWL.Class)):
            class_name = s.split('#')[-1]
            classes[class_name] = {
                'subClassOf': [sc.split('#')[-1] for sc in graph.objects(s, RDFS.subClassOf)]
            }
    
        return classes
    
    def _extract_properties(self, graph):
        properties = {}
        for s, p, o in graph.triples((None, RDF.type, OWL.ObjectProperty)):
            prop_name = s.split('#')[-1]
            domain = [d.split('#')[-1] for d in graph.objects(s, RDFS.domain)]
            range_ = [r.split('#')[-1] for r in graph.objects(s, RDFS.range)]
            properties[prop_name] = {'domain': domain, 'range': range_}
        return properties
    
    def _load_knowledge_graph_classes(self):
        """
        Loads classes and subclasses as nodes into the KG with 'subClassOf' relationships.
        """
        neo4j_graph = self.neo4j_graph
        classes = self.classes
        class_nodes = self.class_nodes
        
        for cls in classes:
            subdict = classes[cls]
    
            main_node = class_nodes.get_or_make_node(cls)
            
            for relationship in subdict:
                if subdict[relationship]:
                    subclasses = subdict[relationship]
                    print_debug(f"{cls}: {relationship} {subclasses}")
    
                    for s in subclasses:
                        prop = relationship
                        prop_node = class_nodes.get_or_make_node(s)
                        rel = Relationship(main_node, prop, prop_node)
                        neo4j_graph.create(rel)
    
    def _load_products(self):
        neo4j_graph = self.neo4j_graph
        csv_data = self.csv_data
        category_nodes = self.class_nodes
        
        for _, row in csv_data.iterrows():
            main_category = row['mainCategory']
    
            # Skip rows with missing mainCategory
            if pd.isna(main_category):
                continue
    
            # Ensure main_category is not None and is a string
            main_category = str(main_category)
            title = row['title']
    
            # Filter attributes, making sure none of them are None
            attributes = {k: v for k, v in row.to_dict().items() if isinstance(v, (int, float, str)) and pd.notna(v)}
            main_node = Node("Product", title, **attributes)
    
            for attr in attributes:
                if attr == 'mainCategory':
                    category = attributes[attr]
                    attr_node = category_nodes.get_or_make_node(category)
                    rel = Relationship(main_node, attr, attr_node)
                    neo4j_graph.create(rel)
                    print_debug(f"Loaded node-rel-node: {title}, {attr}, {category}")

