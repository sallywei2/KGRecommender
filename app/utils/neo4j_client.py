"""
Authors: Nandini, Sally
"""
from . import print_debug #, bigquery_client
from .rag_constants import *
from .dataset_parser import DatasetParser

from neo4j import GraphDatabase, Result
import pandas as pd
from rdflib import Graph as RDFGraph, Namespace, RDF, RDFS, OWL
from py2neo import Graph as Neo4jGraph, Node, Relationship
import os
import collections

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
            'Appliances': 'Appliances',
            'Arts_Crafts_and_Sewing': 'Arts, Crafts & Sewing',
            'Automotive': 'Automotive',
            'Baby_Products': 'Baby Products',
            'Beauty_and_Personal_Care': 'Beauty & Personal Care',
            'Books': 'Books',
            'CDs_and_Vinyl': 'CDs & Vinyl',
            'Cell_Phones_and_Accessories': 'Cell Phones & Accessories',
            'Clothing_Shoes_and_Jewelry': 'Clothes, Shoes & Jewelry',
            'Digital_Music': 'Digital Music',
            'Electronics': 'All Electronics',
            'Grocery_and_Gourmet_Food': 'Grocery',
            'Handmade_Products': 'Handmade Products',
            'Health_and_Household': 'Health & Household',
            'Health_and_Personal_Care': 'Health & Personal Care',
            'Home_and_Kitchen': 'Home & Kitchen',
            'Industrial_and_Scientific': 'Industrial & Scientific',
            'Kindle_Store': 'Kindle Store',
            'Magazine_Subscriptions': 'Magazine Subscriptions',
            'Movies_and_TV': 'Movies & TV',
            'Musical_Instruments': 'Musical Instruments',
            'Office_Products': 'Office Products',
            'Patio_Lawn_and_Garden': 'Patio, Lawn & Garden',
            'Pet_Supplies': 'Pet Supplies',
            'Software': 'Software',
            'Sports_and_Outdoors': 'Sports & Outdoors',
            'Subscription_Boxes': 'Subscription BOxes',
            'Tools_and_Home_Improvement': 'Tools & Home Improvement',
            'Toys_and_Games': 'Toys & Games',
            'Video_Games': 'Video Games',
            'Unknown': 'Unknown',
        }

        def __init__(self, neo4j_graph):
            self.neo4j_graph = neo4j_graph

        def get_or_make_node(self, node_type, node_label, **attributes):
            try:
                if type(node_label) == tuple:
                    node_label = ''.join(node_label)
                reformatted_attributes = self._reformat_attributes(attributes)

                if node_label in self.ontology_csv_category_mapping:
                    mapped_node_label = self.ontology_csv_category_mapping[node_label]
                    if mapped_node_label:
                        print_debug(f"found mapped node label for {node_label}: {mapped_node_label}")
                        node_label = mapped_node_label
                
                if node_label in self.nodes:
                    node = self.nodes[node_label]
                else:
                    print_debug(f"Creating node  ({node_label}:{node_type})")
                    for attr in reformatted_attributes:
                        print_debug(f"  {attr}: {reformatted_attributes[attr]}")
                    node = Node(node_type, node_label, **reformatted_attributes)
                    self.neo4j_graph.create(node)
                    self.nodes[node_label] = node
                return node
            except Exception as e:
                print(f"Error while trying to create the following node in neo4j: (({node_label}:{node_type}))\n  with reformatted attributes:")
                for attr in reformatted_attributes:
                    print(f"{attr}: {reformatted_attributes[attr]}")
                raise e

        def _reformat_attributes(self, attributes):
            """
            py2neo/Neo4J expects attributes of a node to be strings, not tuples.
            convert tuple values into string values before creating a Node.
            """
            reformatted_attributes = {}
            if attributes:
                for attr in attributes:
                    try:
                        if not attr: # attr is not named, then skip it
                            continue
                        value = attributes.get(attr)
                        if value:
                            if type(value) == tuple:
                                reformatted_attributes[attr] = self._convert_tuple_to_string(value)
                            elif type(value) == list:
                                reformatted_attributes[attr] = ','.join(value)
                            elif type(value) in (str, int, float):
                                reformatted_attributes[attr] = value
                            print_debug(f"reformatted attribute {attr}: {value} -> {type(reformatted_attributes[attr])} {reformatted_attributes[attr]}")
                    except Exception as e:
                        print(f"Error while reformatting attribute {attr} with value: {type(value)} {value}")
                        raise e
            return reformatted_attributes
        
        def _convert_tuple_to_string(self, t):
            """
            Recursive function to convert tuples to string
            """
            return_string = None

            if type(t) == list:
                return_string = ', '.join(t)
            elif type(t) is not tuple:
                return_string = t
            else:
                tuple_values = []
                for v in t:
                    if type(v) == tuple:
                        s = self._convert_tuple_to_string(v)
                        if s:
                            tuple_values.append(s)
                    else:
                        if v:
                            tuple_values.append(str(v))
                return_string = ', '.join(tuple_values)
            if type(return_string) == tuple:
                # if the result is still a tuple, convert it again
                return_string = self._convert_tuple_to_string(return_string)
            return return_string


    def __init__(self, ontology, csv_file="", pickle_file=""):
        self.ontology = ontology
        self.csv_file = csv_file
        self.pickle_file = pickle_file
        self.load_ontology()

        if csv_file:
            # Read local CSV Data
            self.csv_data = pd.read_csv(self.csv_file)
        #else:
            # Read CSV data from BigQuery
            #self.csv_data = bigquery_client.get_dataset()

        if pickle_file:
            self.parsed_dataset = DatasetParser()
            self.parsed_dataset.load_from_file(pickle_file)
            
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
        self.product_nodes = self.GraphNodes(self.neo4j_graph)

        # Create the knowledge graph in Neo4j
        # load item categories; node classes and subclasses
        #self._load_knowledge_graph_classes()
        #print(f"Loaded {len(self.class_nodes.nodes)} classes from {self.ontology}")
            
        self._load_products()
        if self.csv_file:
            print(f"Loaded products in {len(self.class_nodes.nodes)} classes from {self.csv_file}")
        elif self.pickle_file:
            print(f"Loaded products in {len(self.class_nodes.nodes)} classes from {self.pickle_file}")
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
    
            main_node = class_nodes.get_or_make_node("Main Category", cls)
            
            for relationship in subdict:
                if subdict[relationship]:
                    subclasses = subdict[relationship]
                    print_debug(f"{cls}: {relationship} {subclasses}")
    
                    for s in subclasses:
                        prop = relationship
                        prop_node = class_nodes.get_or_make_node("Category", s)
                        rel = Relationship(main_node, prop, prop_node)
                        neo4j_graph.create(rel)
    
    def _load_products(self):
        if self.csv_file != "":
            csv_data = self.csv_data
            for _, row in csv_data.iterrows():
                self._load_product_from_row(row)
        if self.pickle_file != "":
            for row in self.parsed_dataset.table:
                self._load_product_from_row(row.fields)
    
    def _load_product_from_row(self, row):
        main_category = row.get('mainCategory')
        if not main_category:
            main_category = row.get('main_category')
        main_category = str(main_category)
        if not main_category:
            return
        
        title = row.get('title')
        if not title:
            return

        attributes = row
        if type(row) != dict:
            row = row.to_dict()
            attributes = {k: v for k, v in row.items() if isinstance(v, (int, float, str)) and pd.notna(v)}

        self._add_product_nodes_and_relationships(title, attributes)

    def _add_product_nodes_and_relationships(self, title, attributes):

        main_node = self.product_nodes.get_or_make_node("Product", title, **attributes)
    
        for attr in attributes:

            attr_node_label = ""
            attr_values = attributes[attr]

            relationships_to_add = {
                'mainCategory':'Main Category' # csv
                , 'main_category': 'Main Category' # pickle
                , 'categories':'Category'
                , 'store':'Store'
            }

            if attr in relationships_to_add:
                attr_node_label = relationships_to_add[attr]
            
            if attr_node_label:
                if attr == 'categories':
                    parentCategory = attributes.get('main_category') or attributes.get('mainCategory')
                    if parentCategory:
                        # relate subcategory with main cateogry
                        self._add_node_and_relationship(
                            main_node = self.class_nodes.get_or_make_node("Main Category", parentCategory)
                            , relationship = attr
                            , attr_node_label = attr_node_label
                            , attr_node_title = parentCategory
                        )
                # create a node for the attribute and relate it to the product
                self._add_node_and_relationship(
                    main_node = main_node
                    , relationship = attr
                    , attr_node_label = attr_node_label
                    , attr_node_title = attr_values)


    def _add_node_and_relationship(self, main_node, relationship, attr_node_label, attr_node_title):
        """
        product_node: a product Node
        relationship: the attribute name. (e.g. brand, feature, catgory)
        attr_node_label: string. Category, Main Category, Brand, etc.
        attr_node_title: string.
        """
        if type(attr_node_title) == tuple:
            for title in attr_node_title:
                self._add_node_and_relationship(
                    main_node = main_node
                    , relationship = relationship
                    , attr_node_label = attr_node_label
                    , attr_node_title = title)
        elif type(attr_node_title) in (str, int, float):
            print_debug(f"Adding relationship...  (n)-[{relationship}]->({attr_node_title}:{attr_node_label})")
            attr_node = self.class_nodes.get_or_make_node(attr_node_label, attr_node_title)
            rel = Relationship(main_node, relationship, attr_node)
            self.neo4j_graph.create(rel)
        else:
            print_debug(f"Skipping relationship: {relationship}, {type(attr_node_title)} {attr_node_title}")