from flask import Flask, render_template, jsonify, request
from neo4j import GraphDatabase
from utils.rag_constants import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

app = Flask(__name__)

try:
    # Neo4j driver initialization
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    # Test the connection
    driver.verify_connectivity()
except Exception as e:
    print(f"Failed to connect to Neo4j: {str(e)}. Please check if the Neo4j server is on and that the connection credentials are correctly set in utils/rag_constants.py")
    driver = None

def get_categories():
    """
    Returns main categories, categories, main category counts, and category counts
    """
    if not driver:
        return [], [], {}, {}
    
    try:
        with driver.session() as session:
            # Query for mainCategory counts
            main_cat_query = """
            MATCH (n)
            WHERE n.mainCategory IS NOT NULL
            WITH n.mainCategory as category, count(n) as count
            RETURN collect({category: category, count: count}) as counts
            """
            
            # Query for category counts (needs to handle comma-separated values)
            cat_query = """
            MATCH (n)
            WHERE n.categories IS NOT NULL
            UNWIND split(n.categories, ',') as category
            WITH trim(category) as trimmed_category
            WHERE trimmed_category <> ''
            WITH trimmed_category, count(*) as count
            RETURN collect({category: trimmed_category, count: count}) as counts
            """
            
            # Get categories and their counts
            main_result = session.run(main_cat_query).single()
            cat_result = session.run(cat_query).single()
            
            # Process mainCategories
            main_cats = []
            main_cat_counts = {}
            for item in main_result['counts']:
                cat = item['category'].strip("'").strip('"').strip()
                if cat:
                    main_cats.append(cat)
                    main_cat_counts[cat] = item['count']
            
            # Process categories
            cats = []
            cat_counts = {}
            for item in cat_result['counts']:
                cat = item['category'].strip("'").strip('"').strip()
                if cat:
                    cats.append(cat)
                    cat_counts[cat] = item['count']
            
            # Sort categories case-insensitive
            main_cats = sorted(set(main_cats), key=str.casefold)
            cats = sorted(set(cats), key=str.casefold)
            
            return main_cats, cats, main_cat_counts, cat_counts
    except Exception as e:
        print(f"Error getting categories: {str(e)}")
        return [], [], {}, {}

def get_filtered_graph_data(main_categories=None, categories=None):
    if not driver:
        return [], []
    
    try:
        with driver.session() as session:
            conditions = []
            if main_categories:
                conditions.append("n.mainCategory IN $main_categories")
            if categories:
                # Modified to check if any of the categories exist in the comma-separated string
                categories_condition = " OR ".join([
                    "any(cat in split(n.categories, ',') WHERE trim(cat) = $cat)" 
                    for cat in categories
                ])
                if categories_condition:
                    conditions.append(f"({categories_condition})")
            
            where_clause = " OR ".join(conditions) if conditions else "true"
            
            query = f"""
            MATCH (n)-[r]->(m)
            WHERE {where_clause}
            RETURN collect(distinct {{
                id: elementId(n), 
                label: labels(n)[0], 
                properties: properties(n)
            }}) as nodes,
                   collect(distinct {{
                       from: elementId(n), 
                       to: elementId(m), 
                       type: type(r)
                   }}) as relationships
            """
            
            params = {
                "main_categories": main_categories or [],
                "cat": categories[0] if categories else None  # We'll use this in the UNWIND
            }
            
            result = session.run(query, params).single()
            return result['nodes'], result['relationships']
    except Exception as e:
        print(f"Error querying Neo4j: {str(e)}")
        return [], []

@app.route('/')
def index():
    main_categories, categories, main_cat_counts, cat_counts = get_categories()
    return render_template('graph.html', 
                         main_categories=main_categories,
                         categories=categories,
                         main_cat_counts=main_cat_counts,
                         cat_counts=cat_counts)

@app.route('/graph-data')
def get_graph():
    main_cats = request.args.getlist('mainCategory')
    cats = request.args.getlist('category')
    
    nodes, relationships = get_filtered_graph_data(
        main_categories=main_cats if main_cats else None,
        categories=cats if cats else None
    )
    return jsonify({"nodes": nodes, "edges": relationships})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
