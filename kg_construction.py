from google.adk.models.lite_llm import LiteLlm 
from graph_utilities import graphdb, tool_success, tool_error
from typing import Dict, Any

MODEL_GPT_4O = "openai/gpt-4o"
llm = LiteLlm(model=MODEL_GPT_4O)

def create_uniqueness_constraint(
    label: str,
    unique_property_key: str,
) -> Dict[str, Any]:
    """Creates a uniqueness constraint for a node label and property key.
    A uniqueness constraint ensures that no two nodes with the same label and property key have the same value.
    This improves the performance and integrity of data import and later queries.

    Args:
        label: The label of the node to create a constraint for.
        unique_property_key: The property key that should have a unique value.

    Returns:
        A dictionary with a status key ('success' or 'error').
        On error, includes an 'error_message' key.
    """    
    # Use string formatting since Neo4j doesn't support parameterization of labels and property keys when creating a constraint
    constraint_name = f"{label}_{unique_property_key}_constraint"
    query = f"""CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS
    FOR (n:`{label}`)
    REQUIRE n.`{unique_property_key}` IS UNIQUE"""
    results = graphdb.send_query(query)
    return results

def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: list[str],
) -> Dict[str, Any]:
    """Batch loading of nodes from a CSV file"""

    # load nodes from CSV file by merging on the unique_column_name value
    query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MERGE (n:$($label) {{ {unique_column_name} : row[$unique_column_name] }})
        FOREACH (k IN $properties | SET n[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """

    results = graphdb.send_query(query, {
        "source_file": source_file,
        "label": label,
        "unique_column_name": unique_column_name,
        "properties": properties
    })
    return results

def import_nodes(node_construction: dict) -> dict:
    """Import nodes as defined by a node construction rule."""

    # create a uniqueness constraint for the unique_column
    uniqueness_result = create_uniqueness_constraint(
        node_construction["label"],
        node_construction["unique_column_name"]
    )

    if (uniqueness_result["status"] == "error"):
        return uniqueness_result

    # import nodes from csv
    load_nodes_result = load_nodes_from_csv(
        node_construction["source_file"],
        node_construction["label"],
        node_construction["unique_column_name"],
        node_construction["properties"]
    )

    return load_nodes_result

def import_relationships(relationship_construction: dict) -> Dict[str, Any]:
    """Import relationships as defined by a relationship construction rule."""

    # load nodes from CSV file by merging on the unique_column_name value 
    from_node_column = relationship_construction["from_node_column"]
    to_node_column = relationship_construction["to_node_column"]
    query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MATCH (from_node:$($from_node_label) {{ {from_node_column} : row[$from_node_column] }}),
              (to_node:$($to_node_label) {{ {to_node_column} : row[$to_node_column] }} )
        MERGE (from_node)-[r:$($relationship_type)]->(to_node)
        FOREACH (k IN $properties | SET r[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """
    
    results = graphdb.send_query(query, {
        "source_file": relationship_construction["source_file"],
        "from_node_label": relationship_construction["from_node_label"],
        "from_node_column": relationship_construction["from_node_column"],
        "to_node_label": relationship_construction["to_node_label"],
        "to_node_column": relationship_construction["to_node_column"],
        "relationship_type": relationship_construction["relationship_type"],
        "properties": relationship_construction["properties"]
    })
    return results

def construct_domain_graph(construction_plan: dict) -> Dict[str, Any]:
    """Construct a domain graph according to a construction plan."""
    # first, import nodes
    node_constructions = [value for value in construction_plan.values() if value['construction_type'] == 'node']
    for node_construction in node_constructions:
        import_nodes(node_construction)

    # second, import relationships
    relationship_constructions = [value for value in construction_plan.values() if value['construction_type'] == 'relationship']
    for relationship_construction in relationship_constructions:
        import_relationships(relationship_construction)
