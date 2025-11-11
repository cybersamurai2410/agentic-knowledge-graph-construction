from google.adk.models.lite_llm import LiteLlm 
from graph_utilities import graphdb, tool_success, tool_error
from typing import Dict, Any
from helper import get_neo4j_import_dir
from rapidfuzz import fuzz

from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from neo4j_graphrag.experimental.components.types import PdfDocument, DocumentInfo
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

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

# Define a custom text splitter. Chunking strategy could be yet-another-agent
class RegexTextSplitter(TextSplitter):
    """Split text using regex matched delimiters."""
    def __init__(self, re: str):
        self.re = re
    
    async def run(self, text: str) -> TextChunks:
        """Splits a piece of text into chunks.

        Args:
            text (str): The text to be split.

        Returns:
            TextChunks: A list of chunks.
        """
        texts = re.split(self.re, text)
        i = 0
        chunks = [TextChunk(text=str(text), index=i) for (i, text) in enumerate(texts)]
        return TextChunks(chunks=chunks)

class MarkdownDataLoader(DataLoader):
    def extract_title(self,markdown_text):
        # Define a regex pattern to match the first h1 header
        pattern = r'^# (.+)$'

        # Search for the first match in the markdown text
        match = re.search(pattern, markdown_text, re.MULTILINE)

        # Return the matched group if found
        return match.group(1) if match else "Untitled"

    async def run(self, filepath: Path, metadata = {}) -> PdfDocument:
        with open(filepath, "r") as f:
            markdown_text = f.read()
        doc_headline = self.extract_title(markdown_text)
        markdown_info = DocumentInfo(
            path=str(filepath),
            metadata={
                "title": doc_headline,
            }
        )
        return PdfDocument(text=markdown_text, document_info=markdown_info)

# create an OpenAI client for use by Neo4j GraphRAG
llm_for_neo4j = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# use OpenAI for creating embeddings
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# use the same driver set up by neo4j_for_adk.py
neo4j_driver = graphdb.get_driver()

schema_node_types = approved_entities
schema_relationship_types = [key.upper() for key in approved_fact_types.keys()]
schema_patterns = [
    [ fact['subject_label'], fact['predicate_label'].upper(), fact['object_label'] ]
    for fact in approved_fact_types.values()
]

# the complete entity schema
entity_schema = {
    "node_types": schema_node_types,
    "relationship_types": schema_relationship_types,
    "patterns": schema_patterns,
    "additional_node_types": False, # True would be less strict, allowing unknown node types
}

def file_context(file_path:str, num_lines=5) -> str:
    """Helper function to extract the first few lines of a file

    Args:
        file_path (str): Path to the file
        num_lines (int, optional): Number of lines to extract. Defaults to 5.

    Returns:
        str: First few lines of the file
    """
    with open(file_path, 'r') as f:
        lines = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line)
    return "\n".join(lines)

# per-chunk entity extraction prompt, with context
def contextualize_er_extraction_prompt(context:str) -> str:
    """Creates a prompt with pre-amble file content for context during entity+relationship extraction.
    The context is concatenated into the string, which later will be used as a template
    for values like {schema} and {text}.
    """
    general_instructions = """
    You are a top-tier algorithm designed for extracting
    information in structured formats to build a knowledge graph.

    Extract the entities (nodes) and specify their type from the following text.
    Also extract the relationships between these nodes.

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
    "relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

    Use only the following node and relationship types (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Make sure you adhere to the following rules to produce valid JSON objects:
    - Do not return any additional information other than the JSON in it.
    - Omit any backticks around the JSON - simply output the JSON on its own.
    - The JSON object must not wrapped into a list - it is its own JSON object.
    - Property names must be enclosed in double quotes
    """

    context_goes_here = f"""
    Consider the following context to help identify entities and relationships:
    <context>
    {context}  
    </context>"""
    
    input_goes_here = """
    Input text:

    {text}
    """

    return general_instructions + "\n" + context_goes_here + "\n" + input_goes_here

def make_kg_builder(file_path:str) -> SimpleKGPipeline:
    """Builds a KG builder for a given file, which is used to contextualize the chunking and entity extraction."""
    context = file_context(file_path)
    contextualized_prompt = contextualize_er_extraction_prompt(context)

    return SimpleKGPipeline(
        llm=llm_for_neo4j, # the LLM to use for Entity and Relation extraction
        driver=neo4j_driver,  # a neo4j driver to write results to graph
        embedder=embedder,  # an Embedder for chunks
        from_pdf=True,   # sortof True because you will use a custom loader
        pdf_loader=MarkdownDataLoader(), # the custom loader for Markdown
        text_splitter=RegexTextSplitter("---"), # the splitter you defined above
        schema=entity_schema, # that you just defined above
        prompt_template=contextualized_prompt,
    )

neo4j_import_dir = get_neo4j_import_dir() or "."

for file_name in approved_files:
    file_path = os.path.join(neo4j_import_dir, file_name)
    print(f"Processing file: {file_name}")
    kg_builder = make_kg_builder(file_path)
    results = await kg_builder.run_async(file_path=str(file_path))
    print("\tResults:", results.result)
print("All files processed.")

def find_unique_entity_labels():
    result = graphdb.send_query("""MATCH (n)
        WHERE n:`__Entity__`
        WITH DISTINCT labels(n) AS entity_labels
        UNWIND entity_labels AS entity_label
        WITH entity_label
        WHERE NOT entity_label STARTS WITH "__"
        RETURN collect(entity_label) as unique_entity_labels
        """)
    if result['status'] == 'error':
        raise Exception(result['message'])
    return result['query_result'][0]['unique_entity_labels']

unique_entity_labels = find_unique_entity_labels()

def find_unique_entity_keys(entityLabel:str):
    result = graphdb.send_query("""MATCH (n:$($entityLabel))
    WHERE n:`__Entity__`
    WITH DISTINCT keys(n) as entityKeys
    UNWIND entityKeys as entityKey
    RETURN collect(distinct(entityKey)) as unique_entity_keys
    """, {
        "entityLabel": entityLabel
    })
    if result['status'] == 'error':
        raise Exception(result['message'])
    return result['query_result'][0]['unique_entity_keys']

def find_unique_domain_keys(domainLabel:str):
    result = graphdb.send_query("""MATCH (n:$($domainLabel))
    WHERE NOT n:`__Entity__` // exclude entities created by the KG builder, these should be domain nodes
    WITH DISTINCT keys(n) as domainKeys
    UNWIND domainKeys as domainKey
    RETURN collect(distinct(domainKey)) as unique_domain_keys
    """, {
        "domainLabel": domainLabel
    })
    if result['status'] == 'error':
        raise Exception(result['message'])
    return result['query_result'][0]['unique_domain_keys']

def normalize_key(label:str, key:str) -> str:
    """Normalizes a a property key for a given label.

    Keys are normalized by:
    - lowercase the key
    - remove any leading/trailing whitespace
    - remove label prefix from key
    - replace internal whitespace with "_"

    for example: 
        - "Product_name" -> "name"
        - "product name" -> "name"
        - "price" -> "price

    Args:
        label (str): The label to normalize keys for
        keys (List[str]): The list of keys to normalize

    Returns:
        List[str]: The normalized list of keys
    """
    lowercase_key = key.lower()
    unprefixed_key = re.sub(f"^{label.lower()}[_ ]*", "", lowercase_key)
    normalized_key = re.sub(" ", "_", unprefixed_key)
    return normalized_key

# for a given label, get pairs of entity and domain keys that correlate
def correlate_entity_and_domain_keys(label: str, entity_keys: list[str], domain_keys: list[str], similarity: float = 0.9) -> list[tuple[str, str]]:
    correlated_keys = []
    for entity_key in entity_keys:
        for domain_key in domain_keys:
            # only consider exact matches. this could use fuzzy matching
            normalized_entity_key = normalize_key(label, entity_key)
            normalized_domain_key = normalize_key(label, domain_key)
            # rapidfuzz similarity is 0.0 -> 100.0, so divide by 100 for 0.0 -> 1.0
            fuzzy_similarity = (fuzz.ratio(normalized_entity_key, normalized_domain_key) / 100)
            if (fuzzy_similarity > similarity): 
                correlated_keys.append((entity_key, domain_key, fuzzy_similarity))
    correlated_keys.sort(key=lambda x: x[2], reverse=True)
    return correlated_keys

# use the Jaro-Winkler function to calculate distance between product names
results = graphdb.send_query("""
// MATCH all pairs of subject and domain nodes -- this is an expensive cartesian product
MATCH (entity:$($entityLabel):`__Entity__`), (domain:$($entityLabel))
WITH entity, domain, apoc.text.jaroWinklerDistance(entity[$entityKey], domain[$domainKey]) as score
// experiment with different thresholds to see how the results change
WHERE score < 0.4
RETURN entity[$entityKey] AS entityValue, domain[$domainKey] AS domainValue, score
// experiment with different limits to see more or fewer pairs
LIMIT 3
""", {
    "entityLabel": "Product",
    "entityKey": "name",
    "domainKey": "product_name"
})

results['query_result']

# connect all corresponding nodes with a relationship
results = graphdb.send_query("""
MATCH (entity:$($entityLabel):`__Entity__`),(domain:$($entityLabel))
// use the score as a predicate to filter the pairs. this is better
WHERE apoc.text.jaroWinklerDistance(entity[$entityKey], domain[$domainKey]) < 0.1
MERGE (entity)-[r:CORRESPONDS_TO]->(domain)
ON CREATE SET r.created_at = datetime()
ON MATCH SET r.updated_at = datetime()
RETURN elementId(entity) as entity_id, r, elementId(domain) as domain_id
""", {
    "entityLabel": "Product",
    "entityKey": "name",
    "domainKey": "product_name"
})

results['query_result']

def correlate_subject_and_domain_nodes(label: str, entity_key: str, domain_key: str, similarity: float = 0.9) -> dict:
    """Correlate entity and domain nodes based on label, entity key, and domain key,
    where the corresponding values of the entity and domain properties are similar
    
    For example, if you have a label "Person" and an entity key "name", and a domain key "person_name",
    this function will create a relationship like:
    (:Person:`__Entity__` {name: "John"})-[:CORRELATES_TO]->(:Person {person_name: "John"}) 
    

    Args:
        label (str): The label of the entity and domain nodes.
        entity_key (str): The key of the entity node.
        domain_key (str): The key of the domain node.
        similarity (float, optional): The similarity threshold for correlation. Defaults to 0.9.
    
    Returns:
        dict: A dictionary containing the correlation between the entity and domain nodes.
    """
    results = graphdb.send_query("""
    MATCH (entity:$($entityLabel):`__Entity__`),(domain:$($entityLabel))
    WHERE apoc.text.jaroWinklerDistance(entity[$entityKey], domain[$domainKey]) < $distance
    MERGE (entity)-[r:CORRESPONDS_TO]->(domain)
    ON CREATE SET r.created_at = datetime() // MERGE sub-clause when the relationship is newly created
    ON MATCH SET r.updated_at = datetime()  // MERGE sub-clause when the relationship already exists
    RETURN $entityLabel as entityLabel, count(r) as relationshipCount
    """, {
        "entityLabel": label,
        "entityKey": entity_key,
        "domainKey": domain_key,
        "distance": (1.0 - similarity)
    })

    if results['status'] == 'error':
        raise Exception(results['message'])

    return results['query_result']

# - loop over all entity labels
# - correlate the keys
# - correlate (and connect) the nodes
for entity_label in find_unique_entity_labels():
    print(f"Correlating entities labeled {entity_label}...")
    
    entity_keys = find_unique_entity_keys(entity_label)
    domain_keys = find_unique_domain_keys(entity_label)

    correlated_keys = correlate_entity_and_domain_keys(entity_label, entity_keys, domain_keys, similarity=0.8)

    if (len(correlated_keys) > 0):
        top_correlated_keypair = correlated_keys[0]
        print("\tbased on:", top_correlated_keypair)
        correlate_subject_and_domain_nodes(entity_label, top_correlated_keypair[0], top_correlated_keypair[1])
    else:
        print("\tNo correlation found")
        
