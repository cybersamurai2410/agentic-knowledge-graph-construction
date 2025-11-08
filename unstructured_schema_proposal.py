from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.tools import ToolContext

from graph_utilities import graphdb, tool_success, tool_error
from tools import get_approved_user_goal, get_approved_files, sample_file
from helper import make_agent_caller

MODEL_GPT_4O = "openai/gpt-4o"
llm = LiteLlm(model=MODEL_GPT_4O)

ner_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the kind of named entities that could be extracted which would be relevant 
  for a user's goal.
  """

ner_agent_hints = """
  Entities are people, places, things and qualities, but not quantities. 
  Your goal is to propose a list of the type of entities, not the actual instances
  of entities.

  There are two general approaches to identifying types of entities:
  - well-known entities: these closely correlate with approved node labels in an existing graph schema
  - discovered entities: these may not exist in the graph schema, but appear consistently in the source text

  Design rules for well-known entities:
  - always use existing well-known entity types. For example, if there is a well-known type "Person", and people appear in the text, then propose "Person" as the type of entity.
  - prefer reusing existing entity types rather than creating new ones
  
  Design rules for discovered entities:
  - discovered entities are consistently mentioned in the text and are highly relevant to the user's goal
  - always look for entities that would provide more depth or breadth to the existing graph
  - for example, if the user goal is to represent social communities and the graph has "Person" nodes, look through the text to discover entities that are relevant like "Hobby" or "Event"
  - avoid quantitative types that may be better represented as a property on an existing entity or relationship.
  - for example, do not propose "Age" as a type of entity. That is better represented as an additional property "age" on a "Person".
"""

ner_agent_chain_of_thought_directions = """
  Prepare for the task:
  - use the 'get_user_goal' tool to get the user goal
  - use the 'get_approved_files' tool to get the list of approved files
  - use the 'get_well_known_types' tool to get the approved node labels

  Think step by step:
  1. Sample some of the files using the 'sample_file' tool to understand the content
  2. Consider what well-known entities are mentioned in the text
  3. Discover entities that are frequently mentioned in the text that support the user's goal
  4. Use the 'set_proposed_entities' tool to save the list of well-known and discovered entity types
  5. Use the 'get_proposed_entities' tool to retrieve the proposed entities and present them to the user for their approval
  6. If the user approves, use the 'approve_proposed_entities' tool to finalize the entity types
  7. If the user does not approve, consider their feedback and iterate on the proposal
"""

ner_agent_instruction = f"""
{ner_agent_role_and_goal}
{ner_agent_hints}
{ner_agent_chain_of_thought_directions}
"""

# tools to propose and approve entity types
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"

def set_proposed_entities(proposed_entity_types: list[str], tool_context:ToolContext) -> dict:
    """Sets the list proposed entity types to extract from unstructured text."""
    tool_context.state[PROPOSED_ENTITIES] = proposed_entity_types
    return tool_success(PROPOSED_ENTITIES, proposed_entity_types)

def get_proposed_entities(tool_context:ToolContext) -> dict:
    """Gets the list of proposed entity types to extract from unstructured text."""
    return tool_context.state.get(PROPOSED_ENTITIES, [])

def approve_proposed_entities(tool_context:ToolContext) -> dict:
    """Upon approval from user, records the proposed entity types as an approved list of entity types 

    Only call this tool if the user has explicitly approved the suggested files.
    """
    if PROPOSED_ENTITIES not in tool_context.state:
        return tool_error("No proposed entity types to approve. Please set proposed entities first, ask for user approval, then call this tool.")
    tool_context.state[APPROVED_ENTITIES] = tool_context.state.get(PROPOSED_ENTITIES)
    return tool_success(APPROVED_ENTITIES, tool_context.state[APPROVED_ENTITIES])

def get_approved_entities(tool_context:ToolContext) -> dict:
    """Get the approved list of entity types to extract from unstructured text."""
    return tool_context.state.get(APPROVED_ENTITIES, [])

def get_well_known_types(tool_context:ToolContext) -> dict:
    """Gets the approved labels that represent well-known entity types in the graph schema."""
    construction_plan = tool_context.state.get("approved_construction_plan", {})
    # approved labels are the keys for each construction plan entry where `construction_type` is "node"
    approved_labels = {entry["label"] for entry in construction_plan.values() if entry["construction_type"] == "node"}
    return tool_success("approved_labels", approved_labels)

ner_agent_tools = [
    get_approved_user_goal, get_approved_files, sample_file,
    get_well_known_types,
    set_proposed_entities,
    approve_proposed_entities
]

NER_AGENT_NAME = "ner_schema_agent_v1"
ner_schema_agent = Agent(
    name=NER_AGENT_NAME,
    description="Proposes the kind of named entities that could be extracted from text files.",
    model=llm,
    instruction=ner_agent_instruction,
    tools=ner_agent_tools, 
)

ner_agent_initial_state = {
    "approved_user_goal": {
        "kind_of_graph": "supply chain analysis",
        "description": """A multi-level bill of materials for manufactured products, useful for root cause analysis. 
        Add product reviews to start analysis from reported issues like quality, difficulty, or durability."""
    },
    "approved_files": [
        "product_reviews/gothenburg_table_reviews.md",
        "product_reviews/helsingborg_dresser_reviews.md",
        "product_reviews/jonkoping_coffee_table_reviews.md",
        "product_reviews/linkoping_bed_reviews.md",
        "product_reviews/malmo_desk_reviews.md",
        "product_reviews/norrkoping_nightstand_reviews.md",
        "product_reviews/orebro_lamp_reviews.md",
        "product_reviews/stockholm_chair_reviews.md",
        "product_reviews/uppsala_sofa_reviews.md",
        "product_reviews/vasteras_bookshelf_reviews.md"
    ],
    "approved_construction_plan": {
        "Product": {
            "construction_type": "node",
            "label": "Product",
        },
        "Assembly": {
            "construction_type": "node",
            "label": "Assembly",
        },
        "Part": {
            "construction_type": "node",
            "label": "Part",
        },
        "Supplier": {
            "construction_type": "node",
            "label": "Supplier",
        }
    }
}

ner_agent_caller = await make_agent_caller(ner_schema_agent, ner_agent_initial_state)

await ner_agent_caller.call("Add product reviews to the knowledge graph to trace product complaints back through the manufacturing process.")

# Alternatively, uncomment this line to get verbose output
# await ner_agent_caller.call("Add product reviews.", True)

session_end = await ner_agent_caller.get_session()

print("\n---\n")

print("\nSession state: ", session_end.state)

if PROPOSED_ENTITIES in session_end.state:
    print("\nProposed entities: ", session_end.state[PROPOSED_ENTITIES])

if APPROVED_ENTITIES in session_end.state:
    print("\nInappropriately approved entities: ", session_end.state[APPROVED_ENTITIES])
else:
    print("\nAwaiting approval.")

await ner_agent_caller.call("Approve the proposed entities.")

session_end = await ner_agent_caller.get_session()

ner_end_state = session_end.state if session_end else {}

print("Session state: ", ner_end_state)

if APPROVED_ENTITIES in ner_end_state:
    print("\nApproved entities: ", ner_end_state[APPROVED_ENTITIES])
else:
    print("\nStill awaiting approval? That is weird. Please check the agent's state and the proposed entities.")

fact_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the type of facts that could be extracted from text that would be relevant 
  for a user's goal. 
"""

fact_agent_hints = """
  Do not propose specific individual facts, but instead propose the general type 
  of facts that would be relevant for the user's goal. 
  For example, do not propose "ABK likes coffee" but the general type of fact "Person likes Beverage".
  
  Facts are triplets of (subject, predicate, object) where the subject and object are
  approved entity types, and the proposed predicate provides information about
  how they are related. For example, a fact type could be (Person, likes, Beverage).

  Design rules for facts:
  - only use approved entity types as subjects or objects. Do not propose new types of entities
  - the proposed predicate should describe the relationship between the approved subject and object
  - the predicate should optimize for information that is relevant to the user's goal
  - the predicate must appear in the source text. Do not guess.
  - use the 'add_proposed_fact' tool to record each proposed fact type
"""

fact_agent_chain_of_thought_directions = """
    Prepare for the task:
    - use the 'get_approved_user_goal' tool to get the user goal
    - use the 'get_approved_files' tool to get the list of approved files
    - use the 'get_approved_entities' tool to get the list of approved entity types

    Think step by step:
    1. Use the 'get_approved_user_goal' tool to get the user goal
    2. Sample some of the approved files using the 'sample_file' tool to understand the content
    3. Consider how subjects and objects are related in the text
    4. Call the 'add_proposed_fact' tool for each type of fact you propose
    5. Use the 'get_proposed_facts' tool to retrieve all the proposed facts
    6. Present the proposed types of facts to the user, along with an explanation
"""

fact_agent_instruction = f"""
{fact_agent_role_and_goal}
{fact_agent_hints}
{fact_agent_chain_of_thought_directions}
"""

PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"

def add_proposed_fact(approved_subject_label:str,
                      proposed_predicate_label:str,
                      approved_object_label:str,
                      tool_context:ToolContext) -> dict:
    """Add a proposed type of fact that could be extracted from the files.

    A proposed fact type is a tuple of (subject, predicate, object) where
    the subject and object are approved entity types and the predicate 
    is a proposed relationship label.

    Args:
      approved_subject_label: approved label of the subject entity type
      proposed_predicate_label: label of the predicate
      approved_object_label: approved label of the object entity type
    """
    # Guard against invalid labels
    approved_entities = tool_context.state.get(APPROVED_ENTITIES, [])
    
    if approved_subject_label not in approved_entities:
        return tool_error(f"Approved subject label {approved_subject_label} not found. Try again.")
    if approved_object_label not in approved_entities:
        return tool_error(f"Approved object label {approved_object_label} not found. Try again.")
    
    current_predicates = tool_context.state.get(PROPOSED_FACTS, {})
    current_predicates[proposed_predicate_label] = {
        "subject_label": approved_subject_label,
        "predicate_label": proposed_predicate_label,
        "object_label": approved_object_label
    }
    tool_context.state[PROPOSED_FACTS] = current_predicates
    return tool_success(PROPOSED_FACTS, current_predicates)
    
def get_proposed_facts(tool_context:ToolContext) -> dict:
    """Get the proposed types of facts that could be extracted from the files."""
    return tool_context.state.get(PROPOSED_FACTS, {})


def approve_proposed_facts(tool_context:ToolContext) -> dict:
    """Upon user approval, records the proposed fact types as approved fact types

    Only call this tool if the user has explicitly approved the proposed fact types.
    """
    if PROPOSED_FACTS not in tool_context.state:
        return tool_error("No proposed fact types to approve. Please set proposed facts first, ask for user approval, then call this tool.")
    tool_context.state[APPROVED_FACTS] = tool_context.state.get(PROPOSED_FACTS)
    return tool_success(APPROVED_FACTS, tool_context.state[APPROVED_FACTS])

fact_agent_tools = [
    get_approved_user_goal, get_approved_files, 
    get_approved_entities,
    sample_file,
    add_proposed_fact,
    get_proposed_facts,
    approve_proposed_facts
]

FACT_AGENT_NAME = "fact_type_extraction_agent_v1"
relevant_fact_agent = Agent(
    name=FACT_AGENT_NAME,
    description="Proposes the kind of relevant facts that could be extracted from text files.",
    model=llm,
    instruction=fact_agent_instruction,
    tools=fact_agent_tools, 
)

# make a copy of the NER agent's end state to use as the initial state for the fact agent
fact_agent_initial_state = ner_end_state.copy()

fact_agent_caller = await make_agent_caller(relevant_fact_agent, fact_agent_initial_state)

await fact_agent_caller.call("Propose fact types that can be found in the text.")
# await fact_agent_caller.call("Propose fact types that can be found in the text.", True)

session_end = await fact_agent_caller.get_session()

print("\n---\n")

print("\nSession state: ", session_end.state)

print("\nApproved entities: ", session_end.state.get(APPROVED_ENTITIES, []))

# Check that the agent proposed facts
if PROPOSED_FACTS in session_end.state:
    print("\nCorrectly proposed facts: ", session_end.state[PROPOSED_FACTS])
else:
    print("\nProposed facts not found in session state. What went wrong?")

# Check that the agent did not inappropriately approve facts
if APPROVED_FACTS in session_end.state:
    print("\nInappriately approved facts: ", session_end.state[APPROVED_FACTS])
else:
    print("\nApproved facts not found in session state, which is good.")

await fact_agent_caller.call("Approve the proposed fact types.")

session_end = await fact_agent_caller.get_session()

print("Session state: ", session_end.state)

if APPROVED_FACTS in session_end.state:
    print("\nApproved fact types: ", session_end.state[APPROVED_FACTS])
else:
    print("\nFailed to approve fact types.")
