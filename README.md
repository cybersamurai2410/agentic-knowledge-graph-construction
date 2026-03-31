# Agentic Knowledge Graph Construction with GraphRAG for Supply Chain Analysis
Multi-agent system for constructing knowledge graphs representing supply chain manufacturing networks for Q&A using GraphRAG.

<img width="1855" height="970" alt="entire_solution" src="https://github.com/user-attachments/assets/aae3eadb-0822-431e-ba5d-0d8fe22e2e42" />

## Knowledge Graph Agent
- Top-level conversational agent
- Responsible for overall interaction with the user
- Guides the user through major phases

  - Structured data agent
    - Workflow agent
    - Data import from CSV files
    - Delegates to sub-agents
    - User intent agent
      - Conversational agent — collaborates with the user to determine the goal for the data import
    - File suggestion agent
      - Tool-use agent — analyzes and suggests relevant CSV files
    - Schema proposal agent
      - A pair of agents in the "Critic Pattern" (proposal + critic)
      - Iteratively refines an appropriate graph schema
    - Graph construction plan
      - Output from the structured data workflow
      - Approved construction rules for turning CSVs into a graph

  - Unstructured data agent
    - Workflow agent
    - Data import from Markdown
    - Delegates to sub-agents
    - User intent & file suggestion agents
    - Entity & fact type proposal agent
      - Tool-use agent — collaborates with the user to determine entity types and relevant facts extractable from the Markdown
    - Knowledge extraction plan
      - Output from the unstructured data workflow
      - Approved entity types
      - Approved facts about entities

  - GraphRAG agent
    - Tool-use agent
    - Chooses retrieval strategy to answer questions
    - Knowledge graph construction tool — executes graph construction and knowledge extraction:
      1. Loop over construction rules to produce a domain graph
      2. Loop over Markdown files to chunk and extract entities and facts
      3. Connect extracted entities to defined domain entities

## API
HTTP API wrapper for the workflow is defined in `api/main.py`.

### Run the API
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Main endpoints
- `POST /v1/runs` create a workflow run.
- `POST /v1/runs/{run_id}/intent/perceive` set perceived intent.
- `POST /v1/runs/{run_id}/intent/approve` approve perceived intent.
- `GET /v1/runs/{run_id}/files/available` list files from `NEO4J_IMPORT_DIR`.
- `POST /v1/runs/{run_id}/files/suggest` heuristically suggest files.
- `POST /v1/runs/{run_id}/files/approve` approve files.
- `POST /v1/runs/{run_id}/schema/structured/propose` propose construction plan.
- `POST /v1/runs/{run_id}/schema/structured/approve` approve construction plan.
- `POST /v1/runs/{run_id}/graph/construct` execute graph import against Neo4j.
- `GET /v1/neo4j/health` verify Neo4j connectivity.
- `POST /v1/neo4j/clear` clear graph data.
- `POST /v1/neo4j/drop-indexes` drop constraints and indexes.
