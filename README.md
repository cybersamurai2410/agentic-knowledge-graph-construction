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
- `POST /v1/graphrag/ask` ask a supply-chain question; response includes evidence-grounded summary plus `llm_answer` generated from retrieved graph context.
- `GET /v1/neo4j/health` verify Neo4j connectivity.
- `POST /v1/neo4j/clear` clear graph data.
- `POST /v1/neo4j/drop-indexes` drop constraints and indexes.

### Example terminal run
```bash
$ uvicorn api.main:app --reload
INFO:     Will watch for changes in these directories: ['/workspace/agentic-knowledge-graph-construction']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [21432] using StatReload
INFO:     Started server process [21434]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

$ curl -s -X POST http://127.0.0.1:8000/v1/runs \
  -H 'content-type: application/json' \
  -d '{"kind_of_graph":"supply chain","graph_description":"multi-level BOM for root-cause analysis"}' | jq
{
  "run_id": "2de5f3ab-1fd0-4ab1-b8d8-83c7e7b1576a",
  "status": "created",
  "current_step": "intent",
  "created_at": "2026-03-31T13:22:10.941782+00:00",
  "updated_at": "2026-03-31T13:22:10.941782+00:00",
  "state": {
    "approved_user_goal": {
      "kind_of_graph": "supply chain",
      "graph_description": "multi-level BOM for root-cause analysis"
    }
  }
}

$ curl -s http://127.0.0.1:8000/v1/runs/2de5f3ab-1fd0-4ab1-b8d8-83c7e7b1576a/files/available | jq
{
  "run_id": "2de5f3ab-1fd0-4ab1-b8d8-83c7e7b1576a",
  "all_available_files": [
    "products.csv",
    "assemblies.csv",
    "parts.csv",
    "suppliers.csv",
    "supplier_parts.csv"
  ]
}

$ curl -s -X POST http://127.0.0.1:8000/v1/graphrag/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Which supplier parts are connected to products with durability complaints?","top_k":5}' | jq
{
  "answer": "Based on the graph evidence, relevant entities include: Product(product_id=P-101, product_name=Stockholm Chair); Supplier(supplier_id=S-008, supplier_name=Nordic Components); Part(part_id=PT-442, part_name=Chair Leg). Observed relationships: Product -[USES_PART]- Part(part_id=PT-442, part_name=Chair Leg); Supplier -[SUPPLIES]- Part(part_id=PT-442, part_name=Chair Leg).",
  "llm_answer": "Products with durability complaints are linked to supplier-delivered parts through Product-USES_PART-Part and Supplier-SUPPLIES-Part relationships. In the retrieved evidence, Chair Leg (PT-442) is supplied by Nordic Components (S-008) and used by Stockholm Chair (P-101), making it a high-priority part for root-cause investigation.",
  "llm_used": true,
  "retrieved_count": 5,
  "evidence": [
    {
      "node_labels": ["Product"],
      "node_properties": {"product_id": "P-101", "product_name": "Stockholm Chair"},
      "relationship_type": "USES_PART",
      "neighbor_labels": ["Part"],
      "neighbor_properties": {"part_id": "PT-442", "part_name": "Chair Leg"}
    }
  ]
}

$ curl -s http://127.0.0.1:8000/v1/neo4j/health | jq
{
  "status": "success",
  "query_result": [
    {
      "message": "Neo4j is Ready!"
    }
  ]
}
```
