from __future__ import annotations

from fastapi import FastAPI, HTTPException

from api.models import (
    ApiResponse,
    FileApprovalRequest,
    FileSuggestionRequest,
    HealthResponse,
    PerceivedIntentRequest,
    RunCreateRequest,
    RunStep,
    RunStatus,
    StructuredSchemaApprovalRequest,
    StructuredSchemaProposalResponse,
)
from api.services import (
    clear_neo4j_data,
    construct_domain_graph,
    drop_neo4j_indexes_and_constraints,
    list_available_files,
    neo4j_health,
    propose_structured_schema,
    suggest_files,
)
from api.state import run_store

app = FastAPI(
    title="Agentic Knowledge Graph Construction API",
    version="1.0.0",
    description="REST API wrapper around the agentic knowledge-graph construction workflow.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="agentic-kg-api")


@app.post("/v1/runs")
def create_run(payload: RunCreateRequest):
    run = run_store.create(
        {
            "approved_user_goal": {
                "kind_of_graph": payload.kind_of_graph,
                "graph_description": payload.graph_description,
            }
        }
    )
    return run


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str):
    return run_store.get(run_id)


@app.get("/v1/runs/{run_id}/state")
def get_run_state(run_id: str):
    run = run_store.get(run_id)
    return {"run_id": run_id, "state": run.state}


@app.post("/v1/runs/{run_id}/intent/perceive")
def perceive_intent(run_id: str, payload: PerceivedIntentRequest):
    run = run_store.get(run_id)
    run.state["perceived_user_goal"] = {
        "kind_of_graph": payload.kind_of_graph,
        "graph_description": payload.graph_description,
    }
    run_store.update(run_id, status=RunStatus.awaiting_approval, step=RunStep.intent)
    return {"run_id": run_id, "perceived_user_goal": run.state["perceived_user_goal"]}


@app.post("/v1/runs/{run_id}/intent/approve", response_model=ApiResponse)
def approve_intent(run_id: str):
    run = run_store.get(run_id)
    if "perceived_user_goal" not in run.state:
        raise HTTPException(status_code=400, detail="No perceived_user_goal to approve.")
    run.state["approved_user_goal"] = run.state["perceived_user_goal"]
    run_store.update(run_id, status=RunStatus.created, step=RunStep.file_selection)
    return ApiResponse(message="approved_user_goal recorded")


@app.get("/v1/runs/{run_id}/files/available")
def get_available_files(run_id: str):
    run = run_store.get(run_id)
    files = list_available_files()
    run.state["all_available_files"] = files
    return {"run_id": run_id, "all_available_files": files}


@app.post("/v1/runs/{run_id}/files/suggest")
def suggest_run_files(run_id: str, payload: FileSuggestionRequest):
    run = run_store.get(run_id)
    files = run.state.get("all_available_files") or list_available_files()
    suggested = suggest_files(files, payload.include_extensions, payload.contains_any)
    run.state["suggested_files"] = suggested
    run_store.update(run_id, status=RunStatus.awaiting_approval, step=RunStep.file_selection)
    return {"run_id": run_id, "suggested_files": suggested}


@app.post("/v1/runs/{run_id}/files/approve", response_model=ApiResponse)
def approve_run_files(run_id: str, payload: FileApprovalRequest):
    run = run_store.get(run_id)
    run.state["approved_files"] = payload.approved_files
    run_store.update(run_id, status=RunStatus.created, step=RunStep.structured_schema)
    return ApiResponse(message="approved_files recorded")


@app.post("/v1/runs/{run_id}/schema/structured/propose", response_model=StructuredSchemaProposalResponse)
def propose_structured(run_id: str):
    run = run_store.get(run_id)
    approved_files = run.state.get("approved_files", [])
    if not approved_files:
        raise HTTPException(status_code=400, detail="No approved_files found. Approve files first.")
    plan = propose_structured_schema(approved_files)
    run.state["proposed_construction_plan"] = plan
    run_store.update(run_id, status=RunStatus.awaiting_approval, step=RunStep.structured_schema)
    return StructuredSchemaProposalResponse(proposed_construction_plan=plan)


@app.post("/v1/runs/{run_id}/schema/structured/approve", response_model=ApiResponse)
def approve_structured(run_id: str, payload: StructuredSchemaApprovalRequest):
    run = run_store.get(run_id)
    run.state["approved_construction_plan"] = payload.approved_construction_plan
    run_store.update(run_id, status=RunStatus.ready_for_construction, step=RunStep.construction)
    return ApiResponse(message="approved_construction_plan recorded")


@app.post("/v1/runs/{run_id}/graph/construct")
def construct_graph(run_id: str):
    run = run_store.get(run_id)
    plan = run.state.get("approved_construction_plan")
    if not plan:
        raise HTTPException(status_code=400, detail="No approved_construction_plan found.")

    run_store.update(run_id, status=RunStatus.running, step=RunStep.construction)
    result = construct_domain_graph(plan)

    if result.get("status") == "error":
        run_store.update(run_id, status=RunStatus.failed, step=RunStep.construction)
        return result

    run.state["construction_result"] = result
    run_store.update(run_id, status=RunStatus.completed, step=RunStep.construction)
    return {"run_id": run_id, **result}


@app.get("/v1/neo4j/health")
def neo4j_is_ready():
    return neo4j_health()


@app.post("/v1/neo4j/clear")
def neo4j_clear_data():
    return clear_neo4j_data()


@app.post("/v1/neo4j/drop-indexes")
def neo4j_drop_indexes():
    return drop_neo4j_indexes_and_constraints()
