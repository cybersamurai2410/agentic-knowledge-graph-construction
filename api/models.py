from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    created = "created"
    awaiting_approval = "awaiting_approval"
    ready_for_construction = "ready_for_construction"
    running = "running"
    completed = "completed"
    failed = "failed"


class RunStep(str, Enum):
    intent = "intent"
    file_selection = "file_selection"
    structured_schema = "structured_schema"
    unstructured_schema = "unstructured_schema"
    construction = "construction"


class ApiResponse(BaseModel):
    status: str = "success"
    message: Optional[str] = None


class RunCreateRequest(BaseModel):
    kind_of_graph: str = Field(..., min_length=2, max_length=120)
    graph_description: str = Field(..., min_length=10, max_length=4000)


class RunRecord(BaseModel):
    run_id: str
    status: RunStatus
    current_step: RunStep
    created_at: datetime
    updated_at: datetime
    state: Dict[str, Any]


class PerceivedIntentRequest(BaseModel):
    kind_of_graph: str = Field(..., min_length=2, max_length=120)
    graph_description: str = Field(..., min_length=10, max_length=4000)


class FileSuggestionRequest(BaseModel):
    include_extensions: List[str] = Field(default_factory=lambda: [".csv", ".json", ".md"])
    contains_any: List[str] = Field(default_factory=list)


class FileApprovalRequest(BaseModel):
    approved_files: List[str]


class NodeConstruction(BaseModel):
    construction_type: str = "node"
    source_file: str
    label: str
    unique_column_name: str
    properties: List[str]


class RelationshipConstruction(BaseModel):
    construction_type: str = "relationship"
    source_file: str
    relationship_type: str
    from_node_label: str
    from_node_column: str
    to_node_label: str
    to_node_column: str
    properties: List[str]


class StructuredSchemaProposalResponse(BaseModel):
    proposed_construction_plan: Dict[str, Dict[str, Any]]


class StructuredSchemaApprovalRequest(BaseModel):
    approved_construction_plan: Dict[str, Dict[str, Any]]


class Neo4jQueryResponse(BaseModel):
    status: str
    query_result: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
