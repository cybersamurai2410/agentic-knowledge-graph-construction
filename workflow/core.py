from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


PERCEIVED_USER_GOAL = "perceived_user_goal"
APPROVED_USER_GOAL = "approved_user_goal"
ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"
PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"


def set_perceived_user_goal(state: Dict[str, Any], kind_of_graph: str, graph_description: str) -> Dict[str, Any]:
    data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    state[PERCEIVED_USER_GOAL] = data
    return data


def approve_perceived_user_goal(state: Dict[str, Any]) -> Dict[str, Any]:
    if PERCEIVED_USER_GOAL not in state:
        raise ValueError("perceived_user_goal not set")
    state[APPROVED_USER_GOAL] = state[PERCEIVED_USER_GOAL]
    return state[APPROVED_USER_GOAL]


def list_available_files(import_dir: Path) -> List[str]:
    return sorted(str(x.relative_to(import_dir)) for x in import_dir.rglob("*") if x.is_file())


def suggest_files(files: List[str], include_extensions: List[str], contains_any: List[str]) -> List[str]:
    allowed = {ext.lower() for ext in include_extensions}
    filters = [f.lower() for f in contains_any]

    out: List[str] = []
    for file_name in files:
        lower = file_name.lower()
        ext = Path(lower).suffix
        ext_ok = (not allowed) or (ext in allowed)
        text_ok = (not filters) or any(k in lower for k in filters)
        if ext_ok and text_ok:
            out.append(file_name)
    return out


def set_suggested_files(state: Dict[str, Any], files: List[str]) -> List[str]:
    state[SUGGESTED_FILES] = files
    return state[SUGGESTED_FILES]


def approve_suggested_files(state: Dict[str, Any], approved_files: List[str] | None = None) -> List[str]:
    if approved_files is not None:
        state[APPROVED_FILES] = approved_files
        return approved_files

    if SUGGESTED_FILES not in state:
        raise ValueError("suggested_files not set")

    state[APPROVED_FILES] = state[SUGGESTED_FILES]
    return state[APPROVED_FILES]


def _count_distinct_values(csv_path: Path, column_name: str) -> tuple[int, int]:
    total = 0
    unique_values = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            value = row.get(column_name)
            if value is not None and value != "":
                unique_values.add(value)

    return total, len(unique_values)


def propose_structured_schema(import_dir: Path, approved_files: List[str]) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}

    for rel_path in approved_files:
        csv_path = import_dir / rel_path
        if not csv_path.exists() or csv_path.suffix.lower() != ".csv":
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []

        if not fields:
            continue

        id_like = [f for f in fields if f.lower().endswith("_id")]

        # Better-than-heuristic: if exactly one *_id column and values are unique => node candidate.
        if len(id_like) == 1:
            total, distinct = _count_distinct_values(csv_path, id_like[0])
            if total > 0 and total == distinct:
                stem = csv_path.stem
                label = stem[:-1].title() if stem.endswith("s") else stem.title()
                plan[label] = {
                    "construction_type": "node",
                    "source_file": rel_path,
                    "label": label,
                    "unique_column_name": id_like[0],
                    "properties": [c for c in fields if c != id_like[0]],
                }
                continue

        # Relationship candidate: at least two id-like columns.
        if len(id_like) >= 2:
            rel_type = csv_path.stem.upper()
            plan[rel_type] = {
                "construction_type": "relationship",
                "source_file": rel_path,
                "relationship_type": rel_type,
                "from_node_label": id_like[0].replace("_id", "").title(),
                "from_node_column": id_like[0],
                "to_node_label": id_like[1].replace("_id", "").title(),
                "to_node_column": id_like[1],
                "properties": [c for c in fields if c not in id_like],
            }
            continue

        # Fallback node rule.
        unique_col = fields[0]
        stem = csv_path.stem
        label = stem[:-1].title() if stem.endswith("s") else stem.title()
        plan[label] = {
            "construction_type": "node",
            "source_file": rel_path,
            "label": label,
            "unique_column_name": unique_col,
            "properties": [c for c in fields if c != unique_col],
        }

    return plan


def approve_structured_schema(state: Dict[str, Any], approved_plan: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    state[APPROVED_CONSTRUCTION_PLAN] = approved_plan
    return state[APPROVED_CONSTRUCTION_PLAN]
