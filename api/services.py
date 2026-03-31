from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from graph_utilities import graphdb
from helper import get_neo4j_import_dir


def import_dir() -> Path:
    root = get_neo4j_import_dir()
    if not root:
        raise HTTPException(status_code=500, detail="NEO4J_IMPORT_DIR is not configured")
    p = Path(root)
    if not p.exists():
        raise HTTPException(status_code=500, detail=f"NEO4J_IMPORT_DIR does not exist: {p}")
    return p


def list_available_files() -> List[str]:
    root = import_dir()
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def suggest_files(files: List[str], include_extensions: List[str], contains_any: List[str]) -> List[str]:
    allowed = {ext.lower() for ext in include_extensions}
    needle = [q.lower() for q in contains_any]

    suggested: List[str] = []
    for name in files:
        lower_name = name.lower()
        suffix = Path(lower_name).suffix
        ext_ok = (not allowed) or suffix in allowed
        contains_ok = (not needle) or any(q in lower_name for q in needle)
        if ext_ok and contains_ok:
            suggested.append(name)
    return suggested


def sample_file_lines(rel_path: str, max_rows: int = 5) -> List[Dict[str, str]]:
    root = import_dir()
    full = root / rel_path
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {rel_path}")

    if full.suffix.lower() != ".csv":
        return []

    with open(full, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for _, row in zip(range(max_rows), reader)]


def propose_structured_schema(approved_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Heuristic schema proposal from CSV headers.

    - singular/plural file stem -> node label
    - *_id first column -> unique id
    - two *_id columns -> relationship
    """
    plan: Dict[str, Dict[str, Any]] = {}
    root = import_dir()

    for rel in approved_files:
        full = root / rel
        if full.suffix.lower() != ".csv" or not full.exists():
            continue

        with open(full, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []

        id_fields = [c for c in fields if c.lower().endswith("_id")]
        stem = full.stem

        if len(id_fields) >= 2:
            rel_type = stem.upper()
            plan[rel_type] = {
                "construction_type": "relationship",
                "source_file": rel,
                "relationship_type": rel_type,
                "from_node_label": id_fields[0].replace("_id", "").title(),
                "from_node_column": id_fields[0],
                "to_node_label": id_fields[1].replace("_id", "").title(),
                "to_node_column": id_fields[1],
                "properties": [c for c in fields if c not in id_fields],
            }
            continue

        unique_column = id_fields[0] if id_fields else (fields[0] if fields else "id")
        label = stem[:-1].title() if stem.endswith("s") else stem.title()
        plan[label] = {
            "construction_type": "node",
            "source_file": rel,
            "label": label,
            "unique_column_name": unique_column,
            "properties": [c for c in fields if c != unique_column],
        }

    return plan


def create_uniqueness_constraint(label: str, unique_property_key: str) -> Dict[str, Any]:
    constraint_name = f"{label}_{unique_property_key}_constraint"
    query = f"""CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS
    FOR (n:`{label}`)
    REQUIRE n.`{unique_property_key}` IS UNIQUE"""
    return graphdb.send_query(query)


def load_nodes_from_csv(source_file: str, label: str, unique_column_name: str, properties: List[str]) -> Dict[str, Any]:
    query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MERGE (n:$($label) {{ {unique_column_name} : row[$unique_column_name] }})
        FOREACH (k IN $properties | SET n[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """
    return graphdb.send_query(
        query,
        {
            "source_file": source_file,
            "label": label,
            "unique_column_name": unique_column_name,
            "properties": properties,
        },
    )


def load_relationships_from_csv(construction: Dict[str, Any]) -> Dict[str, Any]:
    from_node_column = construction["from_node_column"]
    to_node_column = construction["to_node_column"]
    query = f"""LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MATCH (from_node:$($from_node_label) {{ {from_node_column} : row[$from_node_column] }}),
              (to_node:$($to_node_label) {{ {to_node_column} : row[$to_node_column] }})
        MERGE (from_node)-[r:$($relationship_type)]->(to_node)
        FOREACH (k IN $properties | SET r[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """
    return graphdb.send_query(
        query,
        {
            "source_file": construction["source_file"],
            "from_node_label": construction["from_node_label"],
            "from_node_column": construction["from_node_column"],
            "to_node_label": construction["to_node_label"],
            "to_node_column": construction["to_node_column"],
            "relationship_type": construction["relationship_type"],
            "properties": construction["properties"],
        },
    )


def construct_domain_graph(approved_construction_plan: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    node_results = []
    relationship_results = []

    nodes = [v for v in approved_construction_plan.values() if v.get("construction_type") == "node"]
    relationships = [v for v in approved_construction_plan.values() if v.get("construction_type") == "relationship"]

    for n in nodes:
        unique_result = create_uniqueness_constraint(n["label"], n["unique_column_name"])
        if unique_result.get("status") == "error":
            return unique_result
        load_result = load_nodes_from_csv(n["source_file"], n["label"], n["unique_column_name"], n["properties"])
        if load_result.get("status") == "error":
            return load_result
        node_results.append(load_result)

    for r in relationships:
        load_result = load_relationships_from_csv(r)
        if load_result.get("status") == "error":
            return load_result
        relationship_results.append(load_result)

    return {
        "status": "success",
        "node_import_results": node_results,
        "relationship_import_results": relationship_results,
    }


def neo4j_health() -> Dict[str, Any]:
    return graphdb.send_query("RETURN 'Neo4j is Ready!' as message")


def clear_neo4j_data() -> Dict[str, Any]:
    return graphdb.send_query("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")


def drop_neo4j_indexes_and_constraints() -> Dict[str, Any]:
    constraints = graphdb.send_query("SHOW CONSTRAINTS YIELD name")
    if constraints.get("status") == "error":
        return constraints

    for row in constraints.get("query_result", []):
        result = graphdb.send_query("DROP CONSTRAINT `" + row["name"] + "`")
        if result.get("status") == "error":
            return result

    indexes = graphdb.send_query("SHOW INDEXES YIELD name")
    if indexes.get("status") == "error":
        return indexes

    for row in indexes.get("query_result", []):
        result = graphdb.send_query("DROP INDEX `" + row["name"] + "`")
        if result.get("status") == "error":
            return result

    return {"status": "success", "message": "Dropped all indexes and constraints."}
