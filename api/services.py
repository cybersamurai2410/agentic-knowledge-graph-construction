from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib import request, error

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


def retrieve_graph_context(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve graph evidence relevant to a natural-language question."""
    query = """
    MATCH (n)
    WHERE any(k IN keys(n) WHERE toLower(toString(n[k])) CONTAINS toLower($q))
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN labels(n) AS node_labels, properties(n) AS node_properties,
           type(r) AS relationship_type, labels(m) AS neighbor_labels, properties(m) AS neighbor_properties
    LIMIT $top_k
    """
    return graphdb.send_query(query, {"q": question, "top_k": top_k})


def _build_graph_context(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        n_labels = "/".join(row.get("node_labels") or ["Node"])
        n_props = row.get("node_properties") or {}
        rel = row.get("relationship_type") or "RELATED_TO"
        m_labels = "/".join(row.get("neighbor_labels") or ["Node"])
        m_props = row.get("neighbor_properties") or {}
        lines.append(f"{n_labels} {n_props} -[{rel}]- {m_labels} {m_props}")
    return "\n".join(lines[:30])


def _generate_llm_answer(question: str, graph_context: str) -> tuple[str, bool]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "OpenAI API key is not configured. Returning evidence-grounded summary instead of model-generated answer.",
            False,
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You answer supply-chain questions using only provided graph evidence. Be concise and explicit.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nGraph evidence:\n{graph_context}",
            },
        ],
        "temperature": 0.2,
    }

    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            message = body["choices"][0]["message"]["content"]
            return message.strip(), True
    except error.HTTPError as exc:
        return f"LLM request failed with HTTP {exc.code}. Returning evidence-grounded summary.", False
    except Exception:
        return "LLM request failed. Returning evidence-grounded summary.", False


def answer_question_with_graph(question: str, top_k: int = 5) -> Dict[str, Any]:
    result = retrieve_graph_context(question, top_k)
    if result.get("status") == "error":
        return result

    rows = result.get("query_result", [])
    if not rows:
        return {
            "status": "success",
            "answer": "I could not find matching graph evidence for that question yet.",
            "llm_answer": "I could not find enough graph evidence to answer confidently.",
            "llm_used": False,
            "evidence": [],
            "retrieved_count": 0,
        }

    focus = []
    relation_lines = []
    for row in rows:
        labels = row.get("node_labels") or []
        props = row.get("node_properties") or {}
        rel_type = row.get("relationship_type")
        neighbor_labels = row.get("neighbor_labels") or []
        neighbor_props = row.get("neighbor_properties") or {}

        label_text = "/".join(labels) if labels else "Node"
        if props:
            keyvals = ", ".join(f"{k}={v}" for k, v in list(props.items())[:3])
            focus.append(f"{label_text}({keyvals})")
        else:
            focus.append(label_text)

        if rel_type:
            nb_text = "/".join(neighbor_labels) if neighbor_labels else "Node"
            if neighbor_props:
                nkeyvals = ", ".join(f"{k}={v}" for k, v in list(neighbor_props.items())[:2])
                nb_text = f"{nb_text}({nkeyvals})"
            relation_lines.append(f"{label_text} -[{rel_type}]- {nb_text}")

    focus_summary = "; ".join(dict.fromkeys(focus))
    relationship_summary = "; ".join(dict.fromkeys(relation_lines[:5]))

    answer = (
        f"Based on the graph evidence, relevant entities include: {focus_summary}. "
        f"Observed relationships: {relationship_summary if relationship_summary else 'no direct relationships in retrieved sample'}."
    )

    graph_context = _build_graph_context(rows)
    llm_answer, llm_used = _generate_llm_answer(question, graph_context)

    return {
        "status": "success",
        "answer": answer,
        "llm_answer": llm_answer if llm_used else answer,
        "llm_used": llm_used,
        "evidence": rows,
        "retrieved_count": len(rows),
    }

