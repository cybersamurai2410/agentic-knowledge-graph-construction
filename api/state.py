from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Dict
from uuid import uuid4

from fastapi import HTTPException

from api.models import RunRecord, RunStatus, RunStep


class RunStore:
    """Thread-safe in-memory run state store.

    This intentionally mirrors ADK session state style so it can be
    swapped for Redis/Postgres later without changing API contracts.
    """

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._lock = Lock()

    def create(self, initial_state: dict) -> RunRecord:
        now = datetime.now(timezone.utc)
        run_id = str(uuid4())
        run = RunRecord(
            run_id=run_id,
            status=RunStatus.created,
            current_step=RunStep.intent,
            created_at=now,
            updated_at=now,
            state=initial_state,
        )
        with self._lock:
            self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunRecord:
        run = self._runs.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return run

    def update(self, run_id: str, *, status: RunStatus | None = None, step: RunStep | None = None) -> RunRecord:
        with self._lock:
            run = self.get(run_id)
            if status:
                run.status = status
            if step:
                run.current_step = step
            run.updated_at = datetime.now(timezone.utc)
            self._runs[run_id] = run
            return run


run_store = RunStore()
