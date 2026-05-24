from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Iterator

from evolab.contracts.common import Message
from evolab.contracts.retrieval import MemoryItem


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="microseconds") + "Z"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str) -> Any:
    return json.loads(value)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def lemmatize_text(text: str) -> str:
    return " ".join(token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text))


class SQLiteMemoryStore:
    content_hash = staticmethod(content_hash)
    lemmatize_text = staticmethod(lemmatize_text)

    def __init__(self, path: str | Path, audit_log_path: str | Path | None = None):
        self.path = Path(path)
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.audit_log_path is not None:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_messages(
        self,
        scope: str,
        scope_id: str,
        messages: list[Message],
        metadata: dict[str, Any],
    ) -> None:
        created_at = _now()
        metadata_json = _json(metadata)
        audit_events = [
            ("messages.add", {"scope": scope, "scope_id": scope_id, "count": len(messages)})
        ]
        with self._connect() as conn:
            start_sequence = self._next_message_sequence(conn)
            rows = [
                (
                    str(uuid.uuid4()),
                    start_sequence + index,
                    scope,
                    scope_id,
                    message.role,
                    message.content,
                    message.name,
                    message.tool_call_id,
                    created_at,
                    metadata_json,
                )
                for index, message in enumerate(messages)
            ]
            conn.executemany(
                """
                INSERT INTO message_history(
                    message_id, sequence, scope, scope_id, role, content, name, tool_call_id,
                    created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        self._write_audit_events(audit_events)

    def save_ingest(
        self,
        scope: str,
        scope_id: str,
        messages: list[Message],
        message_metadata: dict[str, Any],
        memory_specs: list[dict[str, Any]],
    ) -> list[str]:
        created_at = _now()
        metadata_json = _json(message_metadata)
        memory_ids: list[str] = []
        audit_events: list[tuple[str, dict[str, Any]]] = []
        with self._connect() as conn:
            start_sequence = self._next_message_sequence(conn)
            rows = [
                (
                    str(uuid.uuid4()),
                    start_sequence + index,
                    scope,
                    scope_id,
                    message.role,
                    message.content,
                    message.name,
                    message.tool_call_id,
                    created_at,
                    metadata_json,
                )
                for index, message in enumerate(messages)
            ]
            conn.executemany(
                """
                INSERT INTO message_history(
                    message_id, sequence, scope, scope_id, role, content, name, tool_call_id,
                    created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            for spec in memory_specs:
                memory_ids.append(
                    self._insert_memory_record(
                        conn,
                        scope,
                        scope_id,
                        str(spec["content"]),
                        list(spec["embedding"]),
                        dict(spec["metadata"]),
                        list(spec.get("linked_memory_ids", [])),
                        list(spec.get("entities", [])),
                    )
                )
            audit_events.append(
                ("messages.add", {"scope": scope, "scope_id": scope_id, "count": len(rows)})
            )
            for memory_id in memory_ids:
                audit_events.append(
                    (
                        "memory.add",
                        {"memory_id": memory_id, "scope": scope, "scope_id": scope_id},
                    )
                )
        self._write_audit_events(audit_events)
        return memory_ids

    def recent_messages(self, scope: str, scope_id: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, name, tool_call_id, created_at, metadata_json
                FROM message_history
                WHERE scope = ? AND scope_id = ?
                ORDER BY sequence DESC, created_at DESC, message_id DESC
                LIMIT ?
                """,
                (scope, scope_id, limit),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "name": row["name"],
                "tool_call_id": row["tool_call_id"],
                "created_at": row["created_at"],
                "metadata": _loads(row["metadata_json"]),
            }
            for row in reversed(rows)
        ]

    def insert_memory(
        self,
        scope: str,
        scope_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any],
        linked_memory_ids: list[str],
        entities: list[dict[str, Any]],
    ) -> str:
        memory_id = str(uuid.uuid4())
        created_at = _now()

        with self._connect() as conn:
            memory_id = self._insert_memory_record(
                conn,
                scope,
                scope_id,
                content,
                embedding,
                metadata,
                linked_memory_ids,
                entities,
                memory_id=memory_id,
                created_at=created_at,
            )
        self._write_audit_events(
            [("memory.add", {"memory_id": memory_id, "scope": scope, "scope_id": scope_id})]
        )
        return memory_id

    def list_memories(self, scope: str, scope_id: str, include_deleted: bool = False) -> list[MemoryItem]:
        deleted_clause = "" if include_deleted else "AND deleted_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT memory_id, content, metadata_json
                FROM memory_records
                WHERE scope = ? AND scope_id = ? {deleted_clause}
                ORDER BY created_at ASC, memory_id ASC
                """,
                (scope, scope_id),
            ).fetchall()
        return [
            MemoryItem(
                memory_id=row["memory_id"],
                content=row["content"],
                metadata=_loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def memory_history(self, memory_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event, old_content, new_content, created_at, metadata_json
                FROM memory_history
                WHERE memory_id = ?
                ORDER BY created_at ASC, history_id ASC
                """,
                (memory_id,),
            ).fetchall()
        return [
            {
                "event": row["event"],
                "old_content": row["old_content"],
                "new_content": row["new_content"],
                "created_at": row["created_at"],
                "metadata": _loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def semantic_candidates(self, scope: str, scope_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, content, embedding_json, metadata_json
                FROM memory_records
                WHERE scope = ? AND scope_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC, memory_id ASC
                """,
                (scope, scope_id),
            ).fetchall()
        return [
            {
                "memory_id": row["memory_id"],
                "content": row["content"],
                "embedding": [float(value) for value in _loads(row["embedding_json"])],
                "metadata": _loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def keyword_candidates(self, scope: str, scope_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, content, text_lemmatized, metadata_json
                FROM memory_records
                WHERE scope = ? AND scope_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC, memory_id ASC
                """,
                (scope, scope_id),
            ).fetchall()
        return [
            {
                "memory_id": row["memory_id"],
                "content": row["content"],
                "text_lemmatized": row["text_lemmatized"],
                "metadata": _loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def entity_links(self, scope: str, scope_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.entity_text, e.entity_type, l.memory_id
                FROM memory_entities AS e
                JOIN entity_memory_links AS l ON e.entity_id = l.entity_id
                JOIN memory_records AS m ON l.memory_id = m.memory_id
                WHERE e.scope = ?
                    AND e.scope_id = ?
                    AND l.scope = ?
                    AND l.scope_id = ?
                    AND m.scope = ?
                    AND m.scope_id = ?
                    AND m.deleted_at IS NULL
                ORDER BY e.created_at ASC, e.entity_id ASC
                """,
                (scope, scope_id, scope, scope_id, scope, scope_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def scope_state_version(self, scope: str, scope_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM message_history WHERE scope = ? AND scope_id = ?) +
                    (SELECT COUNT(*) FROM memory_records WHERE scope = ? AND scope_id = ? AND deleted_at IS NULL)
                    AS version
                """,
                (scope, scope_id, scope, scope_id),
            ).fetchone()
        return int(row["version"])

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    text_lemmatized TEXT NOT NULL,
                    attributed_to TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS message_history (
                    message_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    name TEXT,
                    tool_call_id TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_history (
                    history_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    old_content TEXT,
                    new_content TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(memory_id) REFERENCES memory_records(memory_id)
                );

                CREATE TABLE IF NOT EXISTS memory_links (
                    link_id TEXT PRIMARY KEY,
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(source_memory_id) REFERENCES memory_records(memory_id),
                    FOREIGN KEY(target_memory_id) REFERENCES memory_records(memory_id)
                );

                CREATE TABLE IF NOT EXISTS memory_entities (
                    entity_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    entity_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entity_memory_links (
                    link_id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(entity_id) REFERENCES memory_entities(entity_id),
                    FOREIGN KEY(memory_id) REFERENCES memory_records(memory_id)
                );

                CREATE INDEX IF NOT EXISTS idx_memory_records_scope_id
                    ON memory_records(scope_id, deleted_at, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_records_scope
                    ON memory_records(scope, scope_id, deleted_at, created_at);
                CREATE INDEX IF NOT EXISTS idx_message_history_scope_id
                    ON message_history(scope_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_message_history_scope
                    ON message_history(scope, scope_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_memory_history_memory_id
                    ON memory_history(memory_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_entities_scope_id
                    ON memory_entities(scope_id, entity_text);
                CREATE INDEX IF NOT EXISTS idx_entity_memory_links_scope_id
                    ON entity_memory_links(scope_id, memory_id);
                """
            )
            self._ensure_message_sequence_column(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def _next_message_sequence(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM message_history"
        ).fetchone()
        return int(row["next_sequence"])

    def _insert_memory_record(
        self,
        conn: sqlite3.Connection,
        scope: str,
        scope_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any],
        linked_memory_ids: list[str],
        entities: list[dict[str, Any]],
        memory_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        memory_id = memory_id or str(uuid.uuid4())
        created_at = created_at or _now()
        validated_embedding = _validate_embedding_vector(embedding)
        linked_memory_id_set = set(linked_memory_ids)
        existing_linked_memory_ids = self._existing_memory_ids(
            conn,
            scope,
            scope_id,
            linked_memory_id_set,
        )
        skipped_linked_memory_ids = sorted(linked_memory_id_set - existing_linked_memory_ids)
        payload = {**metadata, "content_hash": content_hash(content)}
        if skipped_linked_memory_ids:
            payload["skipped_linked_memory_ids"] = skipped_linked_memory_ids

        conn.execute(
            """
            INSERT INTO memory_records(
                memory_id, scope, scope_id, content, content_hash, embedding_json,
                text_lemmatized, attributed_to, created_at, updated_at,
                metadata_json, deleted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                memory_id,
                scope,
                scope_id,
                content,
                payload["content_hash"],
                _json(validated_embedding),
                lemmatize_text(content),
                metadata.get("attributed_to"),
                created_at,
                created_at,
                _json(payload),
            ),
        )
        conn.execute(
            """
            INSERT INTO memory_history(
                history_id, memory_id, event, old_content, new_content,
                created_at, metadata_json
            )
            VALUES (?, ?, ?, NULL, ?, ?, ?)
            """,
            (str(uuid.uuid4()), memory_id, "ADD", content, created_at, _json(payload)),
        )
        conn.executemany(
            """
            INSERT INTO memory_links(link_id, source_memory_id, target_memory_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (str(uuid.uuid4()), memory_id, linked_memory_id, created_at)
                for linked_memory_id in linked_memory_ids
                if linked_memory_id in existing_linked_memory_ids
            ],
        )
        for entity in entities:
            entity_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO memory_entities(
                    entity_id, scope, scope_id, entity_text, entity_type,
                    created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    scope,
                    scope_id,
                    str(entity.get("entity_text", "")),
                    str(entity.get("entity_type", "")),
                    created_at,
                    _json(entity),
                ),
            )
            conn.execute(
                """
                INSERT INTO entity_memory_links(
                    link_id, entity_id, memory_id, scope, scope_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), entity_id, memory_id, scope, scope_id, created_at),
            )
        return memory_id

    def _existing_memory_ids(
        self,
        conn: sqlite3.Connection,
        scope: str,
        scope_id: str,
        memory_ids: set[str],
    ) -> set[str]:
        if not memory_ids:
            return set()
        placeholders = ", ".join("?" for _ in memory_ids)
        rows = conn.execute(
            f"""
            SELECT memory_id
            FROM memory_records
            WHERE scope = ? AND scope_id = ? AND memory_id IN ({placeholders})
            """,
            (scope, scope_id, *tuple(memory_ids)),
        ).fetchall()
        return {str(row["memory_id"]) for row in rows}

    def _ensure_message_sequence_column(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(message_history)").fetchall()
        }
        if "sequence" not in columns:
            conn.execute("ALTER TABLE message_history ADD COLUMN sequence INTEGER NOT NULL DEFAULT 0")
            conn.execute("UPDATE message_history SET sequence = rowid")

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        if self.audit_log_path is None:
            return
        record = {"event": event, "created_at": _now(), **payload}
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(_json(record) + "\n")

    def _write_audit_events(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        if self.audit_log_path is None or not events:
            return
        created_at = _now()
        records = [
            {"event": event, "created_at": created_at, **payload}
            for event, payload in events
        ]
        try:
            with self.audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write("".join(_json(record) + "\n" for record in records))
        except OSError:
            return


def _validate_embedding_vector(embedding: list[float]) -> list[float]:
    vector: list[float] = []
    for index, value in enumerate(embedding):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"embedding[{index}] must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"embedding[{index}] must be finite")
        vector.append(number)
    return vector
