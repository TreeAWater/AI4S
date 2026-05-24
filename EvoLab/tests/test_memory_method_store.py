import json
import math
import sqlite3
from pathlib import Path

import pytest

from evolab.backends.memory.methods.retrieval import cosine_similarity
from evolab.backends.memory.methods.store import SQLiteMemoryStore
from evolab.contracts.common import Message


def test_sqlite_store_persists_messages_and_memories_across_instances(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(db_path)
    store.save_messages(
        scope="agent",
        scope_id="agent:solver",
        messages=[Message(role="assistant", content="first run")],
        metadata={"run_ref": "run-1"},
    )
    memory_id = store.insert_memory(
        scope="agent",
        scope_id="agent:solver",
        content="User wants exact extraction records preserved",
        embedding=[1.0, 0.0],
        metadata={"attributed_to": "user"},
        linked_memory_ids=[],
        entities=[],
    )

    reopened = SQLiteMemoryStore(db_path)

    assert [item.content for item in reopened.list_memories("agent", "agent:solver")] == [
        "User wants exact extraction records preserved"
    ]
    assert reopened.recent_messages("agent", "agent:solver", limit=1)[0]["content"] == "first run"
    assert reopened.memory_history(memory_id)[0]["event"] == "ADD"


def test_sqlite_store_enforces_scope_isolation(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.insert_memory("agent", "agent:solver", "solver memory", [1.0], {}, [], [])
    store.insert_memory("task", "task:task-1", "task memory", [1.0], {}, [], [])

    assert [item.content for item in store.list_memories("agent", "agent:solver")] == ["solver memory"]
    assert [item.content for item in store.list_memories("task", "task:task-1")] == ["task memory"]


def test_sqlite_store_isolates_same_scope_id_across_scopes(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.save_messages(
        "agent",
        "shared",
        [Message(role="assistant", content="agent-only message")],
        {},
    )
    store.save_messages(
        "task",
        "shared",
        [Message(role="assistant", content="task-only message")],
        {},
    )
    agent_memory_id = store.insert_memory("agent", "shared", "agent memory", [1.0], {}, [], [])
    task_memory_id = store.insert_memory("task", "shared", "task memory", [1.0], {}, [], [])
    linked_agent_memory_id = store.insert_memory(
        "agent",
        "shared",
        "agent linked memory",
        [1.0],
        {},
        [task_memory_id],
        [],
    )

    assert [item.memory_id for item in store.list_memories("agent", "shared")] == [
        agent_memory_id,
        linked_agent_memory_id,
    ]
    assert [item.memory_id for item in store.list_memories("task", "shared")] == [task_memory_id]
    assert [item["content"] for item in store.recent_messages("agent", "shared", limit=5)] == [
        "agent-only message"
    ]
    assert [item["content"] for item in store.recent_messages("task", "shared", limit=5)] == [
        "task-only message"
    ]
    assert store.scope_state_version("agent", "shared") == 3
    assert store.scope_state_version("task", "shared") == 2
    with sqlite3.connect(tmp_path / "memory.sqlite") as conn:
        rows = conn.execute(
            "SELECT target_memory_id FROM memory_links WHERE source_memory_id = ?",
            (linked_agent_memory_id,),
        ).fetchall()
    assert rows == []


def test_sqlite_store_audit_log_records_events(tmp_path: Path):
    audit_path = tmp_path / "memory.audit.jsonl"
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite", audit_log_path=audit_path)

    store.insert_memory("agent", "agent:solver", "audit memory", [0.5], {}, [], [])

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["event"] == "memory.add"


def test_recent_messages_returns_last_messages_in_chronological_order(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.save_messages(
        scope="agent",
        scope_id="agent:solver",
        messages=[
            Message(role="assistant", content="first"),
            Message(role="assistant", content="second"),
            Message(role="assistant", content="third"),
        ],
        metadata={},
    )

    assert [item["content"] for item in store.recent_messages("agent", "agent:solver", limit=2)] == [
        "second",
        "third",
    ]


def test_audit_write_failure_does_not_roll_back_committed_memory_insert(tmp_path: Path):
    audit_path = tmp_path / "audit-directory"
    audit_path.mkdir()
    db_path = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(db_path, audit_log_path=audit_path)

    memory_id = store.insert_memory("agent", "agent:solver", "still durable", [1.0], {}, [], [])

    reopened = SQLiteMemoryStore(db_path)
    assert [item.memory_id for item in reopened.list_memories("agent", "agent:solver")] == [memory_id]


@pytest.mark.parametrize("embedding", [[math.nan], [math.inf], [-math.inf], ["not-a-number"]])
def test_insert_memory_rejects_non_finite_or_non_numeric_embeddings(tmp_path: Path, embedding):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    with pytest.raises(ValueError, match="embedding"):
        store.insert_memory("agent", "agent:solver", "bad vector", embedding, {}, [], [])

    assert store.list_memories("agent", "agent:solver") == []


def test_cosine_similarity_returns_zero_for_non_finite_vectors():
    assert cosine_similarity([math.nan, 0.0], [math.nan, 0.0]) == 0.0
    assert cosine_similarity([math.inf, 0.0], [math.inf, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], ["not-a-number", 0.0]) == 0.0


def test_unknown_linked_memory_ids_are_not_stored_as_dangling_links(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(db_path)

    store.insert_memory("agent", "agent:solver", "memory", [1.0], {}, ["missing"], [])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT target_memory_id FROM memory_links WHERE target_memory_id = ?",
            ("missing",),
        ).fetchall()

    assert rows == []


def test_cross_scope_linked_memory_ids_are_not_stored(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(db_path)
    target_id = store.insert_memory("task", "task:task-1", "task memory", [1.0], {}, [], [])

    store.insert_memory("agent", "agent:solver", "agent memory", [1.0], {}, [target_id], [])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT target_memory_id FROM memory_links WHERE target_memory_id = ?",
            (target_id,),
        ).fetchall()

    assert rows == []


def test_semantic_candidates_decode_embedding_and_metadata(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory_id = store.insert_memory(
        "agent",
        "agent:solver",
        "semantic memory",
        [0.25, 0.75],
        {"source": "test"},
        [],
        [],
    )

    candidates = store.semantic_candidates("agent", "agent:solver")

    assert candidates == [
        {
            "memory_id": memory_id,
            "content": "semantic memory",
            "embedding": [0.25, 0.75],
            "metadata": {
                "content_hash": SQLiteMemoryStore.content_hash("semantic memory"),
                "source": "test",
            },
        }
    ]


def test_entity_links_are_scoped(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    agent_memory_id = store.insert_memory(
        "agent",
        "agent:solver",
        "Sigma70 controls transcription",
        [1.0],
        {},
        [],
        [{"entity_text": "Sigma70", "entity_type": "PROPER"}],
    )
    store.insert_memory(
        "task",
        "task:task-1",
        "Sigma70 appears elsewhere",
        [1.0],
        {},
        [],
        [{"entity_text": "Sigma70", "entity_type": "PROPER"}],
    )

    assert store.entity_links("agent", "agent:solver") == [
        {
            "entity_text": "Sigma70",
            "entity_type": "PROPER",
            "memory_id": agent_memory_id,
        }
    ]


def test_search_candidates_are_scoped_by_scope_and_scope_id(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    agent_memory_id = store.insert_memory(
        "agent",
        "shared",
        "agent shared memory",
        [0.25, 0.75],
        {},
        [],
        [{"entity_text": "Sigma70", "entity_type": "PROPER"}],
    )
    store.insert_memory(
        "task",
        "shared",
        "task shared memory",
        [0.5, 0.5],
        {},
        [],
        [{"entity_text": "Sigma70", "entity_type": "PROPER"}],
    )

    assert [item["memory_id"] for item in store.semantic_candidates("agent", "shared")] == [
        agent_memory_id
    ]
    assert [item["memory_id"] for item in store.keyword_candidates("agent", "shared")] == [
        agent_memory_id
    ]
    assert store.entity_links("agent", "shared") == [
        {
            "entity_text": "Sigma70",
            "entity_type": "PROPER",
            "memory_id": agent_memory_id,
        }
    ]


def test_content_hash_strips_content_and_is_memory_metadata(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory_id = store.insert_memory(
        "agent",
        "agent:solver",
        "  hash me  ",
        [1.0],
        {"label": "example"},
        [],
        [],
    )

    expected_hash = SQLiteMemoryStore.content_hash("hash me")

    assert SQLiteMemoryStore.content_hash("  hash me\n") == expected_hash
    item = store.list_memories("agent", "agent:solver")[0]
    assert item.memory_id == memory_id
    assert item.metadata["content_hash"] == expected_hash
    assert item.metadata["label"] == "example"


def test_save_ingest_audit_write_failure_does_not_leave_rolled_back_audit_lines(tmp_path: Path):
    audit_path = tmp_path / "audit-directory"
    audit_path.mkdir()
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite", audit_log_path=audit_path)

    memory_ids = store.save_ingest(
        "agent",
        "agent:solver",
        [Message(role="user", content="remember atomically")],
        {},
        [
            {
                "content": "atomic memory",
                "embedding": [1.0],
                "metadata": {"attributed_to": "user"},
                "linked_memory_ids": [],
                "entities": [],
            }
        ],
    )

    reopened = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    assert [item.memory_id for item in reopened.list_memories("agent", "agent:solver")] == memory_ids
    assert reopened.recent_messages("agent", "agent:solver", limit=1)[0]["content"] == "remember atomically"


def test_scope_state_version_counts_messages_and_active_memories(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    assert store.scope_state_version("agent", "agent:solver") == 0

    store.save_ingest(
        "agent",
        "agent:solver",
        [Message(role="user", content="remember version")],
        {},
        [
            {
                "content": "version memory",
                "embedding": [1.0],
                "metadata": {"attributed_to": "user"},
                "linked_memory_ids": [],
                "entities": [],
            }
        ],
    )

    assert store.scope_state_version("agent", "agent:solver") == 2
