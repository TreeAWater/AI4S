from __future__ import annotations

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
from urllib.parse import quote, unquote, urlparse

from evolab.backends.memory.methods.base import (
    MemoryIngestRequest,
    MemoryIngestResult,
    MemorySearchRequest,
    MemorySearchResult,
)
from evolab.backends.memory.methods.retrieval import (
    bm25_keyword_scores,
    cosine_similarity,
)
from evolab.backends.memory.methods.store import content_hash, lemmatize_text
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig
from evolab.contracts.retrieval import MemoryItem


EVEROS_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "memcells": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "episode": {"type": "string"},
                    "salience": {"type": "number"},
                    "atomic_facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "entities": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["text", "entities"],
                        },
                    },
                    "foresights": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "evidence": {"type": "string"},
                                "validity": {"type": "string"},
                            },
                            "required": ["text", "evidence", "validity"],
                        },
                    },
                    "agent_case": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "task_intent": {"type": "string"},
                                    "approach": {"type": "string"},
                                    "key_insight": {"type": "string"},
                                    "quality_score": {"type": "number"},
                                },
                                "required": [
                                    "task_intent",
                                    "approach",
                                    "key_insight",
                                    "quality_score",
                                ],
                            },
                            {"type": "null"},
                        ],
                    },
                    "agent_skills": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "content": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["name", "description", "content", "confidence"],
                        },
                    },
                },
                "required": [
                    "summary",
                    "episode",
                    "salience",
                    "atomic_facts",
                    "foresights",
                    "agent_case",
                    "agent_skills",
                ],
            },
        }
    },
    "required": ["memcells"],
}


EVEROS_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "tags"],
}


EVEROS_RECOLLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "scene_id": {"type": "string"},
                    "selected_memory_ids": {"type": "array", "items": {"type": "string"}},
                    "reconstructed_context": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "scene_id",
                    "selected_memory_ids",
                    "reconstructed_context",
                    "rationale",
                ],
            },
        }
    },
    "required": ["selected_scenes"],
}


EVEROS_EXTRACTION_PROMPT = """You are EvoLab's native EverOS memory construction method.
Construct durable memory from an agent/task trajectory using EverOS-style memory:
- MemCells: coherent event units from the new messages.
- Atomic facts: concise facts useful for exact future recall.
- Foresights: future constraints, expectations, or likely follow-up needs.
- Agent cases/skills: reusable task-solving experience when the trajectory contains one.

Only store durable, reusable knowledge. Do not store boilerplate, transient prompt text,
or unsupported guesses. Return JSON only."""


EVEROS_SCENE_PROMPT = """You are EvoLab's native EverOS semantic consolidation method.
Given one new MemCell and an optional existing MemScene, produce the consolidated
scene title, scene summary, and short tags. The summary must preserve specific
scientific, workflow, and agent-experience details that will help later tasks.
Return JSON only."""


EVEROS_RECOLLECTION_PROMPT = """You are EvoLab's native EverOS reconstructive recollection method.
Select the memories that are necessary and sufficient for the query. Prefer compact
scene-grounded context over long raw dumps. Do not invent information. Return JSON only."""


class EverOSMemoryMethod:
    method_name = "everos"

    def __init__(
        self,
        store_path: str | Path,
        llm_backend_id: str,
        embedding_backend_id: str,
        audit_log_path: str | Path | None = None,
        scene_similarity_threshold: float = 0.78,
        extraction_recent_message_limit: int = 20,
        max_scene_candidates: int = 8,
        recollection_mode: str = "scene",
        recollection_candidate_limit: int = 16,
        state_ref_backend_id: str = "everos",
    ):
        if recollection_mode not in {"scene", "agentic"}:
            raise ValueError("recollection_mode must be 'scene' or 'agentic'")
        self.store = EverOSSQLiteStore(store_path, audit_log_path=audit_log_path)
        self.llm_backend_id = llm_backend_id
        self.embedding_backend_id = embedding_backend_id
        self.scene_similarity_threshold = scene_similarity_threshold
        self.extraction_recent_message_limit = extraction_recent_message_limit
        self.max_scene_candidates = max_scene_candidates
        self.recollection_mode = recollection_mode
        self.recollection_candidate_limit = recollection_candidate_limit
        self.state_ref_backend_id = state_ref_backend_id
        self.state_ref: str | None = None
        self._versions_by_scope: dict[tuple[str, str], int] = {}
        self._legacy_versions_by_scope_id: dict[str, int] = {}
        self.llm_runtime: Any | None = None
        self.embedding_runtime: Any | None = None

    def bind_backend_id(self, backend_id: str) -> None:
        self.state_ref_backend_id = backend_id

    def bind_runtimes(self, *, llm_runtimes: dict[str, Any], embedding_runtimes: dict[str, Any]) -> None:
        try:
            self.llm_runtime = llm_runtimes[self.llm_backend_id]
        except KeyError as exc:
            raise RuntimeError(f"missing LLM runtime for backend id {self.llm_backend_id!r}") from exc
        try:
            self.embedding_runtime = embedding_runtimes[self.embedding_backend_id]
        except KeyError as exc:
            raise RuntimeError(
                f"missing embedding runtime for backend id {self.embedding_backend_id!r}"
            ) from exc

    def instantiate(self, state_ref: str | None) -> "EverOSMemoryMethod":
        self.state_ref = state_ref
        parsed_state_ref = _parse_state_ref(state_ref)
        if parsed_state_ref is not None:
            scope, scope_id, version = parsed_state_ref
            if scope is None:
                self._legacy_versions_by_scope_id[scope_id] = max(
                    self._legacy_versions_by_scope_id.get(scope_id, 0),
                    version,
                )
            else:
                key = (scope, scope_id)
                self._versions_by_scope[key] = max(
                    self._versions_by_scope.get(key, 0),
                    version,
                )
        return self

    def add(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        self._validate_scope(request.scope, request.scope_id)
        self._require_dependencies()
        previous_version = self._version(request.scope, request.scope_id)
        previous_state_ref = self._state_ref(request.scope, request.scope_id)

        if not request.messages:
            return MemoryIngestResult(
                status="skipped",
                state_ref=previous_state_ref,
                previous_state_ref=previous_state_ref,
                metadata={"memcell_count": 0, "record_count": 0, "reason": "no_messages"},
            )

        recent_messages = self.store.recent_messages(
            request.scope,
            request.scope_id,
            limit=self.extraction_recent_message_limit,
        )
        scene_summaries = self.store.list_memscenes(
            request.scope,
            request.scope_id,
            limit=self.max_scene_candidates,
        )
        response = self.llm_runtime.generate(
            self._extraction_messages(request, recent_messages, scene_summaries),
            [],
            LLMGenerationConfig(
                model="",
                temperature=0,
                response_json_schema=EVEROS_EXTRACTION_SCHEMA,
            ),
        )
        content = response.action.content if response.action.action == "final_answer" else None
        try:
            memcells = _parse_extraction(content or "")
        except ValueError as exc:
            return MemoryIngestResult(
                status="failed",
                state_ref=previous_state_ref,
                previous_state_ref=previous_state_ref,
                metadata={"error_type": "everos_extraction_parse", "error": str(exc)},
            )

        if not memcells:
            try:
                self.store.save_messages(
                    request.scope,
                    request.scope_id,
                    request.messages,
                    self._message_metadata(request),
                )
            except Exception as exc:
                return _failed_result(previous_state_ref, "store_error", exc)
            self._mark_scope_changed(request.scope, request.scope_id, previous_version)
            return MemoryIngestResult(
                status="updated",
                state_ref=self._state_ref(request.scope, request.scope_id),
                previous_state_ref=previous_state_ref,
                metadata={"memcell_count": 0, "record_count": 0, "extracted_count": 0},
            )

        try:
            memcell_embeddings = self._embed(
                [_memcell_embedding_text(memcell) for memcell in memcells],
                purpose="everos:memcell",
            )
        except Exception as exc:
            return _failed_result(previous_state_ref, "embedding_error", exc)

        existing_scenes = self.store.list_memscenes(request.scope, request.scope_id)
        planned_scenes = [dict(scene) for scene in existing_scenes]
        plans: list[dict[str, Any]] = []
        degraded_errors: list[str] = []
        existing_hashes = self.store.record_content_hashes(request.scope, request.scope_id)

        for memcell, memcell_embedding in zip(memcells, memcell_embeddings):
            matched_scene = _best_scene_match(planned_scenes, memcell_embedding)
            if matched_scene is not None and matched_scene["score"] < self.scene_similarity_threshold:
                matched_scene = None
            scene_input = matched_scene["scene"] if matched_scene is not None else None
            try:
                scene_payload = self._consolidate_scene(memcell, scene_input)
            except Exception as exc:
                scene_payload = _fallback_scene_payload(memcell, scene_input)
                degraded_errors.append(str(exc))
            scene_id = scene_input["scene_id"] if scene_input is not None else str(uuid.uuid4())
            prior_count = int(scene_input.get("member_count", 0)) if scene_input is not None else 0
            scene_embedding = (
                _merged_embedding(scene_input["embedding"], prior_count, memcell_embedding)
                if scene_input is not None
                else memcell_embedding
            )
            scene_plan = {
                "scene_id": scene_id,
                "is_new": scene_input is None,
                "title": scene_payload["title"],
                "summary": scene_payload["summary"],
                "tags": scene_payload["tags"],
                "embedding": scene_embedding,
                "member_count": prior_count + 1,
                "metadata": {
                    "memory_method": self.method_name,
                    "last_memcell_summary": memcell["summary"],
                },
            }
            _upsert_planned_scene(planned_scenes, scene_plan)
            records = _record_specs_from_memcell(
                memcell=memcell,
                scene_id=scene_id,
                scope=request.scope,
                scope_id=request.scope_id,
                request=request,
                existing_hashes=existing_hashes,
            )
            plans.append(
                {
                    "memcell": memcell,
                    "memcell_embedding": memcell_embedding,
                    "scene": scene_plan,
                    "records": records,
                }
            )

        record_texts = [
            record["content"]
            for plan in plans
            for record in plan["records"]
        ]
        try:
            record_embeddings = self._embed(record_texts, purpose="everos:record") if record_texts else []
        except Exception as exc:
            return _failed_result(previous_state_ref, "embedding_error", exc)
        embedding_index = 0
        for plan in plans:
            for record in plan["records"]:
                record["embedding"] = record_embeddings[embedding_index]
                embedding_index += 1

        try:
            save_result = self.store.save_ingest(
                request.scope,
                request.scope_id,
                request.messages,
                self._message_metadata(request),
                plans,
            )
        except Exception as exc:
            return _failed_result(previous_state_ref, "store_error", exc)

        self._mark_scope_changed(request.scope, request.scope_id, previous_version)
        status = "degraded" if degraded_errors else "updated"
        return MemoryIngestResult(
            status=status,
            state_ref=self._state_ref(request.scope, request.scope_id),
            previous_state_ref=previous_state_ref,
            added_memory_ids=save_result["record_ids"],
            linked_memory_ids=save_result["scene_ids"],
            metadata={
                "memcell_count": len(save_result["memcell_ids"]),
                "scene_count": len(set(save_result["scene_ids"])),
                "record_count": len(save_result["record_ids"]),
                "duplicate_count": save_result["duplicate_count"],
                "recollection_mode": self.recollection_mode,
                "degraded_errors": degraded_errors,
            },
        )

    def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        self._validate_search_scope(request)
        self._require_dependencies()
        top_k = 10 if request.top_k is None else request.top_k
        threshold = 0.05 if request.threshold is None else request.threshold
        query_embedding = self._embed([request.query], purpose="everos:search")[0]
        records = self.store.searchable_records(request.scope, request.scope_id)
        scenes = self.store.list_memscenes(request.scope, request.scope_id)
        if not records and not scenes:
            return MemorySearchResult(
                items=[],
                state_ref=self._state_ref(request.scope, request.scope_id),
                metadata={
                    "memory_method": self.method_name,
                    "candidate_count": 0,
                    "returned_count": 0,
                    "recollection_mode": self.recollection_mode,
                },
            )

        keyword_scores = bm25_keyword_scores(request.query, records)
        record_scores = self._record_scores(
            request=request,
            query_embedding=query_embedding,
            records=records,
            keyword_scores=keyword_scores,
        )
        ranked_scene_contexts = _rank_scene_contexts(
            scenes=scenes,
            records=records,
            record_scores=record_scores,
            query_embedding=query_embedding,
            threshold=threshold,
            limit=max(top_k, 1),
        )
        if self.recollection_mode == "agentic" and ranked_scene_contexts:
            items, metadata = self._agentic_recollection(
                request=request,
                ranked_scene_contexts=ranked_scene_contexts,
                top_k=top_k,
            )
        else:
            items = [
                _scene_context_memory_item(context)
                for context in ranked_scene_contexts[:top_k]
            ]
            metadata = {"recollection_mode": "scene"}

        return MemorySearchResult(
            items=items,
            state_ref=self._state_ref(request.scope, request.scope_id),
            metadata={
                **metadata,
                "memory_method": self.method_name,
                "candidate_count": len(records),
                "scene_candidate_count": len(scenes),
                "returned_count": len(items),
                "threshold": threshold,
                "top_k": top_k,
            },
        )

    def _record_scores(
        self,
        *,
        request: MemorySearchRequest,
        query_embedding: list[float],
        records: list[dict[str, Any]],
        keyword_scores: dict[str, float],
    ) -> dict[str, float]:
        query_entities = {
            entity["entity_text"].lower()
            for entity in _entities_from_text(request.query)
            if entity.get("entity_text")
        }
        scores: dict[str, float] = {}
        for record in records:
            semantic = cosine_similarity(query_embedding, record["embedding"])
            keyword = keyword_scores.get(record["memory_id"], 0.0)
            entity_boost = _entity_overlap_boost(query_entities, record["entities"])
            scores[record["memory_id"]] = _everos_fuse_score(semantic, keyword, entity_boost)
        return scores

    def _agentic_recollection(
        self,
        *,
        request: MemorySearchRequest,
        ranked_scene_contexts: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[MemoryItem], dict[str, Any]]:
        candidate_contexts = ranked_scene_contexts[: self.recollection_candidate_limit]
        response = self.llm_runtime.generate(
            [
                Message(role="system", content=EVEROS_RECOLLECTION_PROMPT),
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "query": request.query,
                            "top_k": top_k,
                            "candidate_scenes": [
                                _scene_context_for_llm(context)
                                for context in candidate_contexts
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            ],
            [],
            LLMGenerationConfig(
                model="",
                temperature=0,
                response_json_schema=EVEROS_RECOLLECTION_SCHEMA,
            ),
        )
        content = response.action.content if response.action.action == "final_answer" else None
        try:
            payload = _parse_recollection(content or "")
        except ValueError as exc:
            return (
                [_scene_context_memory_item(context) for context in ranked_scene_contexts[:top_k]],
                {
                    "recollection_mode": "agentic",
                    "recollection_status": "fallback",
                    "recollection_error": str(exc),
                },
            )
        contexts_by_id = {context["scene"]["scene_id"]: context for context in candidate_contexts}
        items: list[MemoryItem] = []
        for selection in payload[:top_k]:
            scene_id = selection["scene_id"]
            context = contexts_by_id.get(scene_id)
            if context is None:
                continue
            items.append(_agentic_scene_memory_item(context, selection))
        return items, {"recollection_mode": "agentic", "recollection_status": "selected"}

    def _consolidate_scene(
        self,
        memcell: dict[str, Any],
        scene: dict[str, Any] | None,
    ) -> dict[str, Any]:
        response = self.llm_runtime.generate(
            [
                Message(role="system", content=EVEROS_SCENE_PROMPT),
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "existing_scene": _scene_context_for_llm({"scene": scene, "records": []})
                            if scene is not None
                            else None,
                            "new_memcell": memcell,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            ],
            [],
            LLMGenerationConfig(
                model="",
                temperature=0,
                response_json_schema=EVEROS_SCENE_SCHEMA,
            ),
        )
        content = response.action.content if response.action.action == "final_answer" else None
        return _parse_scene(content or "")

    def _extraction_messages(
        self,
        request: MemoryIngestRequest,
        recent_messages: list[dict[str, Any]],
        scene_summaries: list[dict[str, Any]],
    ) -> list[Message]:
        payload = {
            "scope": request.scope,
            "scope_id": request.scope_id,
            "task_id": request.task_id,
            "role": request.role,
            "recent_messages": [
                {"role": item.get("role"), "content": item.get("content")}
                for item in recent_messages
            ],
            "existing_scenes": [
                {
                    "scene_id": scene["scene_id"],
                    "title": scene["title"],
                    "summary": scene["summary"],
                    "tags": scene["tags"],
                }
                for scene in scene_summaries
            ],
            "new_messages": [
                {"role": message.role, "content": message.content, "metadata": message.metadata}
                for message in request.messages
            ],
        }
        return [
            Message(role="system", content=EVEROS_EXTRACTION_PROMPT),
            Message(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        ]

    def _embed(self, texts: list[str], *, purpose: str) -> list[list[float]]:
        response = self.embedding_runtime.embed(texts, purpose=purpose)
        if len(response.vectors) != len(texts):
            raise ValueError(f"expected {len(texts)} embedding vectors, got {len(response.vectors)}")
        for index, vector in enumerate(response.vectors):
            if not _finite_vector(vector):
                raise ValueError(f"embedding vector at index {index} is not finite")
        return response.vectors

    def _validate_search_scope(self, request: MemorySearchRequest) -> None:
        self._validate_scope(request.scope, request.scope_id)
        supported_filters = {"memory_scope", "memory_scope_id"}
        unsupported_filters = sorted(set(request.filters) - supported_filters)
        if unsupported_filters:
            raise ValueError(
                "EverOSMemoryMethod.search does not support filters: "
                + ", ".join(unsupported_filters)
            )
        memory_scope = request.filters.get("memory_scope")
        if memory_scope is not None and memory_scope != request.scope:
            raise ValueError("memory_scope filter conflicts with request scope")
        memory_scope_id = request.filters.get("memory_scope_id")
        if memory_scope_id is not None and memory_scope_id != request.scope_id:
            raise ValueError("memory_scope_id filter conflicts with request scope_id")

    def _validate_scope(self, scope: str, scope_id: str) -> None:
        if scope not in {"agent", "task"}:
            raise ValueError("scope must be 'agent' or 'task'")
        if not scope_id:
            raise ValueError("scope_id must be a non-empty string")

    def _require_dependencies(self) -> None:
        missing = []
        if self.llm_runtime is None:
            missing.append(f"llm runtime {self.llm_backend_id!r}")
        if self.embedding_runtime is None:
            missing.append(f"embedding runtime {self.embedding_backend_id!r}")
        if missing:
            raise RuntimeError(
                "EverOSMemoryMethod requires bind_runtimes before use; missing "
                + ", ".join(missing)
            )

    def _message_metadata(self, request: MemoryIngestRequest) -> dict[str, Any]:
        return {
            **request.metadata,
            "task_id": request.task_id,
            "role": request.role,
            "memory_scope": request.scope,
            "memory_scope_id": request.scope_id,
            "memory_method": self.method_name,
        }

    def _state_ref(self, scope: str, scope_id: str) -> str:
        return (
            "method://everos/"
            f"{_scope_ref_part(self.state_ref_backend_id)}/"
            f"{_scope_ref_part(scope)}/"
            f"{_scope_ref_part(scope_id)}/"
            f"v{self._version(scope, scope_id)}"
        )

    def _version(self, scope: str, scope_id: str) -> int:
        key = (scope, scope_id)
        return max(
            self._versions_by_scope.get(key, 0),
            self._legacy_versions_by_scope_id.get(scope_id, 0),
            self.store.scope_state_version(scope, scope_id),
        )

    def _mark_scope_changed(self, scope: str, scope_id: str, previous_version: int) -> None:
        key = (scope, scope_id)
        self._versions_by_scope[key] = max(
            self._versions_by_scope.get(key, 0),
            previous_version + 1,
        )


class EverOSSQLiteStore:
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
        with self._connect() as conn:
            self._insert_messages(conn, scope, scope_id, messages, metadata, created_at)
            self._increment_scope_version(conn, scope, scope_id)
        self._write_audit_events([("everos.messages.add", {"scope": scope, "scope_id": scope_id})])

    def save_ingest(
        self,
        scope: str,
        scope_id: str,
        messages: list[Message],
        metadata: dict[str, Any],
        plans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        created_at = _now()
        memcell_ids: list[str] = []
        record_ids: list[str] = []
        scene_ids: list[str] = []
        duplicate_count = 0
        audit_events: list[tuple[str, dict[str, Any]]] = []
        with self._connect() as conn:
            self._insert_messages(conn, scope, scope_id, messages, metadata, created_at)
            for plan in plans:
                scene = plan["scene"]
                scene_ids.append(scene["scene_id"])
                if scene["is_new"]:
                    self._insert_scene(conn, scope, scope_id, scene, created_at)
                else:
                    self._update_scene(conn, scope, scope_id, scene, created_at)
                memcell_id = self._insert_memcell(
                    conn,
                    scope,
                    scope_id,
                    scene["scene_id"],
                    plan["memcell"],
                    plan["memcell_embedding"],
                    created_at,
                )
                memcell_ids.append(memcell_id)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO everos_scene_members(scene_id, memcell_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (scene["scene_id"], memcell_id, created_at),
                )
                for record in plan["records"]:
                    if record.pop("duplicate", False):
                        duplicate_count += 1
                        continue
                    record_id = self._insert_record(
                        conn,
                        scope,
                        scope_id,
                        scene["scene_id"],
                        memcell_id,
                        record,
                        created_at,
                    )
                    record_ids.append(record_id)
            self._increment_scope_version(conn, scope, scope_id)
            audit_events.append(
                (
                    "everos.ingest",
                    {
                        "scope": scope,
                        "scope_id": scope_id,
                        "memcell_count": len(memcell_ids),
                        "record_count": len(record_ids),
                    },
                )
            )
        self._write_audit_events(audit_events)
        return {
            "memcell_ids": memcell_ids,
            "scene_ids": scene_ids,
            "record_ids": record_ids,
            "duplicate_count": duplicate_count,
        }

    def recent_messages(self, scope: str, scope_id: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, name, tool_call_id, created_at, metadata_json
                FROM everos_messages
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

    def list_memscenes(
        self,
        scope: str,
        scope_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        limit_clause = "" if limit is None else "LIMIT ?"
        params: tuple[Any, ...] = (scope, scope_id) if limit is None else (scope, scope_id, limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT scene_id, title, summary, tags_json, embedding_json, member_count,
                       metadata_json, created_at, updated_at
                FROM everos_memscenes
                WHERE scope = ? AND scope_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC, scene_id ASC
                {limit_clause}
                """,
                params,
            ).fetchall()
        return [
            {
                "scene_id": row["scene_id"],
                "title": row["title"],
                "summary": row["summary"],
                "tags": _loads(row["tags_json"]),
                "embedding": _loads(row["embedding_json"]),
                "member_count": row["member_count"],
                "metadata": _loads(row["metadata_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def searchable_records(self, scope: str, scope_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, scene_id, memcell_id, record_type, content, content_hash,
                       text_lemmatized, embedding_json, metadata_json, entities_json, created_at
                FROM everos_memory_records
                WHERE scope = ? AND scope_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC, memory_id ASC
                """,
                (scope, scope_id),
            ).fetchall()
        return [
            {
                "memory_id": row["memory_id"],
                "scene_id": row["scene_id"],
                "memcell_id": row["memcell_id"],
                "record_type": row["record_type"],
                "content": row["content"],
                "content_hash": row["content_hash"],
                "text_lemmatized": row["text_lemmatized"],
                "embedding": _loads(row["embedding_json"]),
                "metadata": _loads(row["metadata_json"]),
                "entities": _loads(row["entities_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_content_hashes(self, scope: str, scope_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content_hash
                FROM everos_memory_records
                WHERE scope = ? AND scope_id = ? AND deleted_at IS NULL
                """,
                (scope, scope_id),
            ).fetchall()
        return {row["content_hash"] for row in rows}

    def scope_state_version(self, scope: str, scope_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT version FROM everos_scope_versions
                WHERE scope = ? AND scope_id = ?
                """,
                (scope, scope_id),
            ).fetchone()
        return int(row["version"]) if row is not None else 0

    def _insert_messages(
        self,
        conn: sqlite3.Connection,
        scope: str,
        scope_id: str,
        messages: list[Message],
        metadata: dict[str, Any],
        created_at: str,
    ) -> None:
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
                _json(metadata),
            )
            for index, message in enumerate(messages)
        ]
        conn.executemany(
            """
            INSERT INTO everos_messages(
                message_id, sequence, scope, scope_id, role, content, name,
                tool_call_id, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_scene(
        self,
        conn: sqlite3.Connection,
        scope: str,
        scope_id: str,
        scene: dict[str, Any],
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO everos_memscenes(
                scene_id, scope, scope_id, title, summary, tags_json, embedding_json,
                member_count, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scene["scene_id"],
                scope,
                scope_id,
                scene["title"],
                scene["summary"],
                _json(scene["tags"]),
                _json(scene["embedding"]),
                scene["member_count"],
                _json(scene["metadata"]),
                created_at,
                created_at,
            ),
        )

    def _update_scene(
        self,
        conn: sqlite3.Connection,
        scope: str,
        scope_id: str,
        scene: dict[str, Any],
        updated_at: str,
    ) -> None:
        conn.execute(
            """
            UPDATE everos_memscenes
            SET title = ?, summary = ?, tags_json = ?, embedding_json = ?,
                member_count = ?, metadata_json = ?, updated_at = ?
            WHERE scope = ? AND scope_id = ? AND scene_id = ?
            """,
            (
                scene["title"],
                scene["summary"],
                _json(scene["tags"]),
                _json(scene["embedding"]),
                scene["member_count"],
                _json(scene["metadata"]),
                updated_at,
                scope,
                scope_id,
                scene["scene_id"],
            ),
        )

    def _insert_memcell(
        self,
        conn: sqlite3.Connection,
        scope: str,
        scope_id: str,
        scene_id: str,
        memcell: dict[str, Any],
        embedding: list[float],
        created_at: str,
    ) -> str:
        memcell_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO everos_memcells(
                memcell_id, scope, scope_id, scene_id, summary, episode, salience,
                payload_json, embedding_json, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memcell_id,
                scope,
                scope_id,
                scene_id,
                memcell["summary"],
                memcell["episode"],
                memcell["salience"],
                _json(memcell),
                _json(embedding),
                _json({"memory_method": "everos"}),
                created_at,
            ),
        )
        return memcell_id

    def _insert_record(
        self,
        conn: sqlite3.Connection,
        scope: str,
        scope_id: str,
        scene_id: str,
        memcell_id: str,
        record: dict[str, Any],
        created_at: str,
    ) -> str:
        memory_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO everos_memory_records(
                memory_id, scope, scope_id, scene_id, memcell_id, record_type,
                content, content_hash, text_lemmatized, embedding_json, metadata_json,
                entities_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                scope,
                scope_id,
                scene_id,
                memcell_id,
                record["record_type"],
                record["content"],
                record["content_hash"],
                lemmatize_text(record["content"]),
                _json(record["embedding"]),
                _json(record["metadata"]),
                _json(record["entities"]),
                created_at,
            ),
        )
        return memory_id

    def _increment_scope_version(self, conn: sqlite3.Connection, scope: str, scope_id: str) -> None:
        conn.execute(
            """
            INSERT INTO everos_scope_versions(scope, scope_id, version)
            VALUES (?, ?, 1)
            ON CONFLICT(scope, scope_id)
            DO UPDATE SET version = version + 1
            """,
            (scope, scope_id),
        )

    def _next_message_sequence(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM everos_messages").fetchone()
        return int(row["next_sequence"])

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS everos_messages(
                    message_id TEXT PRIMARY KEY,
                    sequence INTEGER NOT NULL UNIQUE,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    name TEXT,
                    tool_call_id TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS everos_memscenes(
                    scene_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    member_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS everos_memcells(
                    memcell_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    episode TEXT NOT NULL,
                    salience REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT,
                    FOREIGN KEY(scene_id) REFERENCES everos_memscenes(scene_id)
                );

                CREATE TABLE IF NOT EXISTS everos_scene_members(
                    scene_id TEXT NOT NULL,
                    memcell_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(scene_id, memcell_id),
                    FOREIGN KEY(scene_id) REFERENCES everos_memscenes(scene_id),
                    FOREIGN KEY(memcell_id) REFERENCES everos_memcells(memcell_id)
                );

                CREATE TABLE IF NOT EXISTS everos_memory_records(
                    memory_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    memcell_id TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    text_lemmatized TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT,
                    FOREIGN KEY(scene_id) REFERENCES everos_memscenes(scene_id),
                    FOREIGN KEY(memcell_id) REFERENCES everos_memcells(memcell_id)
                );

                CREATE TABLE IF NOT EXISTS everos_scope_versions(
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    PRIMARY KEY(scope, scope_id)
                );

                CREATE INDEX IF NOT EXISTS idx_everos_messages_scope
                    ON everos_messages(scope, scope_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_everos_scenes_scope
                    ON everos_memscenes(scope, scope_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_everos_memcells_scope
                    ON everos_memcells(scope, scope_id, scene_id);
                CREATE INDEX IF NOT EXISTS idx_everos_records_scope
                    ON everos_memory_records(scope, scope_id, scene_id, record_type);
                CREATE INDEX IF NOT EXISTS idx_everos_records_hash
                    ON everos_memory_records(scope, scope_id, content_hash);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _write_audit_events(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        if self.audit_log_path is None:
            return
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            for event, payload in events:
                handle.write(
                    json.dumps(
                        {
                            "created_at": _now(),
                            "event": event,
                            "payload": payload,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def _parse_extraction(content: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid EverOS extraction JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("memcells"), list):
        raise ValueError("invalid EverOS extraction JSON: expected object with memcells list")
    memcells = []
    for index, raw in enumerate(payload["memcells"]):
        if not isinstance(raw, dict):
            raise ValueError(f"invalid EverOS memcell at index {index}: expected object")
        summary = _string(raw.get("summary")).strip()
        episode = _string(raw.get("episode")).strip()
        if not summary and not episode:
            continue
        memcells.append(
            {
                "summary": summary or _truncate(episode, 180),
                "episode": episode or summary,
                "salience": _bounded_float(raw.get("salience"), default=0.5),
                "atomic_facts": _parse_atomic_facts(raw.get("atomic_facts")),
                "foresights": _parse_foresights(raw.get("foresights")),
                "agent_case": _parse_agent_case(raw.get("agent_case")),
                "agent_skills": _parse_agent_skills(raw.get("agent_skills")),
            }
        )
    return memcells


def _parse_scene(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid EverOS scene JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid EverOS scene JSON: expected object")
    title = _string(payload.get("title")).strip()
    summary = _string(payload.get("summary")).strip()
    tags = [
        item.strip()
        for item in payload.get("tags", [])
        if isinstance(item, str) and item.strip()
    ]
    if not title or not summary:
        raise ValueError("invalid EverOS scene JSON: title and summary are required")
    return {"title": title, "summary": summary, "tags": tags[:12]}


def _parse_recollection(content: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid EverOS recollection JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("selected_scenes"), list):
        raise ValueError("invalid EverOS recollection JSON: expected selected_scenes list")
    selections = []
    for raw in payload["selected_scenes"]:
        if not isinstance(raw, dict):
            continue
        scene_id = _string(raw.get("scene_id")).strip()
        if not scene_id:
            continue
        selections.append(
            {
                "scene_id": scene_id,
                "selected_memory_ids": [
                    item
                    for item in raw.get("selected_memory_ids", [])
                    if isinstance(item, str) and item
                ],
                "reconstructed_context": _string(raw.get("reconstructed_context")).strip(),
                "rationale": _string(raw.get("rationale")).strip(),
            }
        )
    return selections


def _parse_atomic_facts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    facts = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            entities = _entities_from_text(text)
        elif isinstance(item, dict):
            text = _string(item.get("text")).strip()
            entities = [
                {"entity_text": entity.strip(), "entity_type": "EXTRACTED"}
                for entity in item.get("entities", [])
                if isinstance(entity, str) and entity.strip()
            ] or _entities_from_text(text)
        else:
            continue
        if text:
            facts.append({"text": text, "entities": entities})
    return facts


def _parse_foresights(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    foresights = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            evidence = ""
            validity = ""
        elif isinstance(item, dict):
            text = _string(item.get("text")).strip()
            evidence = _string(item.get("evidence")).strip()
            validity = _string(item.get("validity")).strip()
        else:
            continue
        if text:
            foresights.append({"text": text, "evidence": evidence, "validity": validity})
    return foresights


def _parse_agent_case(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    task_intent = _string(value.get("task_intent")).strip()
    approach = _string(value.get("approach")).strip()
    key_insight = _string(value.get("key_insight")).strip()
    if not task_intent and not approach and not key_insight:
        return None
    return {
        "task_intent": task_intent,
        "approach": approach,
        "key_insight": key_insight,
        "quality_score": _bounded_float(value.get("quality_score"), default=0.5),
    }


def _parse_agent_skills(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    skills = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name")).strip()
        description = _string(item.get("description")).strip()
        content = _string(item.get("content")).strip()
        if not name and not description and not content:
            continue
        skills.append(
            {
                "name": name or _truncate(description or content, 80),
                "description": description,
                "content": content,
                "confidence": _bounded_float(item.get("confidence"), default=0.5),
            }
        )
    return skills


def _record_specs_from_memcell(
    *,
    memcell: dict[str, Any],
    scene_id: str,
    scope: str,
    scope_id: str,
    request: MemoryIngestRequest,
    existing_hashes: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def append_record(record_type: str, content: str, metadata: dict[str, Any], entities: list[dict[str, Any]]) -> None:
        cleaned = content.strip()
        if not cleaned:
            return
        digest = content_hash(f"{record_type}:{cleaned}")
        duplicate = digest in existing_hashes
        existing_hashes.add(digest)
        records.append(
            {
                "record_type": record_type,
                "content": cleaned,
                "content_hash": digest,
                "metadata": {
                    **metadata,
                    "task_id": request.task_id,
                    "role": request.role,
                    "memory_scope": scope,
                    "memory_scope_id": scope_id,
                    "memory_method": "everos",
                    "scene_id": scene_id,
                },
                "entities": entities or _entities_from_text(cleaned),
                "duplicate": duplicate,
            }
        )

    append_record(
        "episode",
        f"{memcell['summary']}\n\n{memcell['episode']}",
        {"summary": memcell["summary"], "salience": memcell["salience"]},
        _entities_from_text(memcell["summary"] + " " + memcell["episode"]),
    )
    for fact in memcell["atomic_facts"]:
        append_record("atomic_fact", fact["text"], {"salience": memcell["salience"]}, fact["entities"])
    for foresight in memcell["foresights"]:
        append_record(
            "foresight",
            foresight["text"],
            {"evidence": foresight["evidence"], "validity": foresight["validity"]},
            _entities_from_text(foresight["text"] + " " + foresight["evidence"]),
        )
    agent_case = memcell.get("agent_case")
    if agent_case is not None:
        append_record(
            "agent_case",
            "\n".join(
                part
                for part in [
                    f"Task intent: {agent_case['task_intent']}",
                    f"Approach: {agent_case['approach']}",
                    f"Key insight: {agent_case['key_insight']}",
                ]
                if not part.endswith(": ")
            ),
            {"quality_score": agent_case["quality_score"]},
            _entities_from_text(json.dumps(agent_case, ensure_ascii=False)),
        )
    for skill in memcell["agent_skills"]:
        append_record(
            "agent_skill",
            "\n".join(
                part
                for part in [
                    f"Skill: {skill['name']}",
                    f"Description: {skill['description']}",
                    f"Procedure: {skill['content']}",
                ]
                if not part.endswith(": ")
            ),
            {"confidence": skill["confidence"], "skill_name": skill["name"]},
            _entities_from_text(json.dumps(skill, ensure_ascii=False)),
        )
    return records


def _rank_scene_contexts(
    *,
    scenes: list[dict[str, Any]],
    records: list[dict[str, Any]],
    record_scores: dict[str, float],
    query_embedding: list[float],
    threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    records_by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_scene.setdefault(record["scene_id"], []).append(
            {**record, "score": record_scores.get(record["memory_id"], 0.0)}
        )
    contexts = []
    for scene in scenes:
        scene_records = sorted(
            records_by_scene.get(scene["scene_id"], []),
            key=lambda item: (-item["score"], item["record_type"], item["memory_id"]),
        )
        top_record_score = scene_records[0]["score"] if scene_records else 0.0
        scene_semantic = cosine_similarity(query_embedding, scene["embedding"])
        scene_score = min((0.4 * scene_semantic) + (0.6 * top_record_score), 1.0)
        if scene_score < threshold and top_record_score < threshold:
            continue
        contexts.append(
            {
                "scene": scene,
                "records": scene_records[:8],
                "score": scene_score,
                "scene_semantic_score": scene_semantic,
            }
        )
    contexts.sort(key=lambda item: (-item["score"], item["scene"]["scene_id"]))
    return contexts[:limit]


def _scene_context_memory_item(context: dict[str, Any]) -> MemoryItem:
    scene = context["scene"]
    records = context["records"]
    content_lines = [
        f"MemScene: {scene['title']}",
        f"Summary: {scene['summary']}",
    ]
    if scene["tags"]:
        content_lines.append("Tags: " + ", ".join(scene["tags"]))
    if records:
        content_lines.append("Relevant memories:")
        for record in records:
            content_lines.append(
                f"- [{record['record_type']}] {_truncate(record['content'], 800)}"
            )
    return MemoryItem(
        memory_id=f"everos:memscene:{scene['scene_id']}",
        content="\n".join(content_lines),
        score=context["score"],
        metadata={
            "everos_memory_type": "memscene_recollection",
            "scene_id": scene["scene_id"],
            "scene_title": scene["title"],
            "scene_tags": scene["tags"],
            "selected_memory_ids": [record["memory_id"] for record in records],
            "record_types": [record["record_type"] for record in records],
            "scene_member_count": scene["member_count"],
            "scene_semantic_score": context["scene_semantic_score"],
        },
    )


def _agentic_scene_memory_item(context: dict[str, Any], selection: dict[str, Any]) -> MemoryItem:
    item = _scene_context_memory_item(context)
    selected_ids = set(selection["selected_memory_ids"])
    selected_records = [
        record
        for record in context["records"]
        if record["memory_id"] in selected_ids
    ]
    content = selection["reconstructed_context"] or item.content
    return MemoryItem(
        memory_id=item.memory_id,
        content=content,
        score=item.score,
        metadata={
            **item.metadata,
            "selected_memory_ids": [record["memory_id"] for record in selected_records]
            or selection["selected_memory_ids"],
            "recollection_rationale": selection["rationale"],
            "recollection_mode": "agentic",
        },
    )


def _scene_context_for_llm(context: dict[str, Any]) -> dict[str, Any] | None:
    scene = context.get("scene")
    if scene is None:
        return None
    return {
        "scene_id": scene["scene_id"],
        "title": scene["title"],
        "summary": scene["summary"],
        "tags": scene["tags"],
        "member_count": scene["member_count"],
        "records": [
            {
                "memory_id": record["memory_id"],
                "record_type": record["record_type"],
                "content": _truncate(record["content"], 1_000),
                "score": record.get("score"),
            }
            for record in context.get("records", [])
        ],
    }


def _memcell_embedding_text(memcell: dict[str, Any]) -> str:
    parts = [memcell["summary"], memcell["episode"]]
    parts.extend(fact["text"] for fact in memcell["atomic_facts"])
    parts.extend(foresight["text"] for foresight in memcell["foresights"])
    if memcell.get("agent_case") is not None:
        parts.append(json.dumps(memcell["agent_case"], ensure_ascii=False, sort_keys=True))
    parts.extend(json.dumps(skill, ensure_ascii=False, sort_keys=True) for skill in memcell["agent_skills"])
    return "\n".join(part for part in parts if part)


def _best_scene_match(scenes: list[dict[str, Any]], memcell_embedding: list[float]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for scene in scenes:
        score = cosine_similarity(memcell_embedding, scene["embedding"])
        if best is None or score > best["score"]:
            best = {"scene": scene, "score": score}
    return best


def _upsert_planned_scene(planned_scenes: list[dict[str, Any]], scene_plan: dict[str, Any]) -> None:
    replacement = {
        "scene_id": scene_plan["scene_id"],
        "title": scene_plan["title"],
        "summary": scene_plan["summary"],
        "tags": scene_plan["tags"],
        "embedding": scene_plan["embedding"],
        "member_count": scene_plan["member_count"],
        "metadata": scene_plan["metadata"],
        "created_at": "",
        "updated_at": "",
    }
    for index, scene in enumerate(planned_scenes):
        if scene["scene_id"] == scene_plan["scene_id"]:
            planned_scenes[index] = replacement
            return
    planned_scenes.append(replacement)


def _fallback_scene_payload(memcell: dict[str, Any], scene: dict[str, Any] | None) -> dict[str, Any]:
    if scene is not None:
        return {
            "title": scene["title"],
            "summary": _truncate(scene["summary"] + "\n" + memcell["summary"], 2_000),
            "tags": scene["tags"],
        }
    return {
        "title": _truncate(memcell["summary"], 80),
        "summary": _truncate(memcell["episode"], 2_000),
        "tags": [
            entity["entity_text"]
            for entity in _entities_from_text(memcell["summary"] + " " + memcell["episode"])[:8]
        ],
    }


def _merged_embedding(previous: list[float], previous_count: int, new: list[float]) -> list[float]:
    if not previous or len(previous) != len(new) or previous_count <= 0:
        return new
    merged = [
        ((float(left) * previous_count) + float(right)) / (previous_count + 1)
        for left, right in zip(previous, new)
    ]
    norm = math.sqrt(sum(value * value for value in merged))
    if norm == 0 or not math.isfinite(norm):
        return merged
    return [value / norm for value in merged]


def _everos_fuse_score(semantic: float, keyword: float, entity_boost: float) -> float:
    return min((0.55 * semantic) + (0.35 * keyword) + (0.10 * entity_boost), 1.0)


def _entity_overlap_boost(query_entities: set[str], record_entities: list[dict[str, Any]]) -> float:
    if not query_entities:
        return 0.0
    record_entity_texts = {
        str(entity.get("entity_text") or "").lower()
        for entity in record_entities
        if entity.get("entity_text")
    }
    if not record_entity_texts:
        return 0.0
    return min(len(query_entities & record_entity_texts) / max(len(query_entities), 1), 1.0)


def _entities_from_text(text: str) -> list[dict[str, str]]:
    seen = set()
    entities = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)?\b", text):
        entity_text = match.group(0)
        lowered = entity_text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        entities.append({"entity_text": entity_text, "entity_type": "PROPER"})
    return entities


def _finite_vector(vector: list[float]) -> bool:
    return all(
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in vector
    )


def _bounded_float(value: Any, *, default: float) -> float:
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
        return max(0.0, min(float(value), 1.0))
    return default


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"...[truncated {len(value) - max_chars} chars]"


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="microseconds") + "Z"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str) -> Any:
    return json.loads(value)


def _failed_result(previous_state_ref: str, error_type: str, exc: Exception) -> MemoryIngestResult:
    return MemoryIngestResult(
        status="failed",
        state_ref=previous_state_ref,
        previous_state_ref=previous_state_ref,
        metadata={"error_type": error_type, "error": str(exc)},
    )


def _scope_ref_part(value: str) -> str:
    encoded = quote(value, safe=":")
    return f"{len(value)}:{encoded}"


def _parse_scope_ref_part(value: str) -> str | None:
    length_text, separator, encoded_value = value.partition(":")
    if not separator or not length_text.isdigit():
        return None
    decoded_value = unquote(encoded_value)
    if len(decoded_value) != int(length_text):
        return None
    return decoded_value


def _parse_state_ref(state_ref: str | None) -> tuple[str | None, str, int] | None:
    if not state_ref:
        return None
    parsed = urlparse(state_ref)
    if parsed.scheme != "method" or parsed.netloc != "everos":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 4 and parts[3].startswith("v"):
        backend_id = _parse_scope_ref_part(parts[0])
        scope = _parse_scope_ref_part(parts[1])
        scope_id = _parse_scope_ref_part(parts[2])
        version_text = parts[3][1:]
        if backend_id is None or scope is None or scope_id is None or not version_text.isdigit():
            return None
        return scope, scope_id, int(version_text)
    if len(parts) == 3 and parts[2].startswith("v"):
        scope = _parse_scope_ref_part(parts[0])
        scope_id = _parse_scope_ref_part(parts[1])
        version_text = parts[2][1:]
        if scope is None or scope_id is None or not version_text.isdigit():
            return None
        return scope, scope_id, int(version_text)
    if len(parts) != 2 or not parts[1].startswith("v"):
        return None
    scope_id = _parse_scope_ref_part(parts[0])
    version_text = parts[1][1:]
    if scope_id is None or not version_text.isdigit():
        return None
    return None, scope_id, int(version_text)


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        return stripped
    return match.group(1).strip()
