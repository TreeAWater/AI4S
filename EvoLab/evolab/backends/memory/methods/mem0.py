from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
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
    fuse_scores,
)
from evolab.backends.memory.methods.store import SQLiteMemoryStore, content_hash
from evolab.contracts.common import Message
from evolab.contracts.llm import LLMGenerationConfig
from evolab.contracts.retrieval import MemoryItem


MEM0_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "attributed_to": {"type": "string"},
                    "linked_memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["id", "text", "attributed_to", "linked_memory_ids"],
            },
        }
    },
    "required": ["memory"],
}


MEM0_ADDITIVE_EXTRACTION_PROMPT = """You are EvoLab native mem0 extraction.
Extract only durable facts from the new user and assistant messages.
Use existing memories only to avoid duplicates and to populate linked_memory_ids.
Preserve concrete biological, experimental, task, preference, and workflow details.
Do not rewrite, update, or delete existing memories; this is ADD-only extraction.
Return JSON only with a memory array. Each item needs id, text, attributed_to, and linked_memory_ids.
Use an empty memory array when no durable fact should be stored."""


class Mem0MemoryMethod:
    method_name = "mem0"

    def __init__(
        self,
        store_path: str | Path,
        llm_backend_id: str,
        embedding_backend_id: str,
        audit_log_path: str | Path | None = None,
        top_k_existing: int = 10,
        state_ref_backend_id: str = "mem0",
    ):
        self.store = SQLiteMemoryStore(store_path, audit_log_path=audit_log_path)
        self.llm_backend_id = llm_backend_id
        self.embedding_backend_id = embedding_backend_id
        self.top_k_existing = top_k_existing
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

    def instantiate(self, state_ref: str | None) -> "Mem0MemoryMethod":
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
        self._require_dependencies()
        previous_version = self._version(request.scope, request.scope_id)
        previous_state_ref = self._state_ref(request.scope, request.scope_id)
        existing = self.store.list_memories(request.scope, request.scope_id)
        recent_messages = self.store.recent_messages(request.scope, request.scope_id, limit=20)

        response = self.llm_runtime.generate(
            self._extraction_messages(request, existing[: self.top_k_existing], recent_messages),
            [],
            LLMGenerationConfig(
                model="",
                temperature=0,
                response_json_schema=MEM0_EXTRACTION_SCHEMA,
            ),
        )
        content = response.action.content if response.action.action == "final_answer" else None
        try:
            extracted = _parse_extraction(content or "")
        except ValueError as exc:
            return MemoryIngestResult(
                status="failed",
                state_ref=previous_state_ref,
                previous_state_ref=previous_state_ref,
                metadata={"error_type": "extraction_parse", "error": str(exc)},
            )

        if not extracted:
            if request.messages:
                try:
                    self.store.save_ingest(
                        request.scope,
                        request.scope_id,
                        request.messages,
                        self._message_metadata(request),
                        [],
                    )
                except Exception as exc:
                    return _failed_result(previous_state_ref, "store_error", exc)
                self._mark_scope_changed(request.scope, request.scope_id, previous_version)
                return MemoryIngestResult(
                    status="updated",
                    state_ref=self._state_ref(request.scope, request.scope_id),
                    previous_state_ref=previous_state_ref,
                    metadata={
                        "extracted_count": 0,
                        "stored_count": 0,
                        "duplicate_count": 0,
                    },
                )
            return MemoryIngestResult(
                status="skipped",
                state_ref=self._state_ref(request.scope, request.scope_id),
                previous_state_ref=previous_state_ref,
                metadata={
                    "extracted_count": 0,
                    "stored_count": 0,
                    "duplicate_count": 0,
                },
            )

        existing_hash_to_memory_id = _existing_hash_to_memory_id(existing)
        batch_hashes: set[str] = set()
        to_store: list[dict[str, Any]] = []
        skipped_ids: list[str] = []
        linked_memory_ids: list[str] = []

        for item in extracted:
            linked_memory_ids.extend(item["linked_memory_ids"])
            item_hash = content_hash(item["text"])
            existing_memory_id = existing_hash_to_memory_id.get(item_hash)
            if existing_memory_id is not None:
                skipped_ids.append(existing_memory_id)
                continue
            if item_hash in batch_hashes:
                skipped_ids.append(item["id"])
                continue
            batch_hashes.add(item_hash)
            to_store.append({**item, "content_hash": item_hash})

        if not to_store:
            if request.messages:
                try:
                    self.store.save_ingest(
                        request.scope,
                        request.scope_id,
                        request.messages,
                        self._message_metadata(request),
                        [],
                    )
                except Exception as exc:
                    return _failed_result(previous_state_ref, "store_error", exc)
                self._mark_scope_changed(request.scope, request.scope_id, previous_version)
                return MemoryIngestResult(
                    status="updated",
                    state_ref=self._state_ref(request.scope, request.scope_id),
                    previous_state_ref=previous_state_ref,
                    skipped_memory_ids=skipped_ids,
                    linked_memory_ids=linked_memory_ids,
                    metadata={
                        "extracted_count": len(extracted),
                        "stored_count": 0,
                        "duplicate_count": len(skipped_ids),
                    },
                )
            return MemoryIngestResult(
                status="skipped",
                state_ref=self._state_ref(request.scope, request.scope_id),
                previous_state_ref=previous_state_ref,
                skipped_memory_ids=skipped_ids,
                linked_memory_ids=linked_memory_ids,
                metadata={
                    "extracted_count": len(extracted),
                    "stored_count": 0,
                    "duplicate_count": len(skipped_ids),
                },
            )

        try:
            embedding_response = self.embedding_runtime.embed(
                [item["text"] for item in to_store],
                purpose="add",
            )
            if len(embedding_response.vectors) != len(to_store):
                raise ValueError(
                    f"expected {len(to_store)} embedding vectors, got {len(embedding_response.vectors)}"
                )
        except ValueError as exc:
            return _failed_result(previous_state_ref, "embedding_cardinality", exc)
        except Exception as exc:
            return _failed_result(previous_state_ref, "embedding_error", exc)

        memory_specs = [
            {
                "content": item["text"],
                "embedding": vector,
                "metadata": {
                    **self._memory_metadata(request),
                    "attributed_to": item["attributed_to"],
                    "content_hash": item["content_hash"],
                    "source_extraction_id": item["id"],
                },
                "linked_memory_ids": item["linked_memory_ids"],
                "entities": extract_entities(item["text"]),
            }
            for item, vector in zip(to_store, embedding_response.vectors)
        ]
        try:
            added_memory_ids = self.store.save_ingest(
                request.scope,
                request.scope_id,
                request.messages,
                self._message_metadata(request),
                memory_specs,
            )
        except Exception as exc:
            return _failed_result(previous_state_ref, "store_error", exc)

        self._mark_scope_changed(request.scope, request.scope_id, previous_version)
        return MemoryIngestResult(
            status="updated",
            state_ref=self._state_ref(request.scope, request.scope_id),
            previous_state_ref=previous_state_ref,
            added_memory_ids=added_memory_ids,
            skipped_memory_ids=skipped_ids,
            linked_memory_ids=linked_memory_ids,
            metadata={
                "extracted_count": len(extracted),
                "stored_count": len(added_memory_ids),
                "duplicate_count": len(skipped_ids),
            },
        )

    def search(self, request: MemorySearchRequest) -> MemorySearchResult:
        self._validate_search_scope(request)
        self._require_dependencies()
        top_k = 20 if request.top_k is None else request.top_k
        threshold = 0.1 if request.threshold is None else request.threshold
        embedding_response = self.embedding_runtime.embed([request.query], purpose="search")
        if len(embedding_response.vectors) != 1:
            raise RuntimeError(
                f"expected 1 query embedding vector, got {len(embedding_response.vectors)}"
            )
        query_vector = embedding_response.vectors[0]
        candidates = self.store.semantic_candidates(request.scope, request.scope_id)
        keyword_scores = bm25_keyword_scores(
            request.query,
            self.store.keyword_candidates(request.scope, request.scope_id),
        )
        ranked = []

        for candidate in candidates:
            semantic = cosine_similarity(query_vector, candidate["embedding"])
            keyword = keyword_scores.get(candidate["memory_id"], 0.0)
            entity_boost = self._entity_boost(
                request.query,
                candidate["memory_id"],
                request.scope,
                request.scope_id,
            )
            score = fuse_scores(semantic, keyword, entity_boost)
            if score < threshold:
                continue
            ranked.append(
                {
                    "memory_id": candidate["memory_id"],
                    "content": candidate["content"],
                    "score": score,
                    "metadata": candidate["metadata"],
                }
            )

        ranked.sort(key=lambda item: (-item["score"], item["memory_id"], item["content"]))
        items = [
            MemoryItem(
                memory_id=item["memory_id"],
                content=item["content"],
                score=item["score"],
                metadata=item["metadata"],
            )
            for item in ranked[:top_k]
        ]
        return MemorySearchResult(
            items=items,
            state_ref=self._state_ref(request.scope, request.scope_id),
            metadata={
                "memory_method": self.method_name,
                "candidate_count": len(candidates),
                "returned_count": len(items),
                "threshold": threshold,
                "top_k": top_k,
            },
        )

    def _entity_boost(self, query: str, memory_id: str, scope: str, scope_id: str) -> float:
        query_entities = {
            entity["entity_text"].lower()
            for entity in extract_entities(query)
            if entity.get("entity_text")
        }
        if not query_entities:
            return 0.0
        boost = 0.0
        for link in self.store.entity_links(scope, scope_id):
            if link.get("memory_id") != memory_id:
                continue
            entity_text = str(link.get("entity_text") or "").lower()
            if entity_text in query_entities:
                boost += 0.5
        return min(boost, 0.5)

    def _validate_search_scope(self, request: MemorySearchRequest) -> None:
        if request.scope not in {"agent", "task"}:
            raise ValueError("scope must be 'agent' or 'task'")
        if not request.scope_id:
            raise ValueError("scope_id must be a non-empty string")

        supported_filters = {"memory_scope", "memory_scope_id"}
        unsupported_filters = sorted(set(request.filters) - supported_filters)
        if unsupported_filters:
            raise ValueError(
                "Mem0MemoryMethod.search does not support filters: "
                + ", ".join(unsupported_filters)
            )
        memory_scope = request.filters.get("memory_scope")
        if memory_scope is not None and memory_scope != request.scope:
            raise ValueError("memory_scope filter conflicts with request scope")
        memory_scope_id = request.filters.get("memory_scope_id")
        if memory_scope_id is not None and memory_scope_id != request.scope_id:
            raise ValueError("memory_scope_id filter conflicts with request scope_id")

    def _require_dependencies(self) -> None:
        missing = []
        if self.llm_runtime is None:
            missing.append(f"llm runtime {self.llm_backend_id!r}")
        if self.embedding_runtime is None:
            missing.append(f"embedding runtime {self.embedding_backend_id!r}")
        if missing:
            raise RuntimeError("Mem0MemoryMethod requires bind_runtimes before use; missing " + ", ".join(missing))

    def _state_ref(self, scope: str, scope_id: str) -> str:
        return (
            "method://mem0/"
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

    def _message_metadata(self, request: MemoryIngestRequest) -> dict[str, Any]:
        return {
            **request.metadata,
            "task_id": request.task_id,
            "role": request.role,
            "memory_scope": request.scope,
            "memory_scope_id": request.scope_id,
            "memory_method": self.method_name,
        }

    def _memory_metadata(self, request: MemoryIngestRequest) -> dict[str, Any]:
        return {
            **request.metadata,
            "task_id": request.task_id,
            "role": request.role,
            "memory_scope": request.scope,
            "memory_scope_id": request.scope_id,
            "memory_method": self.method_name,
        }

    def _extraction_messages(
        self,
        request: MemoryIngestRequest,
        existing: list[MemoryItem],
        recent_messages: list[dict[str, Any]],
    ) -> list[Message]:
        payload = {
            "scope": request.scope,
            "scope_id": request.scope_id,
            "existing_memories": [
                {"id": item.memory_id, "text": item.content}
                for item in existing
            ],
            "recent_messages": [
                {"role": item.get("role"), "content": item.get("content")}
                for item in recent_messages
            ],
            "new_messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        return [
            Message(role="system", content=MEM0_ADDITIVE_EXTRACTION_PROMPT),
            Message(role="user", content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        ]


def _parse_extraction(content: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid mem0 extraction JSON: {exc.msg}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("memory"), list):
        raise ValueError("invalid mem0 extraction JSON: expected object with memory list")

    extracted = []
    for index, item in enumerate(payload["memory"]):
        if not isinstance(item, dict):
            raise ValueError(f"invalid mem0 extraction item at index {index}: expected object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        extraction_id = item.get("id")
        attributed_to = item.get("attributed_to", "unknown")
        linked_memory_ids = item.get("linked_memory_ids", [])
        if not isinstance(extraction_id, str) or not extraction_id:
            extraction_id = str(index)
        if not isinstance(attributed_to, str) or not attributed_to:
            attributed_to = "unknown"
        if not isinstance(linked_memory_ids, list):
            linked_memory_ids = []
        extracted.append(
            {
                "id": extraction_id,
                "text": text.strip(),
                "attributed_to": attributed_to,
                "linked_memory_ids": [
                    linked_id
                    for linked_id in linked_memory_ids
                    if isinstance(linked_id, str) and linked_id
                ],
            }
        )
    return extracted


def _failed_result(previous_state_ref: str, error_type: str, exc: Exception) -> MemoryIngestResult:
    return MemoryIngestResult(
        status="failed",
        state_ref=previous_state_ref,
        previous_state_ref=previous_state_ref,
        metadata={"error_type": error_type, "error": str(exc)},
    )


def _existing_hash_to_memory_id(existing: list[MemoryItem]) -> dict[str, str]:
    return {
        str(item.metadata.get("content_hash") or content_hash(item.content)): item.memory_id
        for item in existing
    }


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
    if parsed.scheme != "method" or parsed.netloc != "mem0":
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


def extract_entities(text: str) -> list[dict[str, Any]]:
    seen = set()
    entities = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)?\b", text):
        entity_text = match.group(0)
        if entity_text in seen:
            continue
        seen.add(entity_text)
        entities.append({"entity_text": entity_text, "entity_type": "PROPER"})
    return entities
