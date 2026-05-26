from pathlib import Path

import pytest

from evolab.backends.memory.methods.base import MemorySearchRequest
from evolab.backends.memory.methods.mem0 import Mem0MemoryMethod
from evolab.backends.memory.methods.retrieval import fuse_scores
from evolab.contracts.embeddings import EmbeddingResponse


class _NoLLM:
    def generate(self, messages, tool_specs, generation_config):
        raise AssertionError("search must not call LLM")


class _StaticEmbeddingRuntime:
    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self.vectors_by_text = vectors_by_text

    def embed(self, texts, *, purpose):
        return EmbeddingResponse(
            backend_id="static",
            model="static",
            vectors=[self.vectors_by_text[text] for text in texts],
            metadata={"purpose": purpose},
        )


class _FailingEmbeddingRuntime:
    def embed(self, texts, *, purpose):
        raise RuntimeError("configured embedding failure")


def _seeded_method(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime(
        {
            "Sigma70 promoter records must preserve exact sequence coordinates": [1.0, 0.0, 0.0],
            "Synthetic promoter diffusion records contain generated promoter sequences": [0.8, 0.2, 0.0],
            "Unrelated scheduling note": [-1.0, 0.0, 0.0],
            "promoter sequence coordinates": [1.0, 0.0, 0.0],
            "promoter only other scope": [0.0, 1.0, 0.0],
            "other scope": [0.0, 1.0, 0.0],
        }
    )
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    for text in [
        "Sigma70 promoter records must preserve exact sequence coordinates",
        "Synthetic promoter diffusion records contain generated promoter sequences",
        "Unrelated scheduling note",
    ]:
        vector = embedding.embed([text], purpose="add").vectors[0]
        method.store.insert_memory(
            "task",
            "task:bio",
            text,
            vector,
            {"attributed_to": "assistant"},
            [],
            [],
        )
    return method


def test_mem0_search_returns_ranked_hybrid_results(tmp_path: Path):
    method = _seeded_method(tmp_path)

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="promoter sequence coordinates",
            top_k=2,
            threshold=0.0,
        )
    )

    assert len(result.items) == 2
    assert "promoter" in result.items[0].content.lower()
    assert result.items[0].score is not None


def test_mem0_search_rejects_empty_scope_id(tmp_path: Path):
    method = _seeded_method(tmp_path)

    with pytest.raises(ValueError, match="scope_id"):
        method.search(
            MemorySearchRequest(
                task_id="bio",
                role="task",
                scope="task",
                scope_id="",
                query="promoter sequence coordinates",
            )
        )


def test_mem0_search_requires_bound_runtimes(tmp_path: Path):
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )

    with pytest.raises(RuntimeError, match="llm.*embed|embed.*llm"):
        method.search(
            MemorySearchRequest(
                task_id="bio",
                role="task",
                scope="task",
                scope_id="task:bio",
                query="promoter",
            )
        )


def test_mem0_search_requires_llm_runtime_without_calling_it(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime({"semantic query": [1.0, 0.0]})
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.embedding_runtime = embedding

    with pytest.raises(RuntimeError, match="llm runtime 'llm'"):
        method.search(
            MemorySearchRequest(
                task_id="bio",
                role="task",
                scope="task",
                scope_id="task:bio",
                query="semantic query",
            )
        )


def test_mem0_search_propagates_embedding_runtime_failure(tmp_path: Path):
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": _FailingEmbeddingRuntime()},
    )

    with pytest.raises(RuntimeError, match="configured embedding failure"):
        method.search(
            MemorySearchRequest(
                task_id="bio",
                role="task",
                scope="task",
                scope_id="task:bio",
                query="promoter sequence coordinates",
            )
        )


def test_mem0_search_returns_exact_fused_score_without_post_penalty(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime({"semantic query": [1.0, 0.0]})
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    memory_id = method.store.insert_memory(
        "task",
        "task:bio",
        "lexically unrelated",
        [1.0, 0.0],
        {},
        [],
        [],
    )

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="semantic query",
            top_k=1,
            threshold=0.0,
        )
    )

    assert result.items[0].memory_id == memory_id
    assert result.items[0].score == fuse_scores(1.0, 0.0, 0.0)


def test_mem0_search_filters_by_fused_score_not_semantic_only(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime({"rare token": [1.0, 0.0]})
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    returned_id = method.store.insert_memory(
        "task",
        "task:bio",
        "rare token match",
        [0.0, 1.0],
        {},
        [],
        [],
    )
    method.store.insert_memory(
        "task",
        "task:bio",
        "below fused threshold",
        [0.0, 1.0],
        {},
        [],
        [],
    )

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="rare token",
            top_k=10,
            threshold=0.6,
        )
    )

    assert [item.memory_id for item in result.items] == [returned_id]


def test_mem0_search_rejects_unsupported_filters(tmp_path: Path):
    method = _seeded_method(tmp_path)

    with pytest.raises(ValueError, match="filters"):
        method.search(
            MemorySearchRequest(
                task_id="bio",
                role="task",
                scope="task",
                scope_id="task:bio",
                query="promoter",
                filters={"attributed_to": "assistant"},
            )
        )


def test_mem0_search_accepts_matching_scope_filters(tmp_path: Path):
    method = _seeded_method(tmp_path)

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="promoter sequence coordinates",
            filters={"memory_scope": "task", "memory_scope_id": "task:bio"},
            top_k=1,
            threshold=0.0,
        )
    )

    assert len(result.items) == 1


def test_mem0_search_rejects_conflicting_scope_filters(tmp_path: Path):
    method = _seeded_method(tmp_path)

    with pytest.raises(ValueError, match="memory_scope"):
        method.search(
            MemorySearchRequest(
                task_id="bio",
                role="task",
                scope="task",
                scope_id="task:bio",
                query="promoter sequence coordinates",
                filters={"memory_scope": "agent", "memory_scope_id": "task:bio"},
            )
        )

    with pytest.raises(ValueError, match="memory_scope_id"):
        method.search(
            MemorySearchRequest(
                task_id="bio",
                role="task",
                scope="task",
                scope_id="task:bio",
                query="promoter sequence coordinates",
                filters={"memory_scope": "task", "memory_scope_id": "task:other"},
            )
        )


def test_mem0_search_uses_bm25_like_keyword_score_over_scope(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime({"promoter": [1.0, 0.0]})
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    method.store.insert_memory(
        "task",
        "task:bio",
        "promoter",
        [1.0, 0.0],
        {},
        [],
        [],
    )
    repeated_id = method.store.insert_memory(
        "task",
        "task:bio",
        "promoter promoter promoter promoter promoter",
        [0.9, 0.435],
        {},
        [],
        [],
    )

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="promoter",
            top_k=2,
            threshold=0.0,
        )
    )

    assert result.items[0].memory_id == repeated_id


def test_mem0_search_enforces_scope_isolation(tmp_path: Path):
    method = _seeded_method(tmp_path)
    vector = method.embedding_runtime.embed(
        ["promoter only other scope"],
        purpose="add",
    ).vectors[0]
    method.store.insert_memory(
        "task",
        "task:other",
        "promoter only other scope",
        vector,
        {},
        [],
        [],
    )

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="other scope",
            top_k=10,
            threshold=0.0,
        )
    )

    assert all(item.content != "promoter only other scope" for item in result.items)


def test_mem0_search_isolates_same_scope_id_across_scopes(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime({"Sigma70": [1.0, 0.0]})
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    method.store.insert_memory(
        "agent",
        "shared",
        "agent Sigma70 memory",
        [1.0, 0.0],
        {},
        [],
        [{"entity_text": "Sigma70", "entity_type": "PROPER"}],
    )
    task_memory_id = method.store.insert_memory(
        "task",
        "shared",
        "task Sigma70 memory",
        [0.9, 0.435],
        {},
        [],
        [{"entity_text": "Sigma70", "entity_type": "PROPER"}],
    )

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="shared",
            query="Sigma70",
            top_k=10,
            threshold=0.0,
        )
    )

    assert [item.memory_id for item in result.items] == [task_memory_id]


def test_mem0_search_threshold_filters_low_fused_score_results(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime(
        {
            "different request": [1.0, 0.0],
            "opposite memory": [-1.0, 0.0],
        }
    )
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    method.store.insert_memory(
        "task",
        "task:bio",
        "opposite memory",
        [-1.0, 0.0],
        {},
        [],
        [],
    )

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="different request",
            top_k=10,
            threshold=0.1,
        )
    )

    assert result.items == []


def test_mem0_search_entity_boost_changes_ranking(tmp_path: Path):
    embedding = _StaticEmbeddingRuntime({"Sigma70": [1.0, 0.0]})
    method = Mem0MemoryMethod(
        store_path=tmp_path / "mem0.sqlite",
        llm_backend_id="llm",
        embedding_backend_id="embed",
    )
    method.bind_runtimes(
        llm_runtimes={"llm": _NoLLM()},
        embedding_runtimes={"embed": embedding},
    )
    boosted_id = method.store.insert_memory(
        "task",
        "task:bio",
        "linked transcription record",
        [0.6, 0.8],
        {},
        [],
        [{"entity_text": "Sigma70", "entity_type": "PROPER"}],
    )
    method.store.insert_memory(
        "task",
        "task:bio",
        "semantically closer competitor",
        [0.7, 0.714],
        {},
        [],
        [],
    )

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="Sigma70",
            top_k=2,
            threshold=0.0,
        )
    )

    assert [item.memory_id for item in result.items][0] == boosted_id


def test_mem0_search_top_k_limits_results(tmp_path: Path):
    method = _seeded_method(tmp_path)

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="promoter sequence coordinates",
            top_k=1,
            threshold=0.0,
        )
    )

    assert len(result.items) == 1


def test_mem0_search_preserves_explicit_zero_top_k(tmp_path: Path):
    method = _seeded_method(tmp_path)

    result = method.search(
        MemorySearchRequest(
            task_id="bio",
            role="task",
            scope="task",
            scope_id="task:bio",
            query="promoter sequence coordinates",
            top_k=0,
            threshold=0.0,
        )
    )

    assert result.items == []
    assert result.metadata["top_k"] == 0
