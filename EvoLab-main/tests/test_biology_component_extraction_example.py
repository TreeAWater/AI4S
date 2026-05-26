from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "biology_component_extraction_session.py"
    spec = importlib.util.spec_from_file_location("biology_component_extraction_session", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_biology_component_extraction_example_uses_real_backends(tmp_path: Path, monkeypatch):
    module = _load_example_module()
    lab = tmp_path / "lab"
    monkeypatch.setenv("EVOLAB_LAB_DIR", str(lab))
    monkeypatch.setenv("EVOLAB_ENV_FILE", str(lab / ".env"))

    config = module.build_session_config(validate_lab=False)

    assert config.llm["default"]["type"] == "api"
    assert config.llm["memory-extractor"]["type"] == "api"
    assert config.embeddings["memory-embedding"]["type"] == "api"
    assert config.memory["mem0-task-memory"]["type"] == "method"
    assert config.memory["mem0-task-memory"]["method"] == "mem0"
    assert config.memory["mem0-task-memory"]["llm_backend"] == "memory-extractor"
    assert config.memory["mem0-task-memory"]["embedding_backend"] == "memory-embedding"
    assert config.skills["scientific-ie-graph"]["type"] == "graph"
    assert config.skills["scientific-ie-graph"]["strict_packages"] is True
    assert str(lab / ".evolab") in config.skills["scientific-ie-graph"]["repo_root"]
    assert config.tools["builtin"] is True
