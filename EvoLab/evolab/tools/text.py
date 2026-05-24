from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from evolab.contracts.tools import ToolResult, ToolSpec
from evolab.tools.paths import resolve_path_arguments
from evolab.tools.runtime import ToolRegistry


def register_text_tools(registry: ToolRegistry, *, base_dir: str | Path | None = None) -> None:
    _register_if_missing(
        registry,
        _search_text_spec(),
        lambda arguments: search_text(resolve_path_arguments(arguments, base_dir=base_dir, names=("path",))),
    )
    _register_if_missing(
        registry,
        _extract_sections_spec(),
        lambda arguments: extract_sections(resolve_path_arguments(arguments, base_dir=base_dir, names=("path",))),
    )


def search_text(arguments: dict[str, Any]) -> ToolResult:
    text = _text_from_arguments(arguments)
    query = arguments.get("query")
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    case_sensitive = bool(arguments.get("case_sensitive", False))
    context_chars = int(arguments.get("context_chars", 200))
    max_matches = int(arguments.get("max_matches", 20))
    flags = 0 if case_sensitive else re.IGNORECASE
    matches = []
    try:
        pattern = re.compile(query, flags)
    except re.error:
        pattern = re.compile(re.escape(query), flags)
    for match in pattern.finditer(text):
        start, end = match.span()
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        matches.append(
            {
                "start": start,
                "end": end,
                "match": text[start:end],
                "context": text[context_start:context_end],
            }
        )
        if len(matches) >= max_matches:
            break
    payload = {"query": query, "match_count": len(matches), "matches": matches}
    return ToolResult(
        call_id="search_text",
        status="ok",
        content=f"found {len(matches)} matches for {query!r}",
        metadata=payload,
    )


def extract_sections(arguments: dict[str, Any]) -> ToolResult:
    text = _text_from_arguments(arguments)
    heading_patterns = _heading_patterns(arguments.get("heading_patterns"))
    headings = []
    for pattern in heading_patterns:
        headings.extend(_find_headings(text, pattern))
    headings = sorted({(item["start"], item["end"], item["title"], item["level"]): item for item in headings}.values(), key=lambda item: item["start"])
    sections = []
    if not headings:
        sections.append(
            {
                "title": "Document",
                "level": 1,
                "start": 0,
                "end": len(text),
                "preview": text[:500],
            }
        )
    for index, heading in enumerate(headings):
        start = heading["end"]
        end = headings[index + 1]["start"] if index + 1 < len(headings) else len(text)
        sections.append(
            {
                "title": heading["title"],
                "level": heading["level"],
                "start": start,
                "end": end,
                "preview": text[start:end].strip()[:500],
            }
        )
    payload = {"section_count": len(sections), "sections": sections}
    return ToolResult(
        call_id="extract_sections",
        status="ok",
        content=f"extracted {len(sections)} sections",
        metadata=payload,
    )


def _find_headings(text: str, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    headings = []
    for match in pattern.finditer(text):
        title = match.group("title").strip() if "title" in match.groupdict() else match.group(0).strip()
        level_text = match.groupdict().get("level")
        level = len(level_text) if level_text and level_text.startswith("#") else 1
        headings.append({"title": title, "level": level, "start": match.start(), "end": match.end()})
    return headings


def _heading_patterns(value: Any) -> list[re.Pattern[str]]:
    if isinstance(value, list) and value:
        return [re.compile(str(item), re.MULTILINE) for item in value]
    return [
        re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE),
        re.compile(
            r"^(?P<title>Abstract|Introduction|Methods?|Materials and Methods|Results?|Discussion|Conclusion|References|Supplementary(?: Information| Data)?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ]


def _text_from_arguments(arguments: dict[str, Any]) -> str:
    text = arguments.get("text")
    if isinstance(text, str):
        return text
    path = arguments.get("path")
    if isinstance(path, str) and path:
        return Path(path).expanduser().read_text(encoding=str(arguments.get("encoding") or "utf-8"))
    raise ValueError("either text or path must be provided")


def _search_text_spec() -> ToolSpec:
    return ToolSpec(
        name="search_text",
        description="Search text or a local text file for deterministic substring or regex matches.",
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Inline text to search."},
                "path": {"type": "string", "description": "Path to a text file to search."},
                "encoding": {"type": "string", "description": "Text file encoding. Defaults to utf-8."},
                "query": {"type": "string", "description": "Substring or regular expression to search for."},
                "case_sensitive": {"type": "boolean", "description": "Whether matching is case-sensitive."},
                "context_chars": {"type": "integer", "minimum": 0, "description": "Context characters around matches."},
                "max_matches": {"type": "integer", "minimum": 1, "description": "Maximum matches to return."},
            },
            "required": ["query"],
        },
    )


def _extract_sections_spec() -> ToolSpec:
    return ToolSpec(
        name="extract_sections",
        description="Extract markdown-like or scientific paper sections from text.",
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Inline text to section."},
                "path": {"type": "string", "description": "Path to a text file to section."},
                "encoding": {"type": "string", "description": "Text file encoding. Defaults to utf-8."},
                "heading_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional regular expressions for headings.",
                },
            },
            "required": [],
        },
    )


def _register_if_missing(registry: ToolRegistry, spec: ToolSpec, handler: Any) -> None:
    if registry.get_spec(spec.name) is None:
        registry.register(spec, handler)
