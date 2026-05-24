from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from html import escape
import json
from pathlib import Path
from typing import Any

from evolab.registries.trajectory import FileTrajectoryRegistry


DEFAULT_PREVIEW_CHARS = 600


def visualize_trajectory(
    *,
    lab_root: Path | str | None = None,
    trajectory_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    source_dir = resolve_trajectory_dir(lab_root=lab_root, trajectory_dir=trajectory_dir)
    snapshot = load_trajectory_snapshot(source_dir)
    html = render_trajectory_html(
        snapshot,
        source_dir=source_dir,
        title=title or _default_title(lab_root=lab_root, trajectory_dir=trajectory_dir),
    )
    target = _resolve_output_path(
        output_path=output_path,
        lab_root=Path(lab_root) if lab_root is not None else None,
        trajectory_dir=source_dir,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return {
        "output_path": str(target),
        "trajectory_dir": str(source_dir),
        "counts": snapshot["counts"],
    }


def resolve_trajectory_dir(
    *,
    lab_root: Path | str | None = None,
    trajectory_dir: Path | str | None = None,
) -> Path:
    if lab_root is None and trajectory_dir is None:
        raise ValueError("one of lab_root or trajectory_dir is required")
    if lab_root is not None and trajectory_dir is not None:
        raise ValueError("lab_root and trajectory_dir are mutually exclusive")
    if trajectory_dir is not None:
        return Path(trajectory_dir)
    root = Path(lab_root)  # type: ignore[arg-type]
    registry_dir = root / "registries" / "trajectory"
    if registry_dir.exists():
        return registry_dir
    return registry_dir


def load_trajectory_snapshot(trajectory_dir: Path | str) -> dict[str, Any]:
    registry = FileTrajectoryRegistry(Path(trajectory_dir))
    meta_runs = registry.list_meta_agent_runs()
    subagent_runs = registry.list_subagent_runs()
    llm_calls = registry.list_llm_calls()
    tool_calls = registry.list_tool_call_records()
    events = registry.list_events()
    evolution_runs = registry.list_evolution_runs()
    return {
        "meta_runs": meta_runs,
        "subagent_runs": subagent_runs,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "events": events,
        "evolution_runs": evolution_runs,
        "counts": {
            "meta_agent_runs": len(meta_runs),
            "subagent_runs": len(subagent_runs),
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "events": len(events),
            "evolution_runs": len(evolution_runs),
        },
    }


def render_trajectory_html(
    snapshot: dict[str, Any],
    *,
    source_dir: Path | str,
    title: str = "EvoLab Trajectory",
) -> str:
    dashboard = _dashboard_data(snapshot, source_dir=source_dir)
    return _render_dashboard_html(dashboard, title=title)


def _dashboard_data(snapshot: dict[str, Any], *, source_dir: Path | str) -> dict[str, Any]:
    counts = snapshot["counts"]
    subagent_runs = sorted(snapshot["subagent_runs"], key=_subagent_sort_key)
    meta_runs = sorted(snapshot["meta_runs"], key=_created_sort_key)
    llm_calls = sorted(snapshot["llm_calls"], key=_llm_sort_key)
    tool_calls = sorted(snapshot["tool_calls"], key=_tool_sort_key)
    events = sorted(snapshot["events"], key=_created_sort_key)
    evolution_runs = sorted(snapshot["evolution_runs"], key=_created_sort_key)

    run_roles = {run.run_ref: run.role for run in subagent_runs}
    llm_by_run = _group_by(llm_calls, "run_ref")
    tools_by_run = _group_by(tool_calls, "run_ref")
    tool_status_counts = Counter(
        (call.tool_name, call.record.result.status) for call in tool_calls
    )
    started_by_ref = {
        event.run_ref: _time(event.created_at)
        for event in events
        if event.event_type == "subagent_started" and event.run_ref
    }
    meta_items = [_meta_dashboard_item(run) for run in meta_runs]
    subagent_items = [
        _subagent_dashboard_item(
            run,
            llm_count=len(llm_by_run.get(run.run_ref, [])),
            tool_count=len(tools_by_run.get(run.run_ref, [])),
            started_at=started_by_ref.get(run.run_ref),
        )
        for run in subagent_runs
    ]
    tool_items = [_tool_dashboard_item(call) for call in tool_calls]
    llm_items = [_llm_dashboard_item(call, role=run_roles.get(call.run_ref, "")) for call in llm_calls]
    event_items = [_event_dashboard_item(event) for event in events]
    evolution_items = [_evolution_dashboard_item(run) for run in evolution_runs]
    flow_items = sorted(
        [
            _flow_dashboard_item(item)
            for item in [*meta_items, *subagent_items]
        ],
        key=lambda item: (str(item.get("time") or ""), str(item.get("id") or "")),
    )
    return {
        "sourceDir": str(source_dir),
        "generatedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "counts": counts,
        "metaRuns": meta_items,
        "subagentRuns": subagent_items,
        "toolCalls": tool_items,
        "llmCalls": llm_items,
        "events": event_items,
        "evolutionRuns": evolution_items,
        "flow": flow_items,
        "toolStatusCounts": [
            {"tool": tool, "status": status, "count": count}
            for (tool, status), count in sorted(tool_status_counts.items())
        ],
        "roles": sorted({item["role"] for item in subagent_items if item.get("role")}),
        "statuses": sorted(
            {
                item["status"]
                for item in [*subagent_items, *tool_items, *evolution_items]
                if item.get("status")
            }
        ),
    }


def _render_dashboard_html(data: dict[str, Any], *, title: str) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --panel-soft: #fbfcfe;
      --line: #d7dde7;
      --line-strong: #b9c3d3;
      --text: #182233;
      --muted: #5d6d82;
      --accent: #0f766e;
      --accent-soft: #d9f0ec;
      --warn: #a16207;
      --bad: #b42318;
      --good: #047857;
      --code: #edf1f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    button, input, select {
      font: inherit;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(245, 247, 250, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }
    .topbar-inner {
      max-width: 1680px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) minmax(360px, 640px);
      gap: 14px;
      align-items: end;
    }
    h1 {
      margin: 0;
      font-size: 21px;
      font-weight: 740;
      letter-spacing: 0;
    }
    .subhead {
      margin-top: 4px;
      color: var(--muted);
      overflow-wrap: anywhere;
      font-size: 12px;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 150px 150px;
      gap: 8px;
    }
    .controls input, .controls select {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--text);
    }
    .metrics {
      max-width: 1680px;
      margin: 0 auto;
      padding: 14px 18px 0;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px;
      min-height: 70px;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      line-height: 1.1;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    .dashboard {
      max-width: 1680px;
      margin: 0 auto;
      padding: 14px 18px 18px;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 390px;
      gap: 14px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }
    .panel-title {
      font-size: 14px;
      font-weight: 720;
      margin: 0;
    }
    .panel-body {
      padding: 12px;
    }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }
    .tab, .tiny-action {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 7px 10px;
      cursor: pointer;
    }
    .tab[aria-selected="true"] {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #134e4a;
      font-weight: 700;
    }
    .side-list {
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 235px);
      overflow: auto;
    }
    .run-card, .flow-card, .item-card {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      cursor: pointer;
    }
    .run-card.active, .flow-card.active, .item-card.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .eyebrow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .card-title {
      font-weight: 720;
      overflow-wrap: anywhere;
    }
    .card-text {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .status {
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      font-weight: 720;
      background: var(--code);
      color: var(--muted);
      white-space: nowrap;
    }
    .status.ok {
      color: var(--good);
      background: #e0f3eb;
    }
    .status.warn {
      color: var(--warn);
      background: #fbefd4;
    }
    .status.bad {
      color: var(--bad);
      background: #fde2df;
    }
    .flow-lane {
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding-bottom: 8px;
    }
    .flow-card {
      min-width: 245px;
      max-width: 310px;
      position: relative;
    }
    .flow-card::after {
      content: ">";
      position: absolute;
      right: -10px;
      top: 42%;
      color: var(--line-strong);
      font-weight: 800;
    }
    .flow-card:last-child::after {
      content: "";
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .stack {
      display: grid;
      gap: 10px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 6px;
    }
    .chip {
      border-radius: 999px;
      padding: 2px 7px;
      background: var(--accent-soft);
      color: #134e4a;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      background: var(--panel-soft);
    }
    tr:last-child td {
      border-bottom: 0;
    }
    tr.selectable {
      cursor: pointer;
    }
    tr.selectable:hover {
      background: #f8fafc;
    }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    code {
      display: inline;
      background: var(--code);
      border-radius: 4px;
      padding: 1px 4px;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: var(--code);
      border-radius: 6px;
      padding: 10px;
      max-height: 360px;
      overflow: auto;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: 110px minmax(0, 1fr);
      gap: 7px 10px;
      margin-bottom: 12px;
    }
    .detail-key {
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
    }
    .detail-value {
      overflow-wrap: anywhere;
    }
    .empty {
      color: var(--muted);
      padding: 18px;
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .muted {
      color: var(--muted);
    }
    .count {
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 1180px) {
      .dashboard {
        grid-template-columns: 280px minmax(0, 1fr);
      }
      .details-panel {
        grid-column: 1 / -1;
      }
    }
    @media (max-width: 760px) {
      .topbar-inner, .controls, .dashboard, .grid-2 {
        grid-template-columns: 1fr;
      }
      .side-list {
        max-height: none;
      }
      table, thead, tbody, th, td, tr {
        display: block;
      }
      thead {
        display: none;
      }
      tr {
        border-bottom: 1px solid var(--line);
      }
      td {
        border-bottom: 0;
      }
      td::before {
        content: attr(data-label);
        display: block;
        color: var(--muted);
        font-size: 12px;
        font-weight: 720;
      }
    }
  </style>
</head>
<body>
  <script id="trajectory-data" type="application/json">__DATA__</script>
  <header class="topbar">
    <div class="topbar-inner">
      <div>
        <h1>__TITLE__</h1>
        <div class="subhead">
          <span id="source-dir"></span>
          <span id="generated-at"></span>
        </div>
      </div>
      <div class="controls">
        <input id="search" type="search" placeholder="Filter runs, tools, skills, artifacts, prompts">
        <select id="role-filter" aria-label="Role filter"></select>
        <select id="status-filter" aria-label="Status filter"></select>
      </div>
    </div>
  </header>
  <section class="metrics" id="metrics" aria-label="Trajectory summary"></section>
  <main class="dashboard">
    <aside class="panel">
      <div class="panel-head">
        <h2 class="panel-title">Runs</h2>
        <span class="count" id="run-count"></span>
      </div>
      <div class="panel-body">
        <div class="side-list" id="run-list"></div>
      </div>
    </aside>
    <section class="panel">
      <div class="panel-head">
        <h2 class="panel-title" id="view-title">Dispatch Flow</h2>
        <span class="count" id="view-count"></span>
      </div>
      <div class="panel-body">
        <div class="tabs" id="tabs"></div>
        <div id="view"></div>
      </div>
    </section>
    <aside class="panel details-panel">
      <div class="panel-head">
        <h2 class="panel-title">Details</h2>
        <button class="tiny-action" id="copy-detail" type="button">Copy JSON</button>
      </div>
      <div class="panel-body" id="details"></div>
    </aside>
  </main>
  <script>
    const DATA = JSON.parse(document.getElementById('trajectory-data').textContent);
    const VIEWS = [
      ['flow', 'Dispatch Flow'],
      ['runs', 'Subagent Runs'],
      ['skills', 'Skill And Memory'],
      ['tools', 'Tool Calls'],
      ['llm', 'LLM Calls'],
      ['meta', 'MetaAgent Routing'],
      ['events', 'Events'],
      ['evolution', 'Evolution Runs']
    ];
    const state = {
      view: 'flow',
      query: '',
      role: 'all',
      status: 'all',
      selectedId: ''
    };
    const allItems = [
      ...DATA.flow,
      ...DATA.subagentRuns,
      ...DATA.metaRuns,
      ...DATA.toolCalls,
      ...DATA.llmCalls,
      ...DATA.events,
      ...DATA.evolutionRuns
    ];
    const byId = new Map(allItems.map((item) => [item.id, item]));

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function shortText(value, limit) {
      const text = String(value ?? '');
      return text.length <= limit ? text : text.slice(0, Math.max(0, limit - 3)) + '...';
    }

    function statusTone(value) {
      const text = String(value ?? '').toLowerCase();
      if (text.includes('fail') || text.includes('error') || text.includes('abort') || text.includes('reject')) return 'bad';
      if (text.includes('budget') || text.includes('skip') || text.includes('warn') || text.includes('degraded')) return 'warn';
      if (text.includes('complete') || text === 'ok' || text.includes('promoted')) return 'ok';
      return '';
    }

    function statusBadge(value) {
      if (!value) return '';
      return '<span class="status ' + statusTone(value) + '">' + escapeHtml(value) + '</span>';
    }

    function itemMatches(item) {
      const query = state.query.trim().toLowerCase();
      const text = String(item.search || JSON.stringify(item)).toLowerCase();
      const roleOk = state.role === 'all' || item.role === state.role || item.target === state.role;
      const statusOk = state.status === 'all' || item.status === state.status;
      return (!query || text.includes(query)) && roleOk && statusOk;
    }

    function filtered(items) {
      return items.filter(itemMatches);
    }

    function setSelected(id) {
      state.selectedId = id || state.selectedId;
      renderAll();
    }

    function selectView(view) {
      state.view = view;
      renderAll();
    }

    function renderMetrics() {
      const labels = {
        meta_agent_runs: 'MetaAgent runs',
        subagent_runs: 'Subagent runs',
        llm_calls: 'LLM calls',
        tool_calls: 'Tool calls',
        events: 'Events',
        evolution_runs: 'Evolution runs'
      };
      document.getElementById('metrics').innerHTML = Object.entries(DATA.counts).map(([key, value]) => (
        '<div class="metric"><strong>' + escapeHtml(value) + '</strong><span>' + escapeHtml(labels[key] || key) + '</span></div>'
      )).join('');
    }

    function renderFilters() {
      const role = document.getElementById('role-filter');
      const status = document.getElementById('status-filter');
      role.innerHTML = '<option value="all">All roles</option>' + DATA.roles.map((value) => (
        '<option value="' + escapeHtml(value) + '">' + escapeHtml(value) + '</option>'
      )).join('');
      status.innerHTML = '<option value="all">All statuses</option>' + DATA.statuses.map((value) => (
        '<option value="' + escapeHtml(value) + '">' + escapeHtml(value) + '</option>'
      )).join('');
      role.value = state.role;
      status.value = state.status;
    }

    function renderTabs() {
      document.getElementById('tabs').innerHTML = VIEWS.map(([id, label]) => (
        '<button class="tab" type="button" data-view="' + id + '" aria-selected="' + (state.view === id) + '">' + escapeHtml(label) + '</button>'
      )).join('');
      document.querySelectorAll('[data-view]').forEach((button) => {
        button.addEventListener('click', () => selectView(button.dataset.view));
      });
    }

    function renderRunList() {
      const items = filtered(DATA.subagentRuns);
      document.getElementById('run-count').textContent = String(items.length);
      document.getElementById('run-list').innerHTML = items.length ? items.map((item) => (
        '<button class="run-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + escapeHtml(item.id) + '">' +
          '<div class="eyebrow"><span>Stage ' + escapeHtml(item.stage) + '</span>' + statusBadge(item.status) + '</div>' +
          '<div class="card-title">' + escapeHtml(item.role) + '</div>' +
          '<div class="card-text">' + escapeHtml(shortText(item.failure || item.instruction, 120)) + '</div>' +
          '<div class="chips"><span class="chip">' + item.llmCount + ' LLM</span><span class="chip">' + item.toolCount + ' tools</span><span class="chip">' + item.skillCount + ' skills</span></div>' +
        '</button>'
      )).join('') : '<div class="empty">No matching runs.</div>';
      document.querySelectorAll('[data-id]').forEach((node) => {
        node.addEventListener('click', () => setSelected(node.dataset.id));
      });
    }

    function renderView() {
      const viewMeta = VIEWS.find(([id]) => id === state.view) || VIEWS[0];
      document.getElementById('view-title').textContent = viewMeta[1];
      if (state.view === 'flow') return renderFlow();
      if (state.view === 'runs') return renderRuns();
      if (state.view === 'skills') return renderSkills();
      if (state.view === 'tools') return renderTools();
      if (state.view === 'llm') return renderLLM();
      if (state.view === 'meta') return renderMeta();
      if (state.view === 'events') return renderEvents();
      if (state.view === 'evolution') return renderEvolution();
    }

    function renderFlow() {
      const items = filtered(DATA.flow);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? (
        '<div class="flow-lane">' + items.map((item) => (
          '<button class="flow-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + escapeHtml(item.id) + '">' +
            '<div class="eyebrow"><span>' + escapeHtml(item.kind) + '</span>' + statusBadge(item.status || item.action) + '</div>' +
            '<div class="card-title">' + escapeHtml(item.title) + '</div>' +
            '<div class="card-text">' + escapeHtml(shortText(item.summary, 170)) + '</div>' +
          '</button>'
        )).join('') + '</div>'
      ) : '<div class="empty">No matching flow nodes.</div>';
      bindItemClicks();
    }

    function renderRuns() {
      const items = filtered(DATA.subagentRuns);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? (
        '<div class="grid-2">' + items.map(runCard).join('') + '</div>'
      ) : '<div class="empty">No matching subagent runs.</div>';
      bindItemClicks();
    }

    function runCard(item) {
      return '<button class="item-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + escapeHtml(item.id) + '">' +
        '<div class="eyebrow"><span>' + escapeHtml(item.runRef) + '</span>' + statusBadge(item.status) + '</div>' +
        '<div class="card-title">' + escapeHtml(item.role) + ' stage ' + escapeHtml(item.stage) + '</div>' +
        '<div class="card-text">' + escapeHtml(shortText(item.failure || item.instruction, 180)) + '</div>' +
        '<div class="chips"><span class="chip">' + item.llmCount + ' LLM</span><span class="chip">' + item.toolCount + ' tools</span><span class="chip">' + item.artifacts.length + ' artifacts</span></div>' +
      '</button>';
    }

    function renderSkills() {
      const runs = filtered(DATA.subagentRuns);
      const total = runs.reduce((sum, item) => sum + item.skills.length + item.workflowNodes.length, 0);
      document.getElementById('view-count').textContent = String(total);
      document.getElementById('view').innerHTML = runs.length ? (
        '<div class="stack">' + runs.map((run) => (
          '<button class="item-card ' + (state.selectedId === run.id ? 'active' : '') + '" type="button" data-id="' + escapeHtml(run.id) + '">' +
            '<div class="eyebrow"><span>' + escapeHtml(run.role) + '</span>' + statusBadge(run.status) + '</div>' +
            '<div class="card-title">' + run.skills.length + ' skills, ' + run.memoryItems.length + ' memories, ' + run.workflowNodes.length + ' workflow nodes</div>' +
            '<div class="chips">' + run.skills.slice(0, 8).map((skill) => '<span class="chip">' + escapeHtml(shortText(skill.label, 70)) + '</span>').join('') + '</div>' +
            '<div class="card-text">' + escapeHtml(shortText(run.updateSummary, 180)) + '</div>' +
          '</button>'
        )).join('') + '</div>'
      ) : '<div class="empty">No matching skill or memory records.</div>';
      bindItemClicks();
    }

    function renderTools() {
      const items = filtered(DATA.toolCalls);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? table(
        ['Tool', 'Role', 'Stage', 'Status', 'Result'],
        items.map((item) => [
          '<code>' + escapeHtml(item.tool) + '</code>',
          escapeHtml(item.role),
          escapeHtml(item.stage),
          statusBadge(item.status),
          escapeHtml(shortText(item.result, 160))
        ]),
        items.map((item) => item.id)
      ) : '<div class="empty">No matching tool calls.</div>';
      bindItemClicks();
    }

    function renderLLM() {
      const items = filtered(DATA.llmCalls);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? table(
        ['Call', 'Role', 'Stage', 'Model', 'Action', 'Usage'],
        items.map((item) => [
          '<code>' + escapeHtml(item.callRef) + '</code>',
          escapeHtml(item.role),
          escapeHtml(item.stage),
          escapeHtml(shortText(item.model, 36)),
          escapeHtml(item.action),
          escapeHtml(item.usage)
        ]),
        items.map((item) => item.id)
      ) : '<div class="empty">No matching LLM calls.</div>';
      bindItemClicks();
    }

    function renderMeta() {
      const items = filtered(DATA.metaRuns);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? table(
        ['Action', 'Target', 'Run', 'Instruction'],
        items.map((item) => [
          statusBadge(item.action),
          escapeHtml(item.target),
          '<code>' + escapeHtml(item.runRef) + '</code>',
          escapeHtml(shortText(item.instruction, 180))
        ]),
        items.map((item) => item.id)
      ) : '<div class="empty">No matching MetaAgent decisions.</div>';
      bindItemClicks();
    }

    function renderEvents() {
      const items = filtered(DATA.events);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? table(
        ['Time', 'Type', 'Subject', 'Run'],
        items.map((item) => [
          escapeHtml(item.time),
          '<code>' + escapeHtml(item.eventType) + '</code>',
          escapeHtml(item.subject),
          '<code>' + escapeHtml(item.runRef) + '</code>'
        ]),
        items.map((item) => item.id)
      ) : '<div class="empty">No matching events.</div>';
      bindItemClicks();
    }

    function renderEvolution() {
      const items = filtered(DATA.evolutionRuns);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? table(
        ['Run', 'Mode', 'Backend', 'Status', 'Sources'],
        items.map((item) => [
          '<code>' + escapeHtml(item.runRef) + '</code>',
          escapeHtml(item.mode),
          '<code>' + escapeHtml(item.backendId) + '</code>',
          statusBadge(item.status),
          escapeHtml(shortText(item.sources.join(', '), 160))
        ]),
        items.map((item) => item.id)
      ) : '<div class="empty">No evolution records.</div>';
      bindItemClicks();
    }

    function table(headers, rows, ids) {
      const head = '<thead><tr>' + headers.map((h) => '<th>' + escapeHtml(h) + '</th>').join('') + '</tr></thead>';
      const body = '<tbody>' + rows.map((row, index) => (
        '<tr class="selectable" data-id="' + escapeHtml(ids[index]) + '">' +
          row.map((cell, cellIndex) => '<td data-label="' + escapeHtml(headers[cellIndex]) + '">' + cell + '</td>').join('') +
        '</tr>'
      )).join('') + '</tbody>';
      return '<table>' + head + body + '</table>';
    }

    function bindItemClicks() {
      document.querySelectorAll('[data-id]').forEach((node) => {
        node.addEventListener('click', () => setSelected(node.dataset.id));
      });
    }

    function renderDetails() {
      const item = byId.get(state.selectedId) || DATA.subagentRuns[0] || DATA.metaRuns[0] || DATA.flow[0];
      if (!item) {
        document.getElementById('details').innerHTML = '<div class="empty">Select an item to inspect.</div>';
        return;
      }
      state.selectedId = item.id;
      const summary = item.detail || {};
      const rows = Object.entries(summary).map(([key, value]) => (
        '<div class="detail-key">' + escapeHtml(key) + '</div><div class="detail-value">' + renderDetailValue(value) + '</div>'
      )).join('');
      document.getElementById('details').innerHTML =
        '<div class="detail-grid">' + rows + '</div>' +
        '<pre>' + escapeHtml(JSON.stringify(item.raw || item, null, 2)) + '</pre>';
    }

    function renderDetailValue(value) {
      if (Array.isArray(value)) {
        if (!value.length) return '<span class="muted">empty</span>';
        return '<div class="chips">' + value.slice(0, 16).map((item) => '<span class="chip">' + escapeHtml(shortText(typeof item === 'string' ? item : JSON.stringify(item), 90)) + '</span>').join('') + '</div>';
      }
      if (value && typeof value === 'object') {
        return '<code>' + escapeHtml(shortText(JSON.stringify(value), 180)) + '</code>';
      }
      return escapeHtml(value ?? '');
    }

    function renderAll() {
      renderMetrics();
      renderFilters();
      renderTabs();
      renderRunList();
      renderView();
      renderDetails();
      document.getElementById('source-dir').textContent = DATA.sourceDir;
      document.getElementById('generated-at').textContent = 'Generated ' + DATA.generatedAt;
    }

    document.getElementById('search').addEventListener('input', (event) => {
      state.query = event.target.value;
      renderAll();
    });
    document.getElementById('role-filter').addEventListener('change', (event) => {
      state.role = event.target.value;
      renderAll();
    });
    document.getElementById('status-filter').addEventListener('change', (event) => {
      state.status = event.target.value;
      renderAll();
    });
    document.getElementById('copy-detail').addEventListener('click', async () => {
      const item = byId.get(state.selectedId);
      if (!item || !navigator.clipboard) return;
      await navigator.clipboard.writeText(JSON.stringify(item.raw || item, null, 2));
    });

    state.selectedId = (DATA.subagentRuns[0] || DATA.metaRuns[0] || DATA.flow[0] || {}).id || '';
    renderAll();
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__TITLE__", _h(title))
        .replace("__DATA__", _script_json(data))
    )


def _meta_dashboard_item(run: Any) -> dict[str, Any]:
    decision = run.decision
    metadata = _asdict(decision.metadata)
    action = _display_value(decision.action)
    target = decision.target_role or ""
    instruction = decision.instruction or ""
    raw = {
        "run_ref": run.run_ref,
        "task_id": run.task_id,
        "time": _time(run.created_at),
        "action": action,
        "target": target,
        "instruction": instruction,
        "metadata": metadata,
    }
    return {
        "id": run.run_ref,
        "kind": "meta",
        "runRef": run.run_ref,
        "taskId": run.task_id,
        "time": _time(run.created_at),
        "action": action,
        "target": target,
        "title": target or action,
        "instruction": _short_text(instruction, 1200),
        "summary": _short_text(instruction or _json_preview(metadata), 260),
        "status": action,
        "detail": {
            "run": run.run_ref,
            "task": run.task_id,
            "action": action,
            "target": target,
            "metadata": _metadata_labels(metadata),
        },
        "raw": _compact_value(raw, 3200),
        "search": _search_text(raw),
    }


def _subagent_dashboard_item(
    run: Any,
    *,
    llm_count: int,
    tool_count: int,
    started_at: str | None,
) -> dict[str, Any]:
    metadata = _asdict(run.metadata)
    status = str(metadata.get("status") or "recorded")
    failure = str(metadata.get("failure_reason") or "")
    skills = _skill_summaries(run)
    memory_items = _memory_summaries(run)
    workflow_nodes = _workflow_node_summaries(metadata)
    artifacts = _artifact_items(run.artifact_refs)
    update_summary = _update_summary(
        _asdict(metadata.get("skill_update_result")),
        _asdict(
            metadata.get("agent_memory_update_result")
            or metadata.get("task_memory_update_result")
            or metadata.get("memory_update_result")
        ),
    )
    raw = {
        "run_ref": run.run_ref,
        "task_id": run.task_id,
        "stage": run.stage_index,
        "role": run.role,
        "status": status,
        "failure": failure,
        "instruction": run.instruction,
        "skills": skills,
        "memory_items": memory_items,
        "workflow_nodes": workflow_nodes,
        "artifacts": artifacts,
        "updates": update_summary,
    }
    return {
        "id": run.run_ref,
        "kind": "subagent",
        "runRef": run.run_ref,
        "taskId": run.task_id,
        "time": started_at or f"stage-{run.stage_index:06d}",
        "stage": run.stage_index,
        "role": run.role,
        "status": status,
        "title": f"{run.role} stage {run.stage_index}",
        "instruction": _short_text(run.instruction, 1400),
        "failure": _short_text(failure, 900),
        "summary": _short_text(failure or run.instruction, 280),
        "llmCount": llm_count,
        "toolCount": tool_count,
        "skillCount": len(skills),
        "skills": skills,
        "memoryItems": memory_items,
        "workflowNodes": workflow_nodes,
        "artifacts": artifacts,
        "updateSummary": update_summary,
        "detail": {
            "run": run.run_ref,
            "role": run.role,
            "stage": run.stage_index,
            "status": status,
            "llm calls": llm_count,
            "tool calls": tool_count,
            "skills": [item["label"] for item in skills],
            "memory items": [item["content"] for item in memory_items],
            "artifacts": [item["label"] for item in artifacts],
            "failure": failure,
        },
        "raw": _compact_value(raw, 5200),
        "search": _search_text(raw),
    }


def _tool_dashboard_item(call: Any) -> dict[str, Any]:
    record = call.record
    artifacts = _artifact_items(call.artifact_refs or record.result.artifact_refs)
    raw = {
        "record_ref": call.record_ref,
        "run_ref": call.run_ref,
        "tool_call_id": call.tool_call_id,
        "tool": call.tool_name,
        "role": call.role or "",
        "stage": call.runtime_stage or "",
        "status": record.result.status,
        "arguments": record.tool_call.arguments,
        "result": record.result.content,
        "metadata": record.result.metadata,
        "artifacts": artifacts,
    }
    return {
        "id": call.record_ref,
        "kind": "tool",
        "runRef": call.run_ref,
        "time": _time(call.created_at),
        "role": call.role or "",
        "stage": call.runtime_stage or "",
        "tool": call.tool_name,
        "status": record.result.status,
        "result": _short_text(record.result.content, 900),
        "artifacts": artifacts,
        "detail": {
            "tool": call.tool_name,
            "status": record.result.status,
            "role": call.role or "",
            "run": call.run_ref,
            "stage": call.runtime_stage or "",
            "artifacts": [item["label"] for item in artifacts],
        },
        "raw": _compact_value(raw, 4200),
        "search": _search_text(raw),
    }


def _llm_dashboard_item(call: Any, *, role: str) -> dict[str, Any]:
    metadata = _asdict(call.metadata)
    raw = {
        "call_ref": call.call_ref,
        "run_ref": call.run_ref,
        "backend_id": call.backend_id,
        "model": call.model,
        "role": role or str(metadata.get("role") or ""),
        "stage": str(metadata.get("runtime_stage") or ""),
        "action": str(metadata.get("action") or ""),
        "usage": _usage(metadata),
        "input": _messages_preview(call.input_messages[-2:], limit=900),
        "output": _messages_preview(call.output_messages, limit=900),
    }
    return {
        "id": call.call_ref,
        "kind": "llm",
        "callRef": call.call_ref,
        "runRef": call.run_ref,
        "backendId": call.backend_id,
        "model": call.model,
        "role": raw["role"],
        "stage": raw["stage"],
        "action": raw["action"],
        "usage": raw["usage"],
        "input": raw["input"],
        "output": raw["output"],
        "detail": {
            "call": call.call_ref,
            "run": call.run_ref,
            "role": raw["role"],
            "stage": raw["stage"],
            "model": call.model,
            "action": raw["action"],
            "usage": raw["usage"],
        },
        "raw": _compact_value(raw, 4200),
        "search": _search_text(raw),
    }


def _event_dashboard_item(event: Any) -> dict[str, Any]:
    metadata = _asdict(event.metadata)
    raw = {
        "event_ref": event.event_ref,
        "event_type": event.event_type,
        "subject_type": event.subject_type,
        "subject_ref": event.subject_ref or "",
        "task_id": event.task_id or "",
        "run_ref": event.run_ref or "",
        "time": _time(event.created_at),
        "metadata": metadata,
    }
    return {
        "id": event.event_ref,
        "kind": "event",
        "eventType": event.event_type,
        "subject": f"{event.subject_type}: {event.subject_ref or ''}",
        "runRef": event.run_ref or "",
        "role": str(metadata.get("role") or ""),
        "status": str(metadata.get("status") or ""),
        "time": _time(event.created_at),
        "detail": {
            "event": event.event_type,
            "subject": f"{event.subject_type}: {event.subject_ref or ''}",
            "run": event.run_ref or "",
            "metadata": _metadata_labels(metadata),
        },
        "raw": _compact_value(raw, 3000),
        "search": _search_text(raw),
    }


def _evolution_dashboard_item(run: Any) -> dict[str, Any]:
    result = _asdict(run.result)
    raw = {
        "run_ref": run.run_ref,
        "mode": _display_value(run.mode),
        "backend_id": run.backend_id,
        "status": run.result_status,
        "sources": run.training_trajectory_refs,
        "result": result,
    }
    return {
        "id": run.run_ref,
        "kind": "evolution",
        "runRef": run.run_ref,
        "mode": _display_value(run.mode),
        "backendId": run.backend_id,
        "status": run.result_status,
        "sources": list(run.training_trajectory_refs),
        "detail": {
            "run": run.run_ref,
            "mode": _display_value(run.mode),
            "backend": run.backend_id,
            "status": run.result_status,
            "sources": list(run.training_trajectory_refs),
        },
        "raw": _compact_value(raw, 4200),
        "search": _search_text(raw),
    }


def _flow_dashboard_item(item: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "id": item["id"],
        "kind": "MetaAgent" if item["kind"] == "meta" else "Subagent",
        "time": item.get("time", ""),
        "role": item.get("role", ""),
        "target": item.get("target", ""),
        "status": item.get("status", ""),
        "action": item.get("action", ""),
        "title": item.get("title", item.get("id", "")),
        "summary": item.get("summary", ""),
    }
    return {
        **compact,
        "detail": item.get("detail", {}),
        "raw": compact,
        "search": _search_text(compact),
    }


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _skill_summaries(run: Any) -> list[dict[str, Any]]:
    summaries = []
    for skill in _skill_items(run):
        label = _skill_label(skill)
        summaries.append(
            {
                "skill_id": str(skill.get("skill_id") or ""),
                "name": str(skill.get("name") or ""),
                "label": label,
                "required_tools": [str(item) for item in skill.get("required_tools", [])],
                "metadata": _compact_value(skill.get("metadata", {}), 900),
            }
        )
    return summaries


def _memory_summaries(run: Any) -> list[dict[str, Any]]:
    summaries = []
    for item in _memory_items(run):
        summaries.append(
            {
                "memory_id": str(item.get("memory_id") or ""),
                "content": _short_text(str(item.get("content") or ""), 500),
                "score": item.get("score"),
                "metadata": _compact_value(item.get("metadata", {}), 700),
            }
        )
    return summaries


def _workflow_node_summaries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for node in _workflow_nodes(metadata):
        node_metadata = _asdict(node.get("metadata"))
        if not isinstance(node_metadata, dict):
            node_metadata = {}
        tool_names = []
        for call in node.get("tool_calls") or []:
            payload = _asdict(call)
            if isinstance(payload, dict):
                tool_name = payload.get("name") or _asdict(payload.get("tool_call")).get("name")
                if tool_name:
                    tool_names.append(str(tool_name))
        summaries.append(
            {
                "node_id": str(node.get("node_id") or ""),
                "skill_id": str(node.get("skill_id") or node_metadata.get("skill_id") or ""),
                "name": str(node_metadata.get("node_name") or node.get("name") or ""),
                "status": str(node.get("status") or ""),
                "tool_names": tool_names,
                "output_summary": _short_text(str(node.get("output_summary") or ""), 500),
            }
        )
    return summaries


def _artifact_items(artifacts: list[Any]) -> list[dict[str, Any]]:
    items = []
    for artifact in artifacts:
        payload = _asdict(artifact)
        if not isinstance(payload, dict):
            continue
        metadata = _asdict(payload.get("metadata"))
        if not isinstance(metadata, dict):
            metadata = {}
        uri = str(payload.get("uri") or "")
        label = str(
            metadata.get("filename")
            or metadata.get("artifact_kind")
            or Path(uri).name
            or uri
        )
        items.append(
            {
                "label": label,
                "uri": uri,
                "type": str(payload.get("type") or ""),
                "metadata": _compact_value(metadata, 900),
            }
        )
    return items


def _compact_value(value: Any, limit: int) -> Any:
    payload = _asdict(value)
    try:
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        text = str(payload)
    if len(text) <= limit:
        return payload
    return {"preview": _short_text(text, limit), "truncated": True}


def _search_text(value: Any) -> str:
    try:
        text = json.dumps(_asdict(value), ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return _short_text(text, 3000)


def _display_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _metadata_labels(metadata: dict[str, Any]) -> str:
    if not metadata:
        return "empty"
    labels = []
    for key in ("work_item_id", "route", "status", "failure_reason", "trigger", "action"):
        value = metadata.get(key)
        if value is not None:
            labels.append(f"{key}={_short_text(str(value), 80)}")
    if labels:
        return ", ".join(labels)
    return ", ".join(sorted(metadata.keys())[:4])


def _skill_items(run: Any) -> list[dict[str, Any]]:
    bundle = _asdict(run.skill_bundle)
    skills = bundle.get("skills")
    if isinstance(skills, list):
        return [_asdict(skill) for skill in skills if isinstance(_asdict(skill), dict)]
    return []


def _skill_label(skill: dict[str, Any]) -> str:
    skill_id = str(skill.get("skill_id") or "")
    name = str(skill.get("name") or "")
    if skill_id and name:
        return f"{skill_id} - {name}"
    return skill_id or name


def _memory_items(run: Any) -> list[dict[str, Any]]:
    bundle = _asdict(run.memory_bundle)
    items = bundle.get("items")
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        payload = _asdict(item)
        if isinstance(payload, dict):
            result.append(
                {
                    "memory_id": payload.get("memory_id"),
                    "content": payload.get("content"),
                    "score": payload.get("score"),
                    "metadata": payload.get("metadata", {}),
                }
            )
    return result


def _workflow_nodes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = metadata.get("node_execution_records")
    if isinstance(records, list):
        return [_asdict(record) for record in records if isinstance(_asdict(record), dict)]
    plan = _asdict(metadata.get("workflow_plan"))
    nodes = plan.get("nodes")
    if isinstance(nodes, list):
        return [_asdict(node) for node in nodes if isinstance(_asdict(node), dict)]
    return []


def _update_summary(skill_update: dict[str, Any], memory_update: dict[str, Any]) -> str:
    parts = []
    if skill_update:
        parts.append(f"skill={skill_update.get('status', 'recorded')}")
    if memory_update:
        parts.append(f"memory={memory_update.get('status', 'recorded')}")
    return ", ".join(parts) if parts else "no updates recorded"


def _usage(metadata: dict[str, Any]) -> str:
    raw_usage = _asdict(metadata.get("usage"))
    if not raw_usage:
        raw_response = _asdict(metadata.get("raw_response"))
        raw_usage = _asdict(raw_response.get("usage"))
    if not raw_usage:
        return ""
    parts = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in raw_usage:
            parts.append(f"{key.replace('_tokens', '')}={raw_usage[key]}")
    return ", ".join(parts) or _short_text(_json_preview(raw_usage), 120)


def _messages_preview(messages: list[Any], *, limit: int) -> str:
    parts = []
    for message in messages:
        payload = _asdict(message)
        role = payload.get("role", "message")
        content = payload.get("content", "")
        parts.append(f"{role}: {content}")
    return _short_text("\n\n".join(parts), limit)


def _group_by(items: list[Any], attr: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        grouped[str(getattr(item, attr))].append(item)
    return dict(grouped)


def _json_preview(value: Any, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    payload = _asdict(value)
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    except TypeError:
        text = str(payload)
    return _short_text(text, limit)


def _short_text(value: Any, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + f"\n... <truncated {len(text) - limit + 20} chars>"


def _asdict(value: Any) -> Any:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _asdict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_asdict(item) for item in value]
    if isinstance(value, tuple):
        return [_asdict(item) for item in value]
    return value


def _created_sort_key(record: Any) -> str:
    return str(getattr(record, "created_at", ""))


def _subagent_sort_key(record: Any) -> tuple[int, str]:
    return (getattr(record, "stage_index", 0), getattr(record, "run_ref", ""))


def _llm_sort_key(record: Any) -> tuple[str, int, str]:
    metadata = _asdict(getattr(record, "metadata", {}))
    step = metadata.get("step_index")
    if not isinstance(step, int):
        step = 0
    return (str(getattr(record, "run_ref", "")), step, str(getattr(record, "call_ref", "")))


def _tool_sort_key(record: Any) -> tuple[str, int, str]:
    step = getattr(record, "step_index", None)
    if not isinstance(step, int):
        step = 0
    return (str(getattr(record, "created_at", "")), step, str(getattr(record, "record_ref", "")))


def _time(value: Any) -> str:
    return str(value)


def _default_title(
    *,
    lab_root: Path | str | None,
    trajectory_dir: Path | str | None,
) -> str:
    if lab_root is not None:
        return f"EvoLab Trajectory: {Path(lab_root).name}"
    if trajectory_dir is not None:
        return f"EvoLab Trajectory: {Path(trajectory_dir).name}"
    return "EvoLab Trajectory"


def _resolve_output_path(
    *,
    output_path: Path | str | None,
    lab_root: Path | None,
    trajectory_dir: Path,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    if lab_root is not None:
        return lab_root / "artifacts" / "trajectory_view.html"
    return trajectory_dir / "trajectory_view.html"


def _h(value: Any) -> str:
    return escape("" if value is None else str(value), quote=False)
