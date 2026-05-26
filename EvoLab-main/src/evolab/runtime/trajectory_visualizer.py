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
    lab_root_path = (
        Path(lab_root)
        if lab_root is not None
        else _infer_lab_root_from_trajectory_dir(source_dir)
    )
    snapshot = load_trajectory_snapshot(source_dir, lab_root=lab_root_path)
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


def load_trajectory_snapshot(
    trajectory_dir: Path | str,
    *,
    lab_root: Path | str | None = None,
) -> dict[str, Any]:
    registry = FileTrajectoryRegistry(Path(trajectory_dir))
    meta_runs = registry.list_meta_agent_runs()
    subagent_runs = registry.list_subagent_runs()
    llm_calls = registry.list_llm_calls()
    tool_calls = registry.list_tool_call_records()
    events = registry.list_events()
    evolution_runs = registry.list_evolution_runs()
    dynamic_workflows = _load_dynamic_workflows(Path(lab_root) if lab_root is not None else None)
    counts = {
        "meta_agent_runs": len(meta_runs),
        "subagent_runs": len(subagent_runs),
        "llm_calls": len(llm_calls),
        "tool_calls": len(tool_calls),
        "events": len(events),
        "evolution_runs": len(evolution_runs),
    }
    if dynamic_workflows:
        counts.update(
            {
                "dynamic_workflows": len(dynamic_workflows),
                "dynamic_subagent_specs": sum(
                    len(item.get("subagent_specs") or []) for item in dynamic_workflows
                ),
                "dynamic_subagent_records": sum(
                    len(item.get("records") or []) for item in dynamic_workflows
                ),
            }
        )
    return {
        "meta_runs": meta_runs,
        "subagent_runs": subagent_runs,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "events": events,
        "evolution_runs": evolution_runs,
        "dynamic_workflows": dynamic_workflows,
        "counts": counts,
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
    dynamic_workflows = sorted(
        snapshot.get("dynamic_workflows", []),
        key=_dynamic_workflow_sort_key,
    )

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
    dynamic_items = [_dynamic_workflow_dashboard_item(run) for run in dynamic_workflows]
    dynamic_subagent_items = [
        spec for item in dynamic_items for spec in item.get("subagentSpecs", [])
    ]
    dynamic_by_run = {
        item["runRef"]: item
        for item in dynamic_subagent_items
        if item.get("runRef")
    }
    dynamic_by_node = _dynamic_subagents_by_node(dynamic_subagent_items)
    meta_items = [_meta_dashboard_item(run) for run in meta_runs]
    subagent_items = [
        _subagent_dashboard_item(
            run,
            llm_count=len(llm_by_run.get(run.run_ref, [])),
            tool_count=len(tools_by_run.get(run.run_ref, [])),
            started_at=started_by_ref.get(run.run_ref),
            dynamic_subagent=_dynamic_subagent_for_run(
                run,
                dynamic_by_run=dynamic_by_run,
                dynamic_by_node=dynamic_by_node,
            ),
        )
        for run in subagent_runs
    ]
    tool_items = [_tool_dashboard_item(call) for call in tool_calls]
    llm_items = [_llm_dashboard_item(call, role=run_roles.get(call.run_ref, "")) for call in llm_calls]
    event_items = [_event_dashboard_item(event) for event in events]
    evolution_items = [_evolution_dashboard_item(run) for run in evolution_runs]
    overview_items = _evolution_overview_items(
        dynamic_workflows=dynamic_items,
        subagent_runs=subagent_items,
    )
    overview_node_items = [
        node for item in overview_items for node in item.get("nodes", [])
    ]
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
        "dynamicWorkflows": dynamic_items,
        "dynamicSubagents": dynamic_subagent_items,
        "evolutionOverview": overview_items,
        "evolutionOverviewNodes": overview_node_items,
        "flow": flow_items,
        "toolStatusCounts": [
            {"tool": tool, "status": status, "count": count}
            for (tool, status), count in sorted(tool_status_counts.items())
        ],
        "roles": sorted(
            {
                item["role"]
                for item in [*subagent_items, *dynamic_subagent_items, *overview_node_items]
                if item.get("role")
            }
        ),
        "statuses": sorted(
            {
                item["status"]
                for item in [
                    *subagent_items,
                    *tool_items,
                    *evolution_items,
                    *dynamic_items,
                    *dynamic_subagent_items,
                    *overview_items,
                    *overview_node_items,
                ]
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
    .script-status {
      display: inline-block;
      margin-left: 8px;
      color: var(--warn);
      font-weight: 700;
    }
    .script-status.ready {
      color: var(--good);
    }
    .script-status.error {
      color: var(--bad);
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
    .overview-list {
      display: grid;
      gap: 12px;
    }
    .overview-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      text-align: left;
    }
    .overview-card.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .overview-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      margin-bottom: 10px;
    }
    .overview-title {
      font-weight: 740;
      overflow-wrap: anywhere;
    }
    .overview-subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .overview-stats {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 5px;
    }
    .node-lane {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 8px;
    }
    .node-step {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      padding: 9px;
      min-width: 0;
      cursor: pointer;
      text-align: left;
    }
    .node-step.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
      background: #fff;
    }
    .node-step.bad {
      border-color: #efb0aa;
      background: #fff7f6;
    }
    .node-step.warn {
      border-color: #e4c06f;
      background: #fffbeb;
    }
    .node-step.ok {
      border-color: #9bd7c0;
      background: #f0fdf7;
    }
    .node-role {
      font-weight: 730;
      overflow-wrap: anywhere;
    }
    .node-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      overflow-wrap: anywhere;
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
          <span id="script-status" class="script-status">Static fallback: scripts not running</span>
        </div>
      </div>
      <div class="controls">
        <input id="search" type="search" placeholder="Filter runs, tools, skills, artifacts, prompts">
        <select id="role-filter" aria-label="Role filter"></select>
        <select id="status-filter" aria-label="Status filter"></select>
      </div>
    </div>
  </header>
  <section class="metrics" id="metrics" aria-label="Trajectory summary">__STATIC_METRICS__</section>
  <main class="dashboard">
    <aside class="panel">
      <div class="panel-head">
        <h2 class="panel-title">Runs</h2>
        <span class="count" id="run-count">__STATIC_RUN_COUNT__</span>
      </div>
      <div class="panel-body">
        <div class="side-list" id="run-list">__STATIC_RUN_LIST__</div>
      </div>
    </aside>
    <section class="panel">
      <div class="panel-head">
        <h2 class="panel-title" id="view-title">__STATIC_VIEW_TITLE__</h2>
        <span class="count" id="view-count">__STATIC_VIEW_COUNT__</span>
      </div>
      <div class="panel-body">
        <div class="tabs" id="tabs">__STATIC_TABS__</div>
        <div id="view">__STATIC_VIEW__</div>
      </div>
    </section>
    <aside class="panel details-panel">
      <div class="panel-head">
        <h2 class="panel-title">Details</h2>
        <button class="tiny-action" id="copy-detail" type="button">Copy JSON</button>
      </div>
      <div class="panel-body" id="details">__STATIC_DETAILS__</div>
    </aside>
  </main>
  <script>__LEGACY_SCRIPT__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('trajectory-data').textContent);
    const VIEWS = [
      ['overview', 'Evolution Overview'],
      ['flow', 'Dispatch Flow'],
      ['runs', 'Subagent Runs'],
      ['dynamic', 'Dynamic Subagents'],
      ['skills', 'Skill And Memory'],
      ['tools', 'Tool Calls'],
      ['llm', 'LLM Calls'],
      ['meta', 'MetaAgent Routing'],
      ['events', 'Events'],
      ['evolution', 'Evolution Runs']
    ];
    const state = {
      view: 'overview',
      query: '',
      role: 'all',
      status: 'all',
      selectedId: ''
    };
    const allItems = [
      ...DATA.flow,
      ...(DATA.evolutionOverview || []),
      ...(DATA.evolutionOverviewNodes || []),
      ...DATA.subagentRuns,
      ...(DATA.dynamicWorkflows || []),
      ...(DATA.dynamicSubagents || []),
      ...DATA.metaRuns,
      ...DATA.toolCalls,
      ...DATA.llmCalls,
      ...DATA.events,
      ...DATA.evolutionRuns
    ];
    const byId = new Map(allItems.map((item) => [item.id, item]));

    function escapeHtml(value) {
      const text = value === null || value === undefined ? '' : String(value);
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function shortText(value, limit) {
      const text = value === null || value === undefined ? '' : String(value);
      return text.length <= limit ? text : text.slice(0, Math.max(0, limit - 3)) + '...';
    }

    function statusTone(value) {
      const text = String(value === null || value === undefined ? '' : value).toLowerCase();
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
      const roleOk = state.role === 'all' || item.role === state.role || item.target === state.role || (Array.isArray(item.roles) && item.roles.includes(state.role));
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
        evolution_runs: 'Evolution runs',
        dynamic_workflows: 'Dynamic workflows',
        dynamic_subagent_specs: 'Dynamic subagent specs',
        dynamic_subagent_records: 'Dynamic subagent records'
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
          '<div class="chips"><span class="chip">' + item.llmCount + ' LLM</span><span class="chip">' + item.toolCount + ' tools</span><span class="chip">' + item.skillCount + ' skills</span>' + (item.dynamicNode ? '<span class="chip">dynamic ' + escapeHtml(item.dynamicNode) + '</span>' : '') + '</div>' +
        '</button>'
      )).join('') : '<div class="empty">No matching runs.</div>';
      document.querySelectorAll('[data-id]').forEach((node) => {
        node.addEventListener('click', () => setSelected(node.dataset.id));
      });
    }

    function renderView() {
      const viewMeta = VIEWS.find(([id]) => id === state.view) || VIEWS[0];
      document.getElementById('view-title').textContent = viewMeta[1];
      if (state.view === 'overview') return renderOverview();
      if (state.view === 'flow') return renderFlow();
      if (state.view === 'runs') return renderRuns();
      if (state.view === 'dynamic') return renderDynamic();
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

    function renderOverview() {
      const items = filtered(DATA.evolutionOverview || []);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? (
        '<div class="overview-list">' + items.map(overviewCard).join('') + '</div>'
      ) : '<div class="empty">No matching evolution overview records.</div>';
      bindItemClicks();
    }

    function overviewCard(item) {
      return '<div class="overview-card ' + (state.selectedId === item.id ? 'active' : '') + '" data-id="' + escapeHtml(item.id) + '">' +
        '<div class="overview-head">' +
          '<div><div class="overview-title">' + escapeHtml(item.workItemId) + '</div>' +
          '<div class="overview-subtitle">' + escapeHtml(shortText(item.summary || item.workflowId, 180)) + '</div></div>' +
          '<div class="overview-stats">' +
            statusBadge(item.status) +
            '<span class="chip">' + item.nodeCount + ' nodes</span>' +
            '<span class="chip">' + item.toolCount + ' tools</span>' +
            '<span class="chip">' + item.selectedSkillCount + ' skills</span>' +
            '<span class="chip">skill ' + escapeHtml(item.skillEvolutionSummary) + '</span>' +
            '<span class="chip">mem ' + escapeHtml(item.memoryEvolutionSummary) + '</span>' +
          '</div>' +
        '</div>' +
        '<div class="node-lane">' + item.nodes.map((node) => (
          '<button class="node-step ' + statusTone(node.status) + ' ' + (state.selectedId === node.id ? 'active' : '') + '" type="button" data-id="' + escapeHtml(node.id) + '">' +
            '<div class="eyebrow"><span>' + escapeHtml(node.nodeId || node.stage) + '</span>' + statusBadge(node.status) + '</div>' +
            '<div class="node-role">' + escapeHtml(node.role || node.subagentId) + '</div>' +
            '<div class="node-meta">' + node.toolCount + ' tools, ' + node.selectedSkillCount + ' skills, ' + node.memoryItemCount + ' memories</div>' +
            '<div class="chips"><span class="chip">skill ' + escapeHtml(node.skillUpdateStatus || 'none') + '</span><span class="chip">mem ' + escapeHtml(node.memoryUpdateStatus || 'none') + '</span></div>' +
          '</button>'
        )).join('') + '</div>' +
      '</div>';
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
      const dynamicChip = item.dynamicNode ? '<span class="chip">dynamic ' + escapeHtml(item.dynamicNode) + '</span>' : '';
      return '<button class="item-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + escapeHtml(item.id) + '">' +
        '<div class="eyebrow"><span>' + escapeHtml(item.runRef) + '</span>' + statusBadge(item.status) + '</div>' +
        '<div class="card-title">' + escapeHtml(item.role) + ' stage ' + escapeHtml(item.stage) + '</div>' +
        '<div class="card-text">' + escapeHtml(shortText(item.failure || item.instruction, 180)) + '</div>' +
        '<div class="chips"><span class="chip">' + item.llmCount + ' LLM</span><span class="chip">' + item.toolCount + ' tools</span><span class="chip">' + item.artifacts.length + ' artifacts</span>' + dynamicChip + '</div>' +
      '</button>';
    }

    function renderDynamic() {
      const items = filtered(DATA.dynamicSubagents || []);
      document.getElementById('view-count').textContent = String(items.length);
      document.getElementById('view').innerHTML = items.length ? (
        '<div class="stack">' + items.map((item) => (
          '<button class="item-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + escapeHtml(item.id) + '">' +
            '<div class="eyebrow"><span>' + escapeHtml(item.workflowId) + '</span>' + statusBadge(item.status) + '</div>' +
            '<div class="card-title">' + escapeHtml(item.role) + ' / ' + escapeHtml(item.nodeId || item.subagentId) + '</div>' +
            '<div class="card-text">' + escapeHtml(shortText(item.goal || item.systemPrompt, 210)) + '</div>' +
            '<div class="chips">' +
              '<span class="chip">' + escapeHtml(item.workItemId) + '</span>' +
              '<span class="chip">' + item.allowedTools.length + ' tools</span>' +
              '<span class="chip">' + item.requiredSkills.length + ' skills</span>' +
              (item.runRef ? '<span class="chip">' + escapeHtml(item.runRef) + '</span>' : '') +
            '</div>' +
            '<div class="chips">' + item.allowedTools.slice(0, 8).map((tool) => '<span class="chip">' + escapeHtml(shortText(tool, 48)) + '</span>').join('') + '</div>' +
          '</button>'
        )).join('') + '</div>'
      ) : '<div class="empty">No matching dynamic subagents.</div>';
      bindItemClicks();
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
      const item = byId.get(state.selectedId) || (DATA.evolutionOverview || [])[0] || DATA.subagentRuns[0] || (DATA.dynamicSubagents || [])[0] || DATA.metaRuns[0] || DATA.flow[0];
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
      return escapeHtml(value === null || value === undefined ? '' : value);
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
    document.getElementById('copy-detail').addEventListener('click', () => {
      const item = byId.get(state.selectedId);
      if (!item || !navigator.clipboard) return;
      navigator.clipboard.writeText(JSON.stringify(item.raw || item, null, 2));
    });

    state.selectedId = ((DATA.evolutionOverview || [])[0] || DATA.subagentRuns[0] || (DATA.dynamicSubagents || [])[0] || DATA.metaRuns[0] || DATA.flow[0] || {}).id || '';
    renderAll();
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__TITLE__", _h(title))
        .replace("__DATA__", _script_json(data))
        .replace("__LEGACY_SCRIPT__", _legacy_dashboard_script())
        .replace("__STATIC_METRICS__", _static_metrics_html(data))
        .replace("__STATIC_RUN_COUNT__", _h(len(data.get("subagentRuns", []))))
        .replace("__STATIC_RUN_LIST__", _static_run_list_html(data))
        .replace("__STATIC_VIEW_TITLE__", "Evolution Overview")
        .replace("__STATIC_VIEW_COUNT__", _h(len(data.get("evolutionOverview", []))))
        .replace("__STATIC_TABS__", _static_tabs_html())
        .replace("__STATIC_VIEW__", _static_overview_html(data))
        .replace("__STATIC_DETAILS__", _static_details_html(data))
    )


def _static_metrics_html(data: dict[str, Any]) -> str:
    labels = {
        "meta_agent_runs": "MetaAgent runs",
        "subagent_runs": "Subagent runs",
        "llm_calls": "LLM calls",
        "tool_calls": "Tool calls",
        "events": "Events",
        "evolution_runs": "Evolution runs",
        "dynamic_workflows": "Dynamic workflows",
        "dynamic_subagent_specs": "Dynamic subagent specs",
        "dynamic_subagent_records": "Dynamic subagent records",
    }
    return "".join(
        f'<div class="metric"><strong>{_h(value)}</strong><span>{_h(labels.get(key, key))}</span></div>'
        for key, value in data.get("counts", {}).items()
    )


def _legacy_dashboard_script() -> str:
    return r"""
(function () {
  function byId(id) { return document.getElementById(id); }
  function arr(value) { return value && value.length ? value : []; }
  function setStatus(text, className) {
    var node = byId('script-status');
    if (!node) return;
    node.className = 'script-status ' + (className || '');
    node.textContent = text;
  }
  function esc(value) {
    var text = value === null || value === undefined ? '' : String(value);
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  function shortText(value, limit) {
    var text = value === null || value === undefined ? '' : String(value);
    return text.length <= limit ? text : text.slice(0, Math.max(0, limit - 3)) + '...';
  }
  function hasText(text, needle) { return text.indexOf(needle) !== -1; }
  function statusTone(value) {
    var text = String(value === null || value === undefined ? '' : value).toLowerCase();
    if (hasText(text, 'fail') || hasText(text, 'error') || hasText(text, 'abort') || hasText(text, 'reject')) return 'bad';
    if (hasText(text, 'budget') || hasText(text, 'skip') || hasText(text, 'warn') || hasText(text, 'degraded')) return 'warn';
    if (hasText(text, 'complete') || text === 'ok' || hasText(text, 'promoted')) return 'ok';
    return '';
  }
  function statusBadge(value) {
    if (!value) return '';
    return '<span class="status ' + esc(statusTone(value)) + '">' + esc(value) + '</span>';
  }
  function detailValue(value) {
    var i;
    if (Object.prototype.toString.call(value) === '[object Array]') {
      if (!value.length) return '<span class="muted">empty</span>';
      var chips = '';
      for (i = 0; i < value.length && i < 16; i += 1) {
        chips += '<span class="chip">' + esc(shortText(typeof value[i] === 'string' ? value[i] : JSON.stringify(value[i]), 90)) + '</span>';
      }
      return '<div class="chips">' + chips + '</div>';
    }
    if (value && typeof value === 'object') {
      return '<code>' + esc(shortText(JSON.stringify(value), 180)) + '</code>';
    }
    return esc(value);
  }
  function getDataAttr(node, attr) {
    while (node && node !== document) {
      if (node.getAttribute) {
        var value = node.getAttribute(attr);
        if (value) return value;
      }
      node = node.parentNode;
    }
    return '';
  }
  function safeJson(value) {
    try { return JSON.stringify(value); } catch (err) { return String(value || ''); }
  }

  try {
    var dataNode = byId('trajectory-data');
    var DATA = JSON.parse(dataNode.textContent || dataNode.innerText || '{}');
    var VIEWS = [
      ['overview', 'Evolution Overview'],
      ['flow', 'Dispatch Flow'],
      ['runs', 'Subagent Runs'],
      ['dynamic', 'Dynamic Subagents'],
      ['skills', 'Skill And Memory'],
      ['tools', 'Tool Calls'],
      ['llm', 'LLM Calls'],
      ['meta', 'MetaAgent Routing'],
      ['events', 'Events'],
      ['evolution', 'Evolution Runs']
    ];
    var state = { view: 'overview', query: '', role: 'all', status: 'all', selectedId: '' };
    var allItems = []
      .concat(arr(DATA.flow))
      .concat(arr(DATA.evolutionOverview))
      .concat(arr(DATA.evolutionOverviewNodes))
      .concat(arr(DATA.subagentRuns))
      .concat(arr(DATA.dynamicWorkflows))
      .concat(arr(DATA.dynamicSubagents))
      .concat(arr(DATA.metaRuns))
      .concat(arr(DATA.toolCalls))
      .concat(arr(DATA.llmCalls))
      .concat(arr(DATA.events))
      .concat(arr(DATA.evolutionRuns));
    var byItemId = {};
    var i;
    for (i = 0; i < allItems.length; i += 1) {
      if (allItems[i] && allItems[i].id) byItemId[allItems[i].id] = allItems[i];
    }

    function itemMatches(item) {
      var query = String(state.query || '').toLowerCase();
      var text = String(item.search || safeJson(item)).toLowerCase();
      var roleOk = state.role === 'all' || item.role === state.role || item.target === state.role;
      var roles = arr(item.roles);
      for (var r = 0; !roleOk && r < roles.length; r += 1) roleOk = roles[r] === state.role;
      var statusOk = state.status === 'all' || item.status === state.status;
      return (!query || text.indexOf(query) !== -1) && roleOk && statusOk;
    }
    function filtered(items) {
      var result = [];
      items = arr(items);
      for (var j = 0; j < items.length; j += 1) {
        if (itemMatches(items[j])) result.push(items[j]);
      }
      return result;
    }
    function setSelected(id) {
      if (id) state.selectedId = id;
      renderAll();
    }
    function selectView(view) {
      state.view = view || 'flow';
      renderAll();
    }
    function renderMetrics() {
      var labels = {
        meta_agent_runs: 'MetaAgent runs',
        subagent_runs: 'Subagent runs',
        llm_calls: 'LLM calls',
        tool_calls: 'Tool calls',
        events: 'Events',
        evolution_runs: 'Evolution runs',
        dynamic_workflows: 'Dynamic workflows',
        dynamic_subagent_specs: 'Dynamic subagent specs',
        dynamic_subagent_records: 'Dynamic subagent records'
      };
      var html = '';
      var counts = DATA.counts || {};
      for (var key in counts) {
        if (Object.prototype.hasOwnProperty.call(counts, key)) {
          html += '<div class="metric"><strong>' + esc(counts[key]) + '</strong><span>' + esc(labels[key] || key) + '</span></div>';
        }
      }
      byId('metrics').innerHTML = html;
    }
    function renderFilters() {
      var role = byId('role-filter');
      var status = byId('status-filter');
      var html = '<option value="all">All roles</option>';
      var roles = arr(DATA.roles);
      for (var j = 0; j < roles.length; j += 1) html += '<option value="' + esc(roles[j]) + '">' + esc(roles[j]) + '</option>';
      role.innerHTML = html;
      html = '<option value="all">All statuses</option>';
      var statuses = arr(DATA.statuses);
      for (j = 0; j < statuses.length; j += 1) html += '<option value="' + esc(statuses[j]) + '">' + esc(statuses[j]) + '</option>';
      status.innerHTML = html;
      role.value = state.role;
      status.value = state.status;
    }
    function renderTabs() {
      var html = '';
      for (var j = 0; j < VIEWS.length; j += 1) {
        html += '<button class="tab" type="button" data-view="' + esc(VIEWS[j][0]) + '" aria-selected="' + (state.view === VIEWS[j][0]) + '">' + esc(VIEWS[j][1]) + '</button>';
      }
      byId('tabs').innerHTML = html;
    }
    function renderRunList() {
      var items = filtered(DATA.subagentRuns);
      byId('run-count').textContent = String(items.length);
      if (!items.length) {
        byId('run-list').innerHTML = '<div class="empty">No matching runs.</div>';
        return;
      }
      var html = '';
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        html += '<button class="run-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + esc(item.id) + '">' +
          '<div class="eyebrow"><span>Stage ' + esc(item.stage) + '</span>' + statusBadge(item.status) + '</div>' +
          '<div class="card-title">' + esc(item.role) + '</div>' +
          '<div class="card-text">' + esc(shortText(item.failure || item.instruction, 120)) + '</div>' +
          '<div class="chips"><span class="chip">' + esc(item.llmCount) + ' LLM</span><span class="chip">' + esc(item.toolCount) + ' tools</span><span class="chip">' + esc(item.skillCount) + ' skills</span>' +
          (item.dynamicNode ? '<span class="chip">dynamic ' + esc(item.dynamicNode) + '</span>' : '') + '</div></button>';
      }
      byId('run-list').innerHTML = html;
    }
    function runCard(item) {
      return '<button class="item-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + esc(item.id) + '">' +
        '<div class="eyebrow"><span>' + esc(item.runRef) + '</span>' + statusBadge(item.status) + '</div>' +
        '<div class="card-title">' + esc(item.role) + ' stage ' + esc(item.stage) + '</div>' +
        '<div class="card-text">' + esc(shortText(item.failure || item.instruction, 180)) + '</div>' +
        '<div class="chips"><span class="chip">' + esc(item.llmCount) + ' LLM</span><span class="chip">' + esc(item.toolCount) + ' tools</span><span class="chip">' + esc(arr(item.artifacts).length) + ' artifacts</span>' +
        (item.dynamicNode ? '<span class="chip">dynamic ' + esc(item.dynamicNode) + '</span>' : '') + '</div></button>';
    }
    function renderFlow() {
      var items = filtered(DATA.flow);
      byId('view-count').textContent = String(items.length);
      if (!items.length) { byId('view').innerHTML = '<div class="empty">No matching flow nodes.</div>'; return; }
      var html = '<div class="flow-lane">';
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        html += '<button class="flow-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + esc(item.id) + '">' +
          '<div class="eyebrow"><span>' + esc(item.kind) + '</span>' + statusBadge(item.status || item.action) + '</div>' +
          '<div class="card-title">' + esc(item.title) + '</div>' +
          '<div class="card-text">' + esc(shortText(item.summary, 170)) + '</div></button>';
      }
      byId('view').innerHTML = html + '</div>';
    }
    function overviewCard(item) {
      var html = '<div class="overview-card ' + (state.selectedId === item.id ? 'active' : '') + '" data-id="' + esc(item.id) + '">' +
        '<div class="overview-head"><div><div class="overview-title">' + esc(item.workItemId) + '</div>' +
        '<div class="overview-subtitle">' + esc(shortText(item.summary || item.workflowId, 180)) + '</div></div>' +
        '<div class="overview-stats">' + statusBadge(item.status) +
        '<span class="chip">' + esc(item.nodeCount) + ' nodes</span>' +
        '<span class="chip">' + esc(item.toolCount) + ' tools</span>' +
        '<span class="chip">' + esc(item.selectedSkillCount) + ' skills</span>' +
        '<span class="chip">skill ' + esc(item.skillEvolutionSummary) + '</span>' +
        '<span class="chip">mem ' + esc(item.memoryEvolutionSummary) + '</span></div></div><div class="node-lane">';
      var nodes = arr(item.nodes);
      for (var j = 0; j < nodes.length; j += 1) {
        var node = nodes[j];
        html += '<button class="node-step ' + statusTone(node.status) + ' ' + (state.selectedId === node.id ? 'active' : '') + '" type="button" data-id="' + esc(node.id) + '">' +
          '<div class="eyebrow"><span>' + esc(node.nodeId || node.stage) + '</span>' + statusBadge(node.status) + '</div>' +
          '<div class="node-role">' + esc(node.role || node.subagentId) + '</div>' +
          '<div class="node-meta">' + esc(node.toolCount) + ' tools, ' + esc(node.selectedSkillCount) + ' skills, ' + esc(node.memoryItemCount) + ' memories</div>' +
          '<div class="chips"><span class="chip">skill ' + esc(node.skillUpdateStatus || 'none') + '</span><span class="chip">mem ' + esc(node.memoryUpdateStatus || 'none') + '</span></div></button>';
      }
      return html + '</div></div>';
    }
    function renderOverview() {
      var items = filtered(DATA.evolutionOverview);
      byId('view-count').textContent = String(items.length);
      if (!items.length) { byId('view').innerHTML = '<div class="empty">No matching evolution overview records.</div>'; return; }
      var html = '<div class="overview-list">';
      for (var j = 0; j < items.length; j += 1) html += overviewCard(items[j]);
      byId('view').innerHTML = html + '</div>';
    }
    function renderRuns() {
      var items = filtered(DATA.subagentRuns);
      byId('view-count').textContent = String(items.length);
      if (!items.length) { byId('view').innerHTML = '<div class="empty">No matching subagent runs.</div>'; return; }
      var html = '<div class="grid-2">';
      for (var j = 0; j < items.length; j += 1) html += runCard(items[j]);
      byId('view').innerHTML = html + '</div>';
    }
    function renderDynamic() {
      var items = filtered(DATA.dynamicSubagents);
      byId('view-count').textContent = String(items.length);
      if (!items.length) { byId('view').innerHTML = '<div class="empty">No matching dynamic subagents.</div>'; return; }
      var html = '<div class="stack">';
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        html += '<button class="item-card ' + (state.selectedId === item.id ? 'active' : '') + '" type="button" data-id="' + esc(item.id) + '">' +
          '<div class="eyebrow"><span>' + esc(item.workflowId) + '</span>' + statusBadge(item.status) + '</div>' +
          '<div class="card-title">' + esc(item.role) + ' / ' + esc(item.nodeId || item.subagentId) + '</div>' +
          '<div class="card-text">' + esc(shortText(item.goal || item.systemPrompt, 210)) + '</div>' +
          '<div class="chips"><span class="chip">' + esc(item.workItemId) + '</span><span class="chip">' + esc(arr(item.allowedTools).length) + ' tools</span><span class="chip">' + esc(arr(item.requiredSkills).length) + ' skills</span>' +
          (item.runRef ? '<span class="chip">' + esc(item.runRef) + '</span>' : '') + '</div><div class="chips">';
        var tools = arr(item.allowedTools);
        for (var t = 0; t < tools.length && t < 8; t += 1) html += '<span class="chip">' + esc(shortText(tools[t], 48)) + '</span>';
        html += '</div></button>';
      }
      byId('view').innerHTML = html + '</div>';
    }
    function renderSkills() {
      var runs = filtered(DATA.subagentRuns);
      var total = 0;
      var html = '<div class="stack">';
      for (var j = 0; j < runs.length; j += 1) {
        var run = runs[j];
        total += arr(run.skills).length + arr(run.workflowNodes).length;
        html += '<button class="item-card ' + (state.selectedId === run.id ? 'active' : '') + '" type="button" data-id="' + esc(run.id) + '">' +
          '<div class="eyebrow"><span>' + esc(run.role) + '</span>' + statusBadge(run.status) + '</div>' +
          '<div class="card-title">' + esc(arr(run.skills).length) + ' skills, ' + esc(arr(run.memoryItems).length) + ' memories, ' + esc(arr(run.workflowNodes).length) + ' workflow nodes</div><div class="chips">';
        var skills = arr(run.skills);
        for (var s = 0; s < skills.length && s < 8; s += 1) html += '<span class="chip">' + esc(shortText(skills[s].label, 70)) + '</span>';
        html += '</div><div class="card-text">' + esc(shortText(run.updateSummary, 180)) + '</div></button>';
      }
      byId('view-count').textContent = String(total);
      byId('view').innerHTML = runs.length ? html + '</div>' : '<div class="empty">No matching skill or memory records.</div>';
    }
    function table(headers, rows, ids) {
      var html = '<table><thead><tr>';
      var j;
      for (j = 0; j < headers.length; j += 1) html += '<th>' + esc(headers[j]) + '</th>';
      html += '</tr></thead><tbody>';
      for (j = 0; j < rows.length; j += 1) {
        html += '<tr class="selectable" data-id="' + esc(ids[j]) + '">';
        for (var c = 0; c < rows[j].length; c += 1) html += '<td data-label="' + esc(headers[c]) + '">' + rows[j][c] + '</td>';
        html += '</tr>';
      }
      return html + '</tbody></table>';
    }
    function renderTools() {
      var items = filtered(DATA.toolCalls);
      var rows = [];
      var ids = [];
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        rows.push(['<code>' + esc(item.tool) + '</code>', esc(item.role), esc(item.stage), statusBadge(item.status), esc(shortText(item.result, 160))]);
        ids.push(item.id);
      }
      byId('view-count').textContent = String(items.length);
      byId('view').innerHTML = items.length ? table(['Tool', 'Role', 'Stage', 'Status', 'Result'], rows, ids) : '<div class="empty">No matching tool calls.</div>';
    }
    function renderLLM() {
      var items = filtered(DATA.llmCalls);
      var rows = [];
      var ids = [];
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        rows.push(['<code>' + esc(item.callRef) + '</code>', esc(item.role), esc(item.stage), esc(shortText(item.model, 36)), esc(item.action), esc(item.usage)]);
        ids.push(item.id);
      }
      byId('view-count').textContent = String(items.length);
      byId('view').innerHTML = items.length ? table(['Call', 'Role', 'Stage', 'Model', 'Action', 'Usage'], rows, ids) : '<div class="empty">No matching LLM calls.</div>';
    }
    function renderMeta() {
      var items = filtered(DATA.metaRuns);
      var rows = [];
      var ids = [];
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        rows.push([statusBadge(item.action), esc(item.target), '<code>' + esc(item.runRef) + '</code>', esc(shortText(item.instruction, 180))]);
        ids.push(item.id);
      }
      byId('view-count').textContent = String(items.length);
      byId('view').innerHTML = items.length ? table(['Action', 'Target', 'Run', 'Instruction'], rows, ids) : '<div class="empty">No matching MetaAgent decisions.</div>';
    }
    function renderEvents() {
      var items = filtered(DATA.events);
      var rows = [];
      var ids = [];
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        rows.push([esc(item.time), '<code>' + esc(item.eventType) + '</code>', esc(item.subject), '<code>' + esc(item.runRef) + '</code>']);
        ids.push(item.id);
      }
      byId('view-count').textContent = String(items.length);
      byId('view').innerHTML = items.length ? table(['Time', 'Type', 'Subject', 'Run'], rows, ids) : '<div class="empty">No matching events.</div>';
    }
    function renderEvolution() {
      var items = filtered(DATA.evolutionRuns);
      var rows = [];
      var ids = [];
      for (var j = 0; j < items.length; j += 1) {
        var item = items[j];
        rows.push(['<code>' + esc(item.runRef) + '</code>', esc(item.mode), '<code>' + esc(item.backendId) + '</code>', statusBadge(item.status), esc(shortText(arr(item.sources).join(', '), 160))]);
        ids.push(item.id);
      }
      byId('view-count').textContent = String(items.length);
      byId('view').innerHTML = items.length ? table(['Run', 'Mode', 'Backend', 'Status', 'Sources'], rows, ids) : '<div class="empty">No evolution records.</div>';
    }
    function renderDetails() {
      var item = byItemId[state.selectedId] || arr(DATA.evolutionOverview)[0] || arr(DATA.subagentRuns)[0] || arr(DATA.dynamicSubagents)[0] || arr(DATA.metaRuns)[0] || arr(DATA.flow)[0];
      if (!item) { byId('details').innerHTML = '<div class="empty">Select an item to inspect.</div>'; return; }
      state.selectedId = item.id;
      var summary = item.detail || {};
      var rows = '';
      for (var key in summary) {
        if (Object.prototype.hasOwnProperty.call(summary, key)) {
          rows += '<div class="detail-key">' + esc(key) + '</div><div class="detail-value">' + detailValue(summary[key]) + '</div>';
        }
      }
      byId('details').innerHTML = '<div class="detail-grid">' + rows + '</div><pre>' + esc(JSON.stringify(item.raw || item, null, 2)) + '</pre>';
    }
    function renderView() {
      var title = 'Evolution Overview';
      for (var j = 0; j < VIEWS.length; j += 1) if (VIEWS[j][0] === state.view) title = VIEWS[j][1];
      byId('view-title').textContent = title;
      if (state.view === 'overview') renderOverview();
      else if (state.view === 'flow') renderFlow();
      else if (state.view === 'runs') renderRuns();
      else if (state.view === 'dynamic') renderDynamic();
      else if (state.view === 'skills') renderSkills();
      else if (state.view === 'tools') renderTools();
      else if (state.view === 'llm') renderLLM();
      else if (state.view === 'meta') renderMeta();
      else if (state.view === 'events') renderEvents();
      else if (state.view === 'evolution') renderEvolution();
    }
    function renderAll() {
      renderMetrics();
      renderFilters();
      renderTabs();
      renderRunList();
      renderView();
      renderDetails();
      byId('source-dir').textContent = DATA.sourceDir || '';
      byId('generated-at').textContent = 'Generated ' + (DATA.generatedAt || '');
      setStatus('Interactive ready', 'ready');
    }
    function onClick(event) {
      event = event || window.event;
      var target = event.target || event.srcElement;
      var view = getDataAttr(target, 'data-view');
      if (view) { selectView(view); return; }
      var id = getDataAttr(target, 'data-id');
      if (id) setSelected(id);
    }
    byId('tabs').onclick = onClick;
    byId('run-list').onclick = onClick;
    byId('view').onclick = onClick;
    byId('search').oninput = function () { state.query = byId('search').value; renderAll(); };
    byId('role-filter').onchange = function () { state.role = byId('role-filter').value; renderAll(); };
    byId('status-filter').onchange = function () { state.status = byId('status-filter').value; renderAll(); };
    byId('copy-detail').onclick = function () {
      var item = byItemId[state.selectedId];
      if (item && navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(JSON.stringify(item.raw || item, null, 2));
      }
    };
    state.selectedId = (arr(DATA.evolutionOverview)[0] || arr(DATA.subagentRuns)[0] || arr(DATA.dynamicSubagents)[0] || arr(DATA.metaRuns)[0] || arr(DATA.flow)[0] || {}).id || '';
    renderAll();
    if (typeof window !== 'undefined') window.__trajectoryDashboardReady = true;
  } catch (err) {
    setStatus('Interactive error: ' + (err && err.message ? err.message : err), 'error');
    if (typeof console !== 'undefined' && console.error) console.error(err);
  }
}());
"""


def _static_tabs_html() -> str:
    views = [
        ("overview", "Evolution Overview"),
        ("flow", "Dispatch Flow"),
        ("runs", "Subagent Runs"),
        ("dynamic", "Dynamic Subagents"),
        ("skills", "Skill And Memory"),
        ("tools", "Tool Calls"),
        ("llm", "LLM Calls"),
        ("meta", "MetaAgent Routing"),
        ("events", "Events"),
        ("evolution", "Evolution Runs"),
    ]
    return "".join(
        '<button class="tab" type="button" '
        f'data-view="{_ha(view_id)}" aria-selected="{str(view_id == "flow").lower()}">'
        f"{_h(label)}</button>"
        for view_id, label in views
    )


def _static_run_list_html(data: dict[str, Any]) -> str:
    runs = data.get("subagentRuns", [])
    if not runs:
        return '<div class="empty">No matching runs.</div>'
    first_id = str(runs[0].get("id", ""))
    return "".join(_static_run_card_html(run, active_id=first_id) for run in runs)


def _static_run_card_html(run: dict[str, Any], *, active_id: str) -> str:
    run_id = str(run.get("id", ""))
    active = " active" if run_id == active_id else ""
    dynamic_chip = ""
    if run.get("dynamicNode"):
        dynamic_chip = f'<span class="chip">dynamic {_h(run.get("dynamicNode"))}</span>'
    return (
        f'<button class="run-card{active}" type="button" data-id="{_ha(run_id)}">'
        f'<div class="eyebrow"><span>Stage {_h(run.get("stage", ""))}</span>'
        f'{_static_status_badge(run.get("status"))}</div>'
        f'<div class="card-title">{_h(run.get("role", ""))}</div>'
        f'<div class="card-text">{_h(_short_js_text(run.get("failure") or run.get("instruction"), 120))}</div>'
        '<div class="chips">'
        f'<span class="chip">{_h(run.get("llmCount", 0))} LLM</span>'
        f'<span class="chip">{_h(run.get("toolCount", 0))} tools</span>'
        f'<span class="chip">{_h(run.get("skillCount", 0))} skills</span>'
        f"{dynamic_chip}</div></button>"
    )


def _static_overview_html(data: dict[str, Any]) -> str:
    items = data.get("evolutionOverview", [])
    if not items:
        return '<div class="empty">No matching evolution overview records.</div>'
    first_id = str(items[0].get("id", ""))
    cards = "".join(_static_overview_card_html(item, active_id=first_id) for item in items)
    return f'<div class="overview-list">{cards}</div>'


def _static_overview_card_html(item: dict[str, Any], *, active_id: str) -> str:
    item_id = str(item.get("id", ""))
    active = " active" if item_id == active_id else ""
    nodes = item.get("nodes") if isinstance(item.get("nodes"), list) else []
    node_cards = "".join(_static_overview_node_html(node, active_id=active_id) for node in nodes)
    return (
        f'<div class="overview-card{active}" data-id="{_ha(item_id)}">'
        '<div class="overview-head">'
        f'<div><div class="overview-title">{_h(item.get("workItemId", ""))}</div>'
        f'<div class="overview-subtitle">{_h(_short_js_text(item.get("summary") or item.get("workflowId"), 180))}</div></div>'
        '<div class="overview-stats">'
        f'{_static_status_badge(item.get("status"))}'
        f'<span class="chip">{_h(item.get("nodeCount", 0))} nodes</span>'
        f'<span class="chip">{_h(item.get("toolCount", 0))} tools</span>'
        f'<span class="chip">{_h(item.get("selectedSkillCount", 0))} skills</span>'
        f'<span class="chip">skill {_h(item.get("skillEvolutionSummary", ""))}</span>'
        f'<span class="chip">mem {_h(item.get("memoryEvolutionSummary", ""))}</span>'
        '</div></div>'
        f'<div class="node-lane">{node_cards}</div></div>'
    )


def _static_overview_node_html(node: dict[str, Any], *, active_id: str) -> str:
    node_id = str(node.get("id", ""))
    active = " active" if node_id == active_id else ""
    tone = _status_tone(node.get("status"))
    return (
        f'<button class="node-step {tone}{active}" type="button" data-id="{_ha(node_id)}">'
        f'<div class="eyebrow"><span>{_h(node.get("nodeId") or node.get("stage", ""))}</span>{_static_status_badge(node.get("status"))}</div>'
        f'<div class="node-role">{_h(node.get("role") or node.get("subagentId", ""))}</div>'
        f'<div class="node-meta">{_h(node.get("toolCount", 0))} tools, {_h(node.get("selectedSkillCount", 0))} skills, {_h(node.get("memoryItemCount", 0))} memories</div>'
        '<div class="chips">'
        f'<span class="chip">skill {_h(node.get("skillUpdateStatus") or "none")}</span>'
        f'<span class="chip">mem {_h(node.get("memoryUpdateStatus") or "none")}</span>'
        '</div></button>'
    )


def _static_flow_html(data: dict[str, Any]) -> str:
    flow = data.get("flow", [])
    if not flow:
        return '<div class="empty">No matching flow nodes.</div>'
    first_id = str((data.get("subagentRuns") or data.get("metaRuns") or flow or [{}])[0].get("id", ""))
    cards = "".join(_static_flow_card_html(item, active_id=first_id) for item in flow)
    return f'<div class="flow-lane">{cards}</div>'


def _static_flow_card_html(item: dict[str, Any], *, active_id: str) -> str:
    item_id = str(item.get("id", ""))
    active = " active" if item_id == active_id else ""
    status = item.get("status") or item.get("action")
    return (
        f'<button class="flow-card{active}" type="button" data-id="{_ha(item_id)}">'
        f'<div class="eyebrow"><span>{_h(item.get("kind", ""))}</span>{_static_status_badge(status)}</div>'
        f'<div class="card-title">{_h(item.get("title", ""))}</div>'
        f'<div class="card-text">{_h(_short_js_text(item.get("summary", ""), 170))}</div>'
        "</button>"
    )


def _static_details_html(data: dict[str, Any]) -> str:
    item = (
        data.get("evolutionOverview")
        or data.get("subagentRuns")
        or data.get("dynamicSubagents")
        or data.get("metaRuns")
        or data.get("flow")
        or [None]
    )[0]
    if not isinstance(item, dict):
        return '<div class="empty">Select an item to inspect.</div>'
    detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
    rows = "".join(
        f'<div class="detail-key">{_h(key)}</div><div class="detail-value">{_static_detail_value(value)}</div>'
        for key, value in detail.items()
    )
    raw = json.dumps(item.get("raw") or item, indent=2, ensure_ascii=True, default=str)
    return f'<div class="detail-grid">{rows}</div><pre>{_h(raw)}</pre>'


def _static_detail_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return '<span class="muted">empty</span>'
        chips = "".join(
            f'<span class="chip">{_h(_short_js_text(item if isinstance(item, str) else json.dumps(item, default=str), 90))}</span>'
            for item in value[:16]
        )
        return f'<div class="chips">{chips}</div>'
    if isinstance(value, dict):
        return f"<code>{_h(_short_js_text(json.dumps(value, default=str), 180))}</code>"
    return _h(value)


def _static_status_badge(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f'<span class="status {_ha(_status_tone(value))}">{_h(value)}</span>'


def _status_tone(value: Any) -> str:
    text = str(value or "").lower()
    if any(item in text for item in ("fail", "error", "abort", "reject")):
        return "bad"
    if any(item in text for item in ("budget", "skip", "warn", "degraded")):
        return "warn"
    if "complete" in text or text == "ok" or "promoted" in text:
        return "ok"
    return ""


def _short_js_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


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
    dynamic_subagent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _asdict(run.metadata)
    status = str(metadata.get("status") or "recorded")
    failure = str(metadata.get("failure_reason") or "")
    skills = _skill_summaries(run)
    memory_items = _memory_summaries(run)
    workflow_nodes = _workflow_node_summaries(metadata)
    artifacts = _artifact_items(run.artifact_refs)
    dynamic_info = _dynamic_subagent_run_info(dynamic_subagent)
    skill_update = _skill_update_item(_asdict(metadata.get("skill_update_result")))
    memory_update = _memory_update_item(metadata)
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
        "skill_update": skill_update,
        "memory_update": memory_update,
        "dynamic_subagent": dynamic_info,
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
        "skillUpdate": skill_update,
        "memoryUpdate": memory_update,
        "skillUpdateStatus": skill_update.get("status", ""),
        "memoryUpdateStatus": memory_update.get("status", ""),
        "dynamicWorkflowId": dynamic_info.get("workflow_id", ""),
        "dynamicWorkItemId": dynamic_info.get("work_item_id", ""),
        "dynamicNode": dynamic_info.get("node_id", ""),
        "dynamicSubagentId": dynamic_info.get("subagent_id", ""),
        "dynamicAllowedTools": dynamic_info.get("allowed_tools", []),
        "dynamicRequiredSkills": dynamic_info.get("required_skills", []),
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
            "skill update": skill_update,
            "memory update": memory_update,
            "dynamic workflow": dynamic_info.get("workflow_id", ""),
            "dynamic node": dynamic_info.get("node_id", ""),
            "dynamic tools": dynamic_info.get("allowed_tools", []),
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


def _evolution_overview_items(
    *,
    dynamic_workflows: list[dict[str, Any]],
    subagent_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs_by_ref = {
        str(run.get("runRef") or ""): run
        for run in subagent_runs
        if run.get("runRef")
    }
    runs_by_dynamic_key = {
        (
            str(run.get("dynamicWorkflowId") or ""),
            str(run.get("dynamicWorkItemId") or ""),
            str(run.get("dynamicNode") or ""),
        ): run
        for run in subagent_runs
        if run.get("dynamicWorkflowId") or run.get("dynamicWorkItemId") or run.get("dynamicNode")
    }
    items = []
    for workflow in dynamic_workflows:
        nodes = []
        for index, spec in enumerate(workflow.get("subagentSpecs") or []):
            run = runs_by_ref.get(str(spec.get("runRef") or ""))
            if not run:
                run = runs_by_dynamic_key.get(
                    (
                        str(spec.get("workflowId") or ""),
                        str(spec.get("workItemId") or ""),
                        str(spec.get("nodeId") or ""),
                    )
                )
            nodes.append(
                _evolution_overview_node_item(
                    workflow=workflow,
                    spec=spec,
                    run=run,
                    index=index,
                )
            )

        status_counts = Counter(node.get("status") or "unknown" for node in nodes)
        unique_tools = sorted({tool for node in nodes for tool in node.get("allowedTools", [])})
        selected_skills = sorted({skill for node in nodes for skill in node.get("selectedSkills", [])})
        memory_items = sum(int(node.get("memoryItemCount") or 0) for node in nodes)
        skill_updates = Counter(node.get("skillUpdateStatus") or "none" for node in nodes)
        memory_updates = Counter(node.get("memoryUpdateStatus") or "none" for node in nodes)
        item_id = f"overview:{workflow.get('relativePath') or workflow.get('workflowId') or workflow.get('workItemId')}"
        raw = {
            "workflow": workflow,
            "nodes": nodes,
            "status_counts": dict(status_counts),
            "toolset": unique_tools,
            "selected_skills": selected_skills,
            "skill_updates": dict(skill_updates),
            "memory_updates": dict(memory_updates),
        }
        item = {
            "id": item_id,
            "kind": "evolution-overview",
            "workflowId": workflow.get("workflowId", ""),
            "workItemId": workflow.get("workItemId", ""),
            "relativePath": workflow.get("relativePath", ""),
            "status": workflow.get("status", ""),
            "roles": workflow.get("roles", []),
            "summary": workflow.get("summary", ""),
            "rationale": workflow.get("rationale", ""),
            "nodeCount": len(nodes),
            "statusCounts": dict(status_counts),
            "toolCount": len(unique_tools),
            "toolset": unique_tools,
            "selectedSkillCount": len(selected_skills),
            "selectedSkills": selected_skills,
            "memoryItemCount": memory_items,
            "skillEvolutionSummary": _counter_summary(skill_updates),
            "memoryEvolutionSummary": _counter_summary(memory_updates),
            "nodes": nodes,
            "detail": {
                "work item": workflow.get("workItemId", ""),
                "workflow": workflow.get("workflowId", ""),
                "status": workflow.get("status", ""),
                "node statuses": dict(status_counts),
                "toolset": unique_tools,
                "selected skills": selected_skills,
                "skill updates": dict(skill_updates),
                "memory updates": dict(memory_updates),
                "rationale": workflow.get("rationale", ""),
            },
            "raw": _compact_value(raw, 12000),
            "search": _search_text(raw),
        }
        items.append(item)
    return items


def _evolution_overview_node_item(
    *,
    workflow: dict[str, Any],
    spec: dict[str, Any],
    run: dict[str, Any] | None,
    index: int,
) -> dict[str, Any]:
    run = run or {}
    node_id = str(spec.get("nodeId") or run.get("dynamicNode") or "")
    role = str(spec.get("role") or run.get("role") or "")
    status = str(run.get("status") or spec.get("status") or "planned")
    selected_skills = [str(item.get("label") or item.get("skill_id") or item.get("name") or "") for item in run.get("skills", [])]
    selected_skills = [item for item in selected_skills if item]
    memory_items = run.get("memoryItems") or []
    skill_update = run.get("skillUpdate") if isinstance(run.get("skillUpdate"), dict) else {}
    memory_update = run.get("memoryUpdate") if isinstance(run.get("memoryUpdate"), dict) else {}
    allowed_tools = list(spec.get("allowedTools") or run.get("dynamicAllowedTools") or [])
    required_skills = list(spec.get("requiredSkills") or run.get("dynamicRequiredSkills") or [])
    node_item = {
        "id": f"overview-node:{workflow.get('relativePath') or workflow.get('workflowId')}:{node_id or index}:{role}",
        "kind": "evolution-overview-node",
        "workflowId": workflow.get("workflowId", ""),
        "workItemId": workflow.get("workItemId", ""),
        "nodeId": node_id,
        "subagentId": spec.get("subagentId", ""),
        "role": role,
        "status": status,
        "stage": run.get("stage", index),
        "runRef": run.get("runRef") or spec.get("runRef") or "",
        "goal": spec.get("goal", ""),
        "systemPrompt": spec.get("systemPrompt", ""),
        "instruction": run.get("instruction", ""),
        "allowedTools": allowed_tools,
        "toolCount": len(allowed_tools),
        "requiredSkills": required_skills,
        "selectedSkills": selected_skills,
        "selectedSkillCount": len(selected_skills),
        "memoryItems": memory_items,
        "memoryItemCount": len(memory_items),
        "skillUpdate": skill_update,
        "skillUpdateStatus": skill_update.get("status", ""),
        "skillAppliedCount": skill_update.get("applied_count", 0),
        "skillStagedCount": skill_update.get("staged_count", 0),
        "skillRejectedCount": skill_update.get("rejected_count", 0),
        "skillChangedLibrary": skill_update.get("changed_library", False),
        "memoryUpdate": memory_update,
        "memoryUpdateStatus": memory_update.get("status", ""),
        "memoryDisabled": memory_update.get("disabled", False),
        "llmCount": run.get("llmCount", 0),
        "runtimeToolCount": run.get("toolCount", 0),
        "artifactCount": len(run.get("artifacts", []) or spec.get("artifacts", []) or []),
        "failure": run.get("failure") or spec.get("failure") or "",
    }
    node_item["detail"] = {
        "work item": node_item["workItemId"],
        "workflow": node_item["workflowId"],
        "node": node_item["nodeId"],
        "role": node_item["role"],
        "status": node_item["status"],
        "run": node_item["runRef"],
        "dynamic toolset": allowed_tools,
        "required skills": required_skills,
        "selected skills": selected_skills,
        "memory items": [item.get("content", "") for item in memory_items if isinstance(item, dict)],
        "skill update": skill_update,
        "memory update": memory_update,
        "failure": node_item["failure"],
    }
    node_item["raw"] = _compact_value({"spec": spec, "run": run, "overview": node_item}, 10000)
    node_item["search"] = _search_text(node_item["raw"])
    return node_item


def _counter_summary(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    parts = [f"{key}={counter[key]}" for key in sorted(counter)]
    return ", ".join(parts)


def _load_dynamic_workflows(lab_root: Path | None) -> list[dict[str, Any]]:
    if lab_root is None:
        return []
    dynamic_root = lab_root / "dynamic_workflows"
    if not dynamic_root.exists():
        return []

    workflow_dirs: set[Path] = set()
    for filename in (
        "dynamic_workflow_trace.json",
        "dynamic_workflow_spec.json",
        "dynamic_subagents.json",
        "dynamic_subagent_records.jsonl",
        "planner_validation_report.json",
    ):
        workflow_dirs.update(path.parent for path in dynamic_root.glob(f"**/{filename}"))

    workflows = []
    for workflow_dir in sorted(workflow_dirs, key=lambda path: str(path)):
        trace = _asdict(_read_json_file(workflow_dir / "dynamic_workflow_trace.json"))
        spec = _asdict(_read_json_file(workflow_dir / "dynamic_workflow_spec.json"))
        validation = _asdict(_read_json_file(workflow_dir / "planner_validation_report.json"))
        subagents = _asdict(_read_json_file(workflow_dir / "dynamic_subagents.json"))
        if not isinstance(subagents, list):
            subagents = spec.get("dynamic_subagents") if isinstance(spec, dict) else []
        if not isinstance(subagents, list):
            subagents = []
        records = _read_jsonl_file(workflow_dir / "dynamic_subagent_records.jsonl")
        if not records and isinstance(trace, dict):
            node_results = trace.get("node_results")
            if isinstance(node_results, list):
                records = _list_dicts(node_results)
        validation_metadata = validation.get("metadata") if isinstance(validation, dict) else {}
        if not isinstance(validation_metadata, dict):
            validation_metadata = {}
        workflows.append(
            {
                "relative_path": _relative_path(workflow_dir, dynamic_root),
                "path": str(workflow_dir),
                "workflow_id": str(
                    _dict_value(trace, "workflow_id")
                    or _dict_value(spec, "workflow_id")
                    or validation_metadata.get("workflow_id")
                    or workflow_dir.name
                ),
                "work_item_id": str(
                    _dict_value(trace, "work_item_id")
                    or _dict_value(spec, "work_item_id")
                    or validation_metadata.get("work_item_id")
                    or workflow_dir.name
                ),
                "status": str(_dict_value(trace, "status") or "recorded"),
                "execution_mode": str(_dict_value(trace, "execution_mode") or ""),
                "fallback_reason": str(_dict_value(trace, "fallback_reason") or ""),
                "planner_backend_id": str(
                    _dict_value(trace, "planner_backend_id")
                    or _dict_value(validation, "planner_backend_id")
                    or ""
                ),
                "default_worker_backend_id": str(
                    _dict_value(trace, "default_worker_backend_id")
                    or _dict_value(validation, "default_worker_backend_id")
                    or ""
                ),
                "spec": spec if isinstance(spec, dict) else {},
                "trace": trace if isinstance(trace, dict) else {},
                "validation": validation if isinstance(validation, dict) else {},
                "subagent_specs": _list_dicts(subagents),
                "records": records,
            }
        )
    return workflows


def _dynamic_workflow_dashboard_item(workflow: dict[str, Any]) -> dict[str, Any]:
    spec = _asdict(workflow.get("spec"))
    if not isinstance(spec, dict):
        spec = {}
    trace = _asdict(workflow.get("trace"))
    if not isinstance(trace, dict):
        trace = {}
    validation = _asdict(workflow.get("validation"))
    if not isinstance(validation, dict):
        validation = {}

    workflow_nodes = _list_dicts(spec.get("workflow_nodes"))
    workflow_edges = _list_dicts(spec.get("workflow_edges"))
    node_by_id = {str(node.get("node_id") or ""): node for node in workflow_nodes}
    node_by_subagent_id = {
        str(node.get("subagent_id") or ""): node
        for node in workflow_nodes
        if node.get("subagent_id")
    }
    records = _list_dicts(workflow.get("records"))
    records_by_node = {
        str(record.get("node_id") or ""): record
        for record in records
        if record.get("node_id")
    }

    workflow_id = str(workflow.get("workflow_id") or "")
    work_item_id = str(workflow.get("work_item_id") or "")
    status = str(workflow.get("status") or "recorded")
    relative_path = str(workflow.get("relative_path") or workflow_id or work_item_id)
    workflow_summary = _short_text(
        str(spec.get("task_summary") or spec.get("article_context_summary") or ""),
        700,
    )
    rationale = _short_text(str(spec.get("planner_rationale_summary") or ""), 700)
    validation_errors = _string_list(validation.get("errors"))
    validation_warnings = _string_list(validation.get("warnings"))
    record_items = [_dynamic_record_dashboard_item(record) for record in records]
    subagent_specs = []
    for spec_item in _list_dicts(workflow.get("subagent_specs")):
        subagent_id = str(spec_item.get("subagent_id") or "")
        node = node_by_subagent_id.get(subagent_id) or {}
        node_id = str(node.get("node_id") or "")
        record = records_by_node.get(node_id) or {}
        if not record and spec_item.get("role_name"):
            role_name = str(spec_item.get("role_name"))
            record = next(
                (item for item in records if str(item.get("role") or "") == role_name),
                {},
            )
            if record and not node:
                node = node_by_id.get(str(record.get("node_id") or ""), {})
        subagent_specs.append(
            _dynamic_subagent_spec_dashboard_item(
                spec_item,
                workflow=workflow,
                workflow_summary=workflow_summary,
                node=node,
                record=record,
            )
        )

    roles = sorted({item["role"] for item in subagent_specs if item.get("role")})
    tools = sorted({tool for item in subagent_specs for tool in item.get("allowedTools", [])})
    raw = {
        "workflow": {
            "relative_path": relative_path,
            "workflow_id": workflow_id,
            "work_item_id": work_item_id,
            "status": status,
            "execution_mode": workflow.get("execution_mode", ""),
            "fallback_reason": workflow.get("fallback_reason", ""),
            "planner_backend_id": workflow.get("planner_backend_id", ""),
            "default_worker_backend_id": workflow.get("default_worker_backend_id", ""),
        },
        "summary": workflow_summary,
        "rationale": rationale,
        "workflow_nodes": workflow_nodes,
        "workflow_edges": workflow_edges,
        "validation": validation,
        "subagent_specs": workflow.get("subagent_specs", []),
        "records": records,
    }
    return {
        "id": f"dynamic-workflow:{relative_path}",
        "kind": "dynamic-workflow",
        "workflowId": workflow_id,
        "workItemId": work_item_id,
        "relativePath": relative_path,
        "status": status,
        "executionMode": workflow.get("execution_mode", ""),
        "fallbackReason": workflow.get("fallback_reason", ""),
        "plannerBackendId": workflow.get("planner_backend_id", ""),
        "defaultWorkerBackendId": workflow.get("default_worker_backend_id", ""),
        "roles": roles,
        "tools": tools,
        "summary": workflow_summary,
        "rationale": rationale,
        "validationErrors": validation_errors,
        "validationWarnings": validation_warnings,
        "workflowNodes": workflow_nodes,
        "workflowEdges": workflow_edges,
        "records": record_items,
        "subagentSpecs": subagent_specs,
        "detail": {
            "workflow": workflow_id,
            "work item": work_item_id,
            "status": status,
            "execution mode": workflow.get("execution_mode", ""),
            "planner": workflow.get("planner_backend_id", ""),
            "worker backend": workflow.get("default_worker_backend_id", ""),
            "roles": roles,
            "tools": tools[:18],
            "validation errors": validation_errors,
            "validation warnings": validation_warnings,
        },
        "raw": _compact_value(raw, 10000),
        "search": _search_text(raw),
    }


def _dynamic_record_dashboard_item(record: dict[str, Any]) -> dict[str, Any]:
    artifacts = _artifact_items(record.get("artifact_refs") or [])
    return {
        "nodeId": str(record.get("node_id") or ""),
        "role": str(record.get("role") or ""),
        "runRef": str(record.get("run_ref") or ""),
        "status": str(record.get("status") or ""),
        "failure": str(record.get("failure_reason") or ""),
        "artifacts": artifacts,
    }


def _dynamic_subagent_spec_dashboard_item(
    spec: dict[str, Any],
    *,
    workflow: dict[str, Any],
    workflow_summary: str,
    node: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    workflow_id = str(workflow.get("workflow_id") or "")
    work_item_id = str(workflow.get("work_item_id") or "")
    relative_path = str(workflow.get("relative_path") or workflow_id or work_item_id)
    subagent_id = str(spec.get("subagent_id") or "")
    role = str(spec.get("role_name") or record.get("role") or "")
    node_id = str(node.get("node_id") or record.get("node_id") or "")
    allowed_tools = _string_list(spec.get("allowed_tools"))
    required_skills = _string_list(spec.get("required_skills"))
    artifact_inputs = _string_list(spec.get("artifact_inputs") or node.get("input_artifacts"))
    artifact_outputs = _string_list(spec.get("artifact_outputs") or node.get("output_artifacts"))
    status = str(record.get("status") or workflow.get("status") or "planned")
    artifacts = _artifact_items(record.get("artifact_refs") or [])
    raw = {
        "workflow": {
            "workflow_id": workflow_id,
            "work_item_id": work_item_id,
            "relative_path": relative_path,
            "status": workflow.get("status", ""),
            "execution_mode": workflow.get("execution_mode", ""),
            "planner_backend_id": workflow.get("planner_backend_id", ""),
            "default_worker_backend_id": workflow.get("default_worker_backend_id", ""),
            "summary": workflow_summary,
        },
        "subagent_spec": spec,
        "workflow_node": node,
        "execution_record": record,
    }
    return {
        "id": f"dynamic-subagent:{relative_path}:{subagent_id or role}:{node_id}",
        "kind": "dynamic-subagent",
        "workflowId": workflow_id,
        "workItemId": work_item_id,
        "relativePath": relative_path,
        "nodeId": node_id,
        "subagentId": subagent_id,
        "role": role,
        "runRef": str(record.get("run_ref") or ""),
        "status": status,
        "goal": _short_text(str(spec.get("goal") or ""), 1200),
        "systemPrompt": _short_text(str(spec.get("system_prompt") or ""), 1800),
        "allowedTools": allowed_tools,
        "requiredSkills": required_skills,
        "artifactInputs": artifact_inputs,
        "artifactOutputs": artifact_outputs,
        "maxTurns": spec.get("max_turns"),
        "maxToolCalls": spec.get("max_tool_calls"),
        "llmBackendId": str(spec.get("llm_backend_id") or ""),
        "failure": str(record.get("failure_reason") or ""),
        "artifacts": artifacts,
        "detail": {
            "workflow": workflow_id,
            "work item": work_item_id,
            "role": role,
            "node": node_id,
            "subagent": subagent_id,
            "status": status,
            "run": str(record.get("run_ref") or ""),
            "backend": str(spec.get("llm_backend_id") or ""),
            "turn budget": spec.get("max_turns"),
            "tool budget": spec.get("max_tool_calls"),
            "tools": allowed_tools,
            "required skills": required_skills,
            "inputs": artifact_inputs,
            "outputs": artifact_outputs,
        },
        "raw": _compact_value(raw, 7600),
        "search": _search_text(raw),
    }


def _dynamic_subagent_run_info(dynamic_subagent: dict[str, Any] | None) -> dict[str, Any]:
    if not dynamic_subagent:
        return {}
    return {
        "workflow_id": dynamic_subagent.get("workflowId", ""),
        "work_item_id": dynamic_subagent.get("workItemId", ""),
        "node_id": dynamic_subagent.get("nodeId", ""),
        "subagent_id": dynamic_subagent.get("subagentId", ""),
        "allowed_tools": list(dynamic_subagent.get("allowedTools") or []),
        "required_skills": list(dynamic_subagent.get("requiredSkills") or []),
    }


def _dynamic_subagents_by_node(
    dynamic_subagents: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in dynamic_subagents:
        workflow_id = str(item.get("workflowId") or "")
        work_item_id = str(item.get("workItemId") or "")
        node_id = str(item.get("nodeId") or "")
        if not node_id:
            continue
        for key in (
            (workflow_id, work_item_id, node_id),
            (workflow_id, "", node_id),
            ("", work_item_id, node_id),
        ):
            index.setdefault(key, item)
    return index


def _dynamic_subagent_for_run(
    run: Any,
    *,
    dynamic_by_run: dict[str, dict[str, Any]],
    dynamic_by_node: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    direct = dynamic_by_run.get(run.run_ref)
    if direct:
        return direct
    for key in _dynamic_node_lookup_keys(_asdict(run.metadata)):
        match = dynamic_by_node.get(key)
        if match:
            return match
    return None


def _dynamic_node_lookup_keys(metadata: dict[str, Any]) -> list[tuple[str, str, str]]:
    payload = _assigned_task_payload(metadata)
    node_payload = _asdict(payload.get("node"))
    if not isinstance(node_payload, dict):
        node_payload = {}
    workflow_id = str(
        metadata.get("dynamic_workflow_id")
        or payload.get("dynamic_workflow_id")
        or payload.get("workflow_id")
        or ""
    )
    node_id = str(
        metadata.get("dynamic_node_id")
        or payload.get("node_id")
        or node_payload.get("node_id")
        or ""
    )
    work_item_id = str(
        metadata.get("work_item_id")
        or payload.get("work_item_id")
        or _work_item_id_from_assigned_task(payload)
        or ""
    )
    keys: list[tuple[str, str, str]] = []
    if not node_id:
        return keys
    if workflow_id and work_item_id:
        keys.append((workflow_id, work_item_id, node_id))
    if workflow_id:
        keys.append((workflow_id, "", node_id))
    if work_item_id:
        keys.append(("", work_item_id, node_id))
    return keys


def _assigned_task_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    assigned = metadata.get("assigned_task") or metadata.get("dynamic_assigned_task")
    if isinstance(assigned, str):
        try:
            payload = json.loads(assigned)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    assigned = _asdict(assigned)
    return assigned if isinstance(assigned, dict) else {}


def _work_item_id_from_assigned_task(payload: dict[str, Any]) -> str:
    for key in ("work_item_context", "metadata"):
        nested = _asdict(payload.get(key))
        if isinstance(nested, dict) and nested.get("work_item_id"):
            return str(nested.get("work_item_id"))
    for key in ("available_input_artifact_refs", "available_input_artifacts"):
        refs = payload.get(key)
        if not isinstance(refs, list):
            continue
        for ref in refs:
            ref_payload = _asdict(ref)
            if not isinstance(ref_payload, dict):
                continue
            metadata = _asdict(ref_payload.get("metadata"))
            if isinstance(metadata, dict) and metadata.get("work_item_id"):
                return str(metadata.get("work_item_id"))
    return ""


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


def _skill_update_item(update: dict[str, Any]) -> dict[str, Any]:
    if not update:
        return {"status": ""}
    summary = _asdict(update.get("update_summary"))
    if not isinstance(summary, dict):
        summary = {}
    output_paths = _asdict(summary.get("output_paths"))
    if not isinstance(output_paths, dict):
        output_paths = {}
    return {
        "status": str(update.get("status") or ""),
        "proposal_count": summary.get("proposal_count", 0),
        "decision_count": summary.get("decision_count", 0),
        "applied_count": summary.get("applied_count", 0),
        "staged_count": summary.get("staged_count", 0),
        "rejected_count": summary.get("rejected_count", 0),
        "changed_library": bool(summary.get("changed_library")),
        "staged_updates": bool(summary.get("staged_updates")),
        "before_state_hash": str(summary.get("before_state_hash") or summary.get("graph_state_hash_before") or ""),
        "after_state_hash": str(summary.get("after_state_hash") or summary.get("graph_state_hash_after") or ""),
        "observation_id": str(summary.get("observation_id") or ""),
        "output_paths": _compact_value(output_paths, 1000),
    }


def _memory_update_item(metadata: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        _asdict(metadata.get("agent_memory_update_result")),
        _asdict(metadata.get("task_memory_update_result")),
        _asdict(metadata.get("memory_update_result")),
    ]
    updates = [item for item in candidates if isinstance(item, dict) and item]
    if not updates:
        return {"status": ""}
    statuses = [str(item.get("status") or "") for item in updates if item.get("status")]
    metadata_items = [_asdict(item.get("metadata")) for item in updates]
    disabled = any(
        isinstance(item, dict) and bool(item.get("memory_disabled"))
        for item in metadata_items
    )
    scopes = [
        str(item.get("memory_scope") or item.get("memory_scope_id") or "")
        for item in metadata_items
        if isinstance(item, dict) and (item.get("memory_scope") or item.get("memory_scope_id"))
    ]
    artifact_count = sum(len(item.get("artifact_refs") or []) for item in updates)
    return {
        "status": _combined_status(statuses),
        "statuses": statuses,
        "disabled": disabled,
        "scopes": scopes,
        "artifact_count": artifact_count,
        "updates": _compact_value(updates, 1800),
    }


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


def _combined_status(statuses: list[str]) -> str:
    clean = [status for status in statuses if status]
    if not clean:
        return ""
    lowered = [status.lower() for status in clean]
    for marker in ("failed", "error", "interrupted"):
        for status in lowered:
            if marker in status:
                return clean[lowered.index(status)]
    for marker in ("updated", "staged", "skipped", "completed"):
        for status in lowered:
            if marker in status:
                return clean[lowered.index(status)]
    return clean[0]


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


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc), "_path": str(path)}


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = _asdict(json.loads(stripped))
            except Exception as exc:
                payload = {"_load_error": str(exc), "_path": str(path), "_line": line_number}
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _dict_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _list_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        payload = _asdict(item)
        if isinstance(payload, dict):
            result.append(payload)
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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


def _dynamic_workflow_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("work_item_id") or ""),
        str(record.get("workflow_id") or ""),
        str(record.get("relative_path") or ""),
    )


def _time(value: Any) -> str:
    return str(value)


def _infer_lab_root_from_trajectory_dir(trajectory_dir: Path) -> Path | None:
    path = Path(trajectory_dir)
    if path.name == "trajectory" and path.parent.name == "registries":
        return path.parent.parent
    return None


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


def _ha(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)
