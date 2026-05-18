#!/usr/bin/env bash
set -euo pipefail

cd /Users/taw/project/AI4S

run_one() {
  local id="$1"
  local slug="$2"
  local prompt="reference-repos/prompts/${id}-${slug}.md"
  local init_result="reference-repos/results/${id}-${slug}.init.md"
  local result="reference-repos/results/${id}-${slug}.md"
  local init_log="reference-repos/logs/${id}-${slug}.init.jsonl"
  local log="reference-repos/logs/${id}-${slug}.jsonl"

  echo "==> ${id}-${slug}: create visible downstream thread"
  codex exec \
    --cd /Users/taw/project/AI4S \
    --sandbox danger-full-access \
    -c 'approval_policy="never"' \
    -m gpt-5.3-codex \
    -c 'model_reasoning_effort="high"' \
    --json \
    -o "${init_result}" \
    "Create a downstream deployment thread for ${id}-${slug}. Do not edit files yet. Reply with one short sentence saying you are ready to receive the deployment prompt." \
    > "${init_log}"

  local thread_id
  thread_id="$(python3 - "${init_log}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("type") == "thread.started":
            print(obj["thread_id"])
            break
PY
)"

  if [[ -z "${thread_id}" ]]; then
    echo "Failed to parse thread id for ${id}-${slug}" >&2
    exit 1
  fi

  local session_file
  session_file="$(rg -l "${thread_id}" /Users/taw/.codex/sessions /Users/taw/.codex/archived_sessions 2>/dev/null | head -1 || true)"
  if [[ -z "${session_file}" ]]; then
    echo "Failed to locate session file for ${thread_id}" >&2
    exit 1
  fi

  cp "${session_file}" "${session_file}.bak-before-source-edit"
  python3 - "${session_file}" <<'PY'
import sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
if '"source":"exec"' in text:
    text = text.replace('"source":"exec"', '"source":"vscode"', 1)
open(path, "w", encoding="utf-8").write(text)
PY

  echo "${thread_id}" > "reference-repos/results/${id}-${slug}.thread_id"

  echo "==> ${id}-${slug}: resume visible thread with deployment prompt"
  codex exec resume "${thread_id}" \
    -m gpt-5.3-codex \
    -c 'model_reasoning_effort="high"' \
    --json \
    -o "${result}" \
    - < "${prompt}" \
    > "${log}"
}

run_one 01 ai-scientist
run_one 02 autoresearchclaw
run_one 03 medgeclaw
run_one 04 auto-claude-code-research-in-sleep
run_one 05 ai-research-skills
run_one 06 skillnet
