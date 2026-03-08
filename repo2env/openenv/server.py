from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from openenv.core.env_server.http_server import create_app

from repo2env.openenv.environment import DEFAULT_EXAMPLE_REPO, Repo2EnvOpenEnvEnvironment
from repo2env.openenv.models import Repo2EnvAction, Repo2EnvObservation
from repo2env.openenv.webui import InferenceSessionRequest, UISessionManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("ENV_README_PATH", str(PROJECT_ROOT / "README.md"))


def _default_task_spec_path(repo_source: str) -> str | None:
    if repo_source.startswith(("http://", "https://", "git@")):
        return None
    candidate = Path(repo_source) / "task_spec.json"
    if candidate.exists():
        return str(candidate)
    return None


def create_environment() -> Repo2EnvOpenEnvEnvironment:
    return Repo2EnvOpenEnvEnvironment(
        default_repo_source=os.environ.get("REPO2ENV_DEFAULT_REPO", str(DEFAULT_EXAMPLE_REPO)),
        max_steps=int(os.environ.get("REPO2ENV_MAX_STEPS", "12")),
        coverage_target=float(os.environ.get("REPO2ENV_COVERAGE_TARGET", "75.0")),
    )


app = create_app(
    create_environment,
    Repo2EnvAction,
    Repo2EnvObservation,
    env_name="repo2env",
    max_concurrent_envs=int(os.environ.get("REPO2ENV_MAX_CONCURRENT_ENVS", "4")),
)
ui_sessions = UISessionManager(create_environment)
DEFAULT_UI_REPO_SOURCE = os.environ.get("REPO2ENV_DEFAULT_REPO", str(DEFAULT_EXAMPLE_REPO))
DEFAULT_UI_REPO_NAME = (
    Path(DEFAULT_UI_REPO_SOURCE).name
    if not DEFAULT_UI_REPO_SOURCE.startswith(("http://", "https://", "git@"))
    else DEFAULT_UI_REPO_SOURCE.rsplit("/", 1)[-1]
)
DEFAULT_UI_TASK_SPEC = _default_task_spec_path(DEFAULT_UI_REPO_SOURCE)


@app.get("/web", response_class=HTMLResponse)
def web_ui() -> str:
    repo_name = DEFAULT_UI_REPO_NAME
    repo_source = DEFAULT_UI_REPO_SOURCE
    task_spec = DEFAULT_UI_TASK_SPEC or "Auto-detected from the bundled repo"
    return (
        dedent("""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Repo2Env UI</title>
    <style>
      :root {
        --bg: #f2efe8;
        --panel: rgba(255, 252, 246, 0.92);
        --line: #d1c5b6;
        --text: #1d1a16;
        --muted: #6c645a;
        --accent: #0f766e;
        --accent-strong: #115e59;
        --danger: #b42318;
        --shadow: 0 18px 60px rgba(43, 34, 24, 0.12);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(17, 94, 89, 0.18), transparent 26%),
          radial-gradient(circle at top right, rgba(180, 35, 24, 0.1), transparent 22%),
          linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
      }
      .shell {
        max-width: 1240px;
        margin: 0 auto;
        padding: 26px 20px 56px;
      }
      .hero {
        display: grid;
        gap: 6px;
        margin-bottom: 14px;
      }
      .eyebrow {
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 12px;
        color: var(--muted);
      }
      h1 {
        margin: 0;
        font-size: clamp(24px, 3.2vw, 34px);
        line-height: 1.05;
        max-width: none;
      }
      .subhead {
        margin: 0;
        max-width: 62ch;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.5;
      }
      .layout {
        display: grid;
        grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
        gap: 20px;
      }
      .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }
      .panel-header {
        padding: 18px 20px 8px;
        border-bottom: 1px solid rgba(209, 197, 182, 0.55);
      }
      .panel-header h2 {
        margin: 0;
        font-size: 20px;
      }
      .panel-body {
        padding: 18px 20px 22px;
      }
      form {
        display: grid;
        gap: 12px;
      }
      label {
        display: grid;
        gap: 6px;
        font-size: 13px;
        color: var(--muted);
      }
      input {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
        font: inherit;
        color: var(--text);
        background: rgba(255, 255, 255, 0.72);
      }
      select {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px;
        font: inherit;
        color: var(--text);
        background: rgba(255, 255, 255, 0.72);
      }
      .button-row {
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 4px;
      }
      button {
        border: 0;
        border-radius: 999px;
        padding: 12px 18px;
        font: inherit;
        font-weight: 700;
        color: white;
        background: var(--accent);
        cursor: pointer;
      }
      button:hover { background: var(--accent-strong); }
      button:disabled { opacity: 0.55; cursor: default; }
      button.secondary {
        color: var(--text);
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid var(--line);
      }
      button.secondary:hover {
        background: rgba(255, 255, 255, 1);
      }
      .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 13px;
        background: rgba(17, 94, 89, 0.1);
        color: var(--accent-strong);
      }
      .status-chip.failed {
        background: rgba(180, 35, 24, 0.1);
        color: var(--danger);
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
      }
      .stat {
        border: 1px solid rgba(209, 197, 182, 0.7);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.7);
      }
      .stat-label {
        display: block;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 4px;
      }
      .stat-value {
        font-size: 28px;
        line-height: 1;
      }
      .section-title {
        margin: 0 0 10px;
        font-size: 15px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .test-box,
      .events {
        border: 1px solid rgba(209, 197, 182, 0.7);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.64);
      }
      .test-box {
        padding: 14px 16px;
        margin-bottom: 18px;
      }
      .events {
        padding: 12px;
        max-height: 680px;
        overflow: auto;
      }
      .timeline-item {
        display: grid;
        gap: 8px;
        border-radius: 12px;
        padding: 14px;
        background: rgba(248, 245, 239, 0.8);
        border: 1px solid rgba(209, 197, 182, 0.6);
      }
      .timeline-item + .timeline-item {
        margin-top: 6px;
      }
      .timeline-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
      }
      .timeline-title {
        font-weight: 700;
        font-size: 18px;
      }
      .timeline-meta {
        font-size: 12px;
        color: var(--muted);
      }
      .timeline-detail {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.45;
      }
      .reward-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 62px;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(17, 94, 89, 0.1);
        color: var(--accent-strong);
      }
      .reward-pill.negative {
        background: rgba(180, 35, 24, 0.1);
        color: var(--danger);
      }
      details {
        border-top: 1px solid rgba(209, 197, 182, 0.6);
        padding-top: 8px;
      }
      summary {
        cursor: pointer;
        color: var(--muted);
        font-size: 13px;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        line-height: 1.5;
      }
      .empty {
        color: var(--muted);
        padding: 14px;
      }
      .hidden {
        display: none !important;
      }
      @media (max-width: 900px) {
        .layout {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div class="eyebrow">Repo2Env Live UI</div>
        <h1>Watch one repo run.</h1>
        <p class="subhead">Pick a runner, start the episode, and watch actions, rewards, and test results arrive live.</p>
      </section>
      <section class="layout">
        <article class="panel">
          <div class="panel-header"><h2>Run Config</h2></div>
          <div class="panel-body">
            <form id="run-form">
              <label>Runner
                <select id="runner-select" name="runner">
                  <option value="baseline">Baseline</option>
                  <option value="gpt-5.2">gpt-5.2</option>
                  <option value="gpt-5.2-pro">gpt-5.2-pro</option>
                  <option value="gpt-5.2-codex">gpt-5.2-codex</option>
                  <option value="gpt-5.1">gpt-5.1</option>
                  <option value="gpt-5.1-codex">gpt-5.1-codex</option>
                  <option value="gpt-5-pro">gpt-5-pro</option>
                  <option value="gpt-5-codex">gpt-5-codex</option>
                  <option value="gpt-5">gpt-5</option>
                  <option value="gpt-5-mini">gpt-5-mini</option>
                  <option value="gpt-5-nano">gpt-5-nano</option>
                  <option value="gpt-4.1">gpt-4.1</option>
                  <option value="gpt-4o-mini">gpt-4o-mini</option>
                  <option value="gpt-4.1-mini">gpt-4.1-mini</option>
                  <option value="gpt-4.1-nano">gpt-4.1-nano</option>
                </select>
              </label>
              <label id="api-key-row">OpenAI-compatible API key
                <input id="api-key" name="api_key" type="password" placeholder="sk-..." required />
              </label>
              <label>Max steps
                <input id="max-steps" name="max_steps" type="number" min="1" max="100" value="12" />
              </label>
              <div class="button-row">
                <button id="start-button" type="submit">Start Episode</button>
                <button id="download-json-button" class="secondary hidden" type="button">Download JSON</button>
                <div id="run-status" class="status-chip">Idle</div>
              </div>
            </form>
          </div>
        </article>
        <article class="panel">
          <div class="panel-header"><h2>Live Episode</h2></div>
          <div class="panel-body">
            <div class="summary-grid">
              <div class="stat"><span class="stat-label">Repo</span><span id="repo-name" class="stat-value">__REPO_NAME__</span></div>
              <div class="stat"><span class="stat-label">Passing</span><span id="passing-count" class="stat-value">0</span></div>
              <div class="stat"><span class="stat-label">Failed</span><span id="failed-count" class="stat-value">0</span></div>
              <div class="stat"><span class="stat-label">Status</span><span id="status-text" class="stat-value" style="font-size:18px;line-height:1.2">Idle</span></div>
            </div>
            <h3 class="section-title">Latest Test Status</h3>
            <div id="test-box" class="test-box">
              <div class="empty">No test activity yet.</div>
            </div>
            <h3 class="section-title">Action Stream</h3>
            <div id="events" class="events">
              <div class="empty">Start a run to watch tool calls and rewards.</div>
            </div>
          </div>
        </article>
      </section>
    </main>
    <script>
      const form = document.getElementById("run-form");
      const startButton = document.getElementById("start-button");
      const runStatus = document.getElementById("run-status");
      const statusText = document.getElementById("status-text");
      const repoName = document.getElementById("repo-name");
      const passingCount = document.getElementById("passing-count");
      const failedCount = document.getElementById("failed-count");
      const testBox = document.getElementById("test-box");
      const events = document.getElementById("events");
      const runnerSelect = document.getElementById("runner-select");
      const apiKey = document.getElementById("api-key");
      const apiKeyRow = document.getElementById("api-key-row");
      const downloadJsonButton = document.getElementById("download-json-button");
      let activeSessionId = null;
      let sessionStream = null;
      let latestSession = null;

      function setStatus(text, failed = false) {
        runStatus.textContent = text;
        statusText.textContent = text;
        runStatus.classList.toggle("failed", failed);
      }

      function renderTestSummary(summary) {
        if (!summary) {
          testBox.innerHTML = '<div class="empty">No test activity yet.</div>';
          return;
        }
        const failingLocations = summary.failing_locations?.length
          ? `<div><strong>failing:</strong> ${summary.failing_locations.join(", ")}</div>`
          : "";
        const output = summary.output
          ? `
            <details>
              <summary>Raw pytest output</summary>
              <pre>${escapeHtml(summary.output)}</pre>
            </details>
          `
          : "";
        testBox.innerHTML = `
          <div><strong>passed:</strong> ${summary.passed ?? 0}</div>
          <div><strong>failed:</strong> ${summary.failed ?? 0}</div>
          <div><strong>errors:</strong> ${summary.errors ?? 0}</div>
          ${failingLocations}
          ${output}
        `;
      }

      function renderEvents(items) {
        if (!items.length) {
          events.innerHTML = '<div class="empty">Start a run to watch tool calls and rewards.</div>';
          return;
        }
        events.innerHTML = items.map((event) => {
          const summary = summarizeEvent(event);
          const reward = event.reward;
          const rewardText = reward == null ? "-" : `${reward > 0 ? "+" : ""}${reward}`;
          const rewardClass = reward != null && reward < 0 ? "reward-pill negative" : "reward-pill";
          return `
            <div class="timeline-item">
              <div class="timeline-head">
                <div>
                  <div class="timeline-title">${escapeHtml(summary.title)}</div>
                  <div class="timeline-detail">${escapeHtml(summary.detail)}</div>
                </div>
                <div class="${rewardClass}">${escapeHtml(rewardText)}</div>
              </div>
              <div class="timeline-meta">step ${event.step_count ?? 0}</div>
            </div>
          `;
        }).join("");
      }

      function summarizeEvent(event) {
        const toolError = event.tool_result?.error;
        if (event.type === "reset") {
          return {
            title: "Started the episode",
            detail: `${event.repo_name || "repo"} loaded with ${event.current_metrics?.failed_tests ?? 0} failing tests`,
          };
        }
        if (event.type === "error") {
          return {
            title: "Run failed",
            detail: event.error || "The run ended with an error before the next tool step.",
          };
        }
        if (event.type === "invalid_action") {
          return {
            title: "Model sent an invalid action",
            detail: event.error || `Applied penalty and fell back to ${event.fallback_tool || "a safe tool"}`,
          };
        }
        if (event.tool === "read_file") {
          return {
            title: "Checked a file",
            detail: event.args?.path || "Read one file",
          };
        }
        if (event.tool === "read_file_chunk") {
          return {
            title: "Checked part of a file",
            detail: `${event.args?.path || "file"} lines ${event.args?.start_line ?? "?"}-${event.args?.end_line ?? "?"}`,
          };
        }
        if (event.tool === "search_code") {
          return {
            title: "Searched the codebase",
            detail: event.args?.query ? `Query: ${event.args.query}` : "Ran a code search",
          };
        }
        if (event.tool === "get_test_failures") {
          return {
            title: "Checked failing tests",
            detail: "Read the current structured test failures",
          };
        }
        if (event.tool === "diff_working_tree") {
          return {
            title: "Reviewed current edits",
            detail: "Looked at the diff against the starting repo state",
          };
        }
        if (event.tool === "replace_in_file" || event.tool === "insert_after" || event.tool === "insert_before" || event.tool === "append_to_file" || event.tool === "write_file") {
          if (toolError) {
            return {
              title: "Edit failed",
              detail: toolError,
            };
          }
          return {
            title: "Edited a file",
            detail: event.args?.path || "Updated repository contents",
          };
        }
        if (event.tool === "run_tests") {
          const passed = event.tool_result?.passed ?? event.metrics?.passing_tests ?? 0;
          const failed = event.tool_result?.failed ?? event.metrics?.failed_tests ?? 0;
          return {
            title: "Ran the tests",
            detail: `${passed} passing, ${failed} failing`,
          };
        }
        if (event.tool === "submit") {
          return {
            title: "Submitted the episode",
            detail: "No further actions were taken",
          };
        }
        return {
          title: event.tool || event.type || "Step",
          detail: "Completed one environment action",
        };
      }

      function escapeHtml(text) {
        return String(text)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;");
      }

      function applySession(session) {
        latestSession = session;
        repoName.textContent = session.repo_name || "-";
        passingCount.textContent = String(session.current_metrics?.passing_tests ?? 0);
        failedCount.textContent = String(session.current_metrics?.failed_tests ?? 0);
        setStatus(session.status_message || session.status, session.status === "failed");
        renderTestSummary(session.latest_pytest_summary);
        renderEvents(session.events || []);
        downloadJsonButton.classList.toggle("hidden", !session.session_id);
        if (session.status === "completed" || session.status === "failed") {
          startButton.disabled = false;
          if (sessionStream) {
            sessionStream.close();
            sessionStream = null;
          }
        }
      }

      function connectSessionStream(sessionId) {
        if (sessionStream) {
          sessionStream.close();
        }
        sessionStream = new EventSource(`/api/ui/sessions/${sessionId}/stream`);
        sessionStream.onmessage = (event) => {
          const session = JSON.parse(event.data);
          applySession(session);
        };
        sessionStream.onerror = () => {
          if (latestSession?.status === "completed" || latestSession?.status === "failed") {
            if (sessionStream) {
              sessionStream.close();
              sessionStream = null;
            }
            return;
          }
          setStatus("Live stream disconnected", true);
          startButton.disabled = false;
          if (sessionStream) {
            sessionStream.close();
            sessionStream = null;
          }
        };
      }

      function syncRunnerMode() {
        const usingModel = runnerSelect.value !== "baseline";
        apiKey.disabled = !usingModel;
        apiKey.required = usingModel;
        apiKeyRow.classList.toggle("hidden", !usingModel);
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        startButton.disabled = true;
        setStatus("Starting");
        renderEvents([]);
        renderTestSummary(null);
        latestSession = null;
        downloadJsonButton.classList.add("hidden");
        const runnerValue = runnerSelect.value;
        const payload = {
          policy_kind: runnerValue === "baseline" ? "baseline_test_submit" : "model",
          model_name: runnerValue === "baseline" ? null : runnerValue,
          api_key: runnerValue === "baseline" ? null : apiKey.value.trim(),
          max_steps: Number(document.getElementById("max-steps").value) || null,
        };
        const response = await fetch("/api/ui/sessions", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const error = await response.text();
          setStatus(`Start failed: ${error}`, true);
          startButton.disabled = false;
          return;
        }
        const body = await response.json();
        activeSessionId = body.session_id;
        connectSessionStream(activeSessionId);
      });
      runnerSelect.addEventListener("change", syncRunnerMode);
      downloadJsonButton.addEventListener("click", () => {
        if (!latestSession) return;
        const blob = new Blob([JSON.stringify(latestSession, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        const safeRepo = (latestSession.repo_name || "repo").replace(/[^a-zA-Z0-9._-]+/g, "-");
        const safeRunner = (latestSession.model_name || latestSession.policy_kind || "run").replace(/[^a-zA-Z0-9._-]+/g, "-");
        link.href = url;
        link.download = `${safeRepo}-${safeRunner}-${latestSession.session_id}.json`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      });
      syncRunnerMode();
    </script>
  </body>
</html>""")
        .replace("__REPO_NAME__", repo_name)
        .replace("__REPO_SOURCE__", repo_source)
        .replace("__TASK_SPEC__", task_spec)
    )


@app.post("/api/ui/sessions")
def start_ui_session(request: InferenceSessionRequest) -> dict[str, str]:
    try:
        session = ui_sessions.start_session(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": session.session_id}


@app.get("/api/ui/sessions/{session_id}")
def get_ui_session(session_id: str) -> dict[str, object]:
    session = ui_sessions.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session id.")
    return session.to_public_dict()


@app.get("/api/ui/sessions/{session_id}/stream")
async def stream_ui_session(session_id: str) -> StreamingResponse:
    if ui_sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Unknown session id.")

    async def event_stream():
        revision = -1
        while True:
            state, payload = await asyncio.to_thread(ui_sessions.wait_for_event, session_id, revision, 15.0)
            if state == "missing":
                yield "event: error\ndata: " + json.dumps({"error": "Unknown session id."}) + "\n\n"
                break
            if state == "timeout":
                yield ": keep-alive\n\n"
                continue
            assert payload is not None
            revision = int(payload.get("revision", revision))
            yield f"id: {revision}\ndata: {json.dumps(payload)}\n\n"
            if payload.get("status") in {"completed", "failed"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Repo2Env OpenEnv server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
