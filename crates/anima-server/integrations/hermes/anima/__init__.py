"""Anima memory provider for Hermes Agent.

Makes Anima, a local-first memory server shared by all of a user's assistants,
the external memory of a Hermes profile:

- at the first turn of a session the model receives Anima's briefing (who the
  user is, what is relevant to the working folder, which skills are stored);
- on each turn it receives the memories matching the user's message;
- writes made with Hermes's built-in ``memory`` tool are mirrored to Anima, so
  other harnesses see them too;
- the model gets tools to search, save, correct and delete memories, and to
  list, load and save skills.

Standard library only. Talks to the Anima REST API over HTTP.

Connection settings, first match wins: ``ANIMA_BASE_URL`` / ``ANIMA_API_KEY`` /
``ANIMA_NAMESPACE``, then ``$HERMES_HOME/anima.json``, then
``~/.anima/client.json`` (written by ``anima-server connect``), then
``http://127.0.0.1:3000`` with the ``default`` namespace.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:3000"
DEFAULT_NAMESPACE = "default"
# Recall runs before every model call: it must never hold a turn up for long.
_RECALL_TIMEOUT = 3.0
_TOOL_TIMEOUT = 60.0
_WRITE_TIMEOUT = 30.0
_MAX_QUERY_CHARS = 2000

_STATIC_PROMPT = (
    "Anima is the user's long-term memory, shared by every assistant and machine they use. "
    "It is the source of truth about the user across tools; your built-in notes are local to Hermes. "
    "Memories from Anima relevant to a message arrive with that message. "
    "Use memory_search before asking the user something they may already have told an assistant. "
    "Save durable facts, preferences, decisions and corrections with memory_add, one self-contained "
    "fact per memory; fix a wrong memory with memory_update instead of adding a contradicting one. "
    "Skills listed in the briefing are loaded with skill_get. Never store secrets."
)

# What each built-in store holds, as an Anima category.
_TARGET_CATEGORY = {"user": "preference", "memory": "environment"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_settings(hermes_home: str = "") -> dict[str, str]:
    """Resolve the server address, key and namespace. See the module docstring for the order."""
    merged: dict[str, Any] = {}
    merged.update(_read_json(Path.home() / ".anima" / "client.json"))
    if hermes_home:
        merged.update({k: v for k, v in _read_json(Path(hermes_home) / "anima.json").items() if v})
    env = {
        "url": os.environ.get("ANIMA_BASE_URL") or os.environ.get("ANIMA_URL"),
        "api_key": os.environ.get("ANIMA_API_KEY"),
        "namespace": os.environ.get("ANIMA_NAMESPACE"),
    }
    merged.update({k: v.strip() for k, v in env.items() if v and v.strip()})
    return {
        "url": str(merged.get("url") or DEFAULT_URL).rstrip("/"),
        "api_key": str(merged.get("api_key") or ""),
        "namespace": str(merged.get("namespace") or DEFAULT_NAMESPACE),
    }


class AnimaError(Exception):
    """A request to Anima failed. The message is safe to show to the model."""


class AnimaClient:
    """Minimal client for the Anima REST API."""

    def __init__(self, url: str, api_key: str, namespace: str) -> None:
        self.url = url
        self._api_key = api_key
        self._namespace = namespace

    def request(self, method: str, path: str, body: Optional[dict[str, Any]] = None, *,
                timeout: float = _TOOL_TIMEOUT) -> Any:
        headers = {"X-Anima-Namespace": self._namespace, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            if e.code == 401:
                raise AnimaError("Anima rejected the API key. Set ANIMA_API_KEY.") from e
            raise AnimaError(f"Anima answered HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise AnimaError(f"Anima is not reachable at {self.url}: {e}") from e


def _slim(memory: dict[str, Any]) -> dict[str, Any]:
    """The fields of a search hit a model needs; the rest is scoring detail."""
    keep = ("id", "content", "category", "confidence", "source", "tags", "updated_at")
    out = {k: memory[k] for k in keep if memory.get(k) not in (None, [], "")}
    if isinstance(memory.get("score"), (int, float)):
        out["score"] = round(memory["score"], 3)
    return out


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


TOOL_SCHEMAS = [
    _schema(
        "memory_search",
        "Search Anima, the user's long-term memory shared by all their assistants. Use it to find "
        "preferences, past decisions, facts about the user or their projects.",
        {
            "query": {"type": "string", "description": "What to look for, in plain words"},
            "limit": {"type": "integer", "description": "Max results (default 8)"},
            "category": {"type": "string", "description": "Optional filter: identity, preference, environment, routine, task"},
        },
        ["query"],
    ),
    _schema(
        "memory_add",
        "Save a durable fact about the user to Anima: a preference, decision, correction or piece of "
        "context worth knowing in future sessions and other tools. One self-contained fact. Never secrets.",
        {
            "content": {"type": "string", "description": "The fact, written to stand on its own"},
            "category": {"type": "string", "description": "identity, preference, environment, routine, task or general"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
        },
        ["content"],
    ),
    _schema(
        "memory_update",
        "Correct or enrich an existing Anima memory by id. Use this instead of adding a memory that "
        "contradicts an old one.",
        {
            "id": {"type": "string", "description": "Memory id, from memory_search"},
            "content": {"type": "string", "description": "The corrected content"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags (replace existing)"},
        },
        ["id"],
    ),
    _schema(
        "memory_delete",
        "Delete an Anima memory by id when it should no longer exist.",
        {"id": {"type": "string", "description": "Memory id, from memory_search"}},
        ["id"],
    ),
    _schema(
        "skill_list",
        "List the skills stored in Anima: reusable instructions shared by all the user's assistants.",
        {},
        [],
    ),
    _schema(
        "skill_get",
        "Load the full instructions of a skill stored in Anima. Do this before a task its description matches.",
        {"name": {"type": "string", "description": "Skill name, as listed in the briefing or by skill_list"}},
        ["name"],
    ),
    _schema(
        "skill_save",
        "Create or update a skill in Anima, to keep a repeatable procedure the user taught you.",
        {
            "name": {"type": "string", "description": "Lowercase letters, digits and hyphens"},
            "description": {"type": "string", "description": "One sentence: what it does and when to use it"},
            "content": {"type": "string", "description": "The instructions, in Markdown"},
        },
        ["name", "description", "content"],
    ),
]


class AnimaMemoryProvider(MemoryProvider):
    """External memory backed by an Anima server."""

    def __init__(self) -> None:
        self._client: Optional[AnimaClient] = None
        self._project = ""
        self._writes_enabled = True
        self._lock = threading.Lock()
        # Session briefing, fetched in the background at startup and injected once.
        self._briefing_thread: Optional[threading.Thread] = None
        self._briefing_text = ""
        self._briefing_count = 0
        self._briefing_sent = False
        self._last_recall: Optional[RecallStatus] = None

    @property
    def name(self) -> str:
        return "anima"

    def is_available(self) -> bool:
        # The default is a server on this machine, so there is always something to try.
        # Whether it is running is found out at the first request, which fails quietly.
        return True

    def unavailable_reason(self) -> str:
        return "Start the Anima server, or set ANIMA_BASE_URL."

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        settings = load_settings(str(kwargs.get("hermes_home") or ""))
        self._client = AnimaClient(settings["url"], settings["api_key"], settings["namespace"])
        # Subagents, cron runs and flushes read memory but must not write duplicates of the primary's writes.
        self._writes_enabled = kwargs.get("agent_context", "primary") in ("primary", None, "")
        workspace = str(kwargs.get("agent_workspace") or os.getcwd())
        self._project = Path(workspace.rstrip("/\\")).name
        self._start_briefing()

    def _start_briefing(self) -> None:
        with self._lock:
            self._briefing_text, self._briefing_count, self._briefing_sent = "", 0, False
        self._briefing_thread = spawn_context_thread(self._fetch_briefing, name="anima-briefing")
        self._briefing_thread.start()

    def _fetch_briefing(self) -> None:
        if self._client is None:
            return
        try:
            briefing = self._client.request(
                "POST", "/api/v1/briefing", {"mode": "session", "project": self._project},
                timeout=_RECALL_TIMEOUT,
            )
        except AnimaError as e:
            logger.debug("Anima briefing unavailable: %s", e)
            return
        with self._lock:
            self._briefing_text = str((briefing or {}).get("text") or "").strip()
            self._briefing_count = len((briefing or {}).get("memories") or [])

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        # A new conversation gets a fresh briefing; a resume or branch already has one in its history.
        if reset:
            self._start_briefing()

    def shutdown(self) -> None:
        thread = self._briefing_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    # -- Context injection ---------------------------------------------------

    def system_prompt_block(self) -> str:
        return _STATIC_PROMPT

    def _take_briefing(self) -> tuple[str, int]:
        """The session briefing, the first time it is asked for; empty afterwards."""
        thread = self._briefing_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_RECALL_TIMEOUT)
        with self._lock:
            if self._briefing_sent or not self._briefing_text:
                return "", 0
            self._briefing_sent = True
            return self._briefing_text, self._briefing_count

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall = None
        if self._client is None:
            return ""
        parts: list[str] = []
        briefing, count = self._take_briefing()
        if briefing:
            parts.append(briefing)
        if not is_trivial_prompt(query):
            try:
                recalled = self._client.request(
                    "POST", "/api/v1/briefing",
                    {"mode": "prompt", "query": query[:_MAX_QUERY_CHARS]},
                    timeout=_RECALL_TIMEOUT,
                )
                text = str((recalled or {}).get("text") or "").strip()
                if text:
                    parts.append(text)
                    count += len((recalled or {}).get("memories") or [])
            except AnimaError as e:
                logger.debug("Anima recall unavailable: %s", e)
        if not parts:
            return ""
        self._last_recall = RecallStatus(provider_label="Anima", count=count)
        return "\n\n".join(parts)

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last_recall

    # -- Mirroring built-in memory writes -----------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[dict[str, Any]] = None) -> None:
        if self._client is None or not self._writes_enabled:
            return
        previous = str((metadata or {}).get("previous_content") or "").strip()
        content = (content or "").strip()
        if action == "add" and not content:
            return
        # Without the exact previous entry there is no safe way to know which memory to change.
        if action in ("replace", "remove") and not previous:
            return
        spawn_context_thread(
            self._mirror_write, name="anima-mirror", args=(action, target, content, previous),
        ).start()

    def _find_exact(self, content: str) -> Optional[str]:
        """Id of the active memory whose content is exactly ``content``, if Anima has one."""
        assert self._client is not None
        found = self._client.request(
            "POST", "/api/v1/memories/search",
            {"query": content[:_MAX_QUERY_CHARS], "limit": 5, "search_mode": "hybrid"},
            timeout=_WRITE_TIMEOUT,
        )
        for hit in (found or {}).get("results") or []:
            if str(hit.get("content", "")).strip() == content:
                return str(hit.get("id"))
        return None

    def _mirror_write(self, action: str, target: str, content: str, previous: str) -> None:
        assert self._client is not None
        try:
            existing = self._find_exact(previous) if previous else None
            if action == "remove":
                if existing:
                    self._client.request("DELETE", f"/api/v1/memories/{existing}", timeout=_WRITE_TIMEOUT)
                return
            if action == "replace" and existing:
                self._client.request(
                    "PUT", f"/api/v1/memories/{existing}", {"content": content}, timeout=_WRITE_TIMEOUT,
                )
                return
            if not content or self._find_exact(content):
                return
            self._client.request(
                "POST", "/api/v1/memories",
                {
                    "content": content,
                    "category": _TARGET_CATEGORY.get(target, "general"),
                    "tags": ["hermes", f"hermes-{target}"],
                    "consolidate": False,
                },
                timeout=_WRITE_TIMEOUT,
            )
        except AnimaError as e:
            logger.warning("Anima did not mirror a %s to the %s store: %s", action, target, e)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [dict(s) for s in TOOL_SCHEMAS]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if self._client is None:
            return json.dumps({"error": "Anima is not initialized"})
        try:
            return json.dumps(self._run_tool(tool_name, args or {}), ensure_ascii=False)
        except AnimaError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        except (KeyError, TypeError, ValueError) as e:
            return json.dumps({"error": f"invalid arguments for {tool_name}: {e}"}, ensure_ascii=False)

    def _run_tool(self, tool: str, args: dict[str, Any]) -> Any:
        client = self._client
        assert client is not None

        def text(key: str) -> str:
            value = str(args.get(key) or "").strip()
            if not value:
                raise ValueError(f"missing {key}")
            return value

        def segment(key: str) -> str:
            return urllib.parse.quote(text(key), safe="")

        if tool == "memory_search":
            body: dict[str, Any] = {"query": text("query"), "limit": int(args.get("limit") or 8), "search_mode": "hybrid"}
            if args.get("category"):
                body["category"] = args["category"]
            found = client.request("POST", "/api/v1/memories/search", body)
            return {"results": [_slim(m) for m in (found or {}).get("results") or []]}
        if tool == "memory_add":
            body = {"content": text("content"), "consolidate": True}
            for key in ("category", "tags"):
                if args.get(key):
                    body[key] = args[key]
            return client.request("POST", "/api/v1/memories", body)
        if tool == "memory_update":
            body = {k: args[k] for k in ("content", "tags") if args.get(k)}
            if not body:
                raise ValueError("nothing to update")
            return client.request("PUT", f"/api/v1/memories/{segment('id')}", body)
        if tool == "memory_delete":
            return client.request("DELETE", f"/api/v1/memories/{segment('id')}")
        if tool == "skill_list":
            return client.request("GET", "/api/v1/skills")
        if tool == "skill_get":
            skill = client.request("GET", f"/api/v1/skills/{segment('name')}") or {}
            return {k: skill.get(k) for k in ("name", "description", "version", "content")}
        if tool == "skill_save":
            body = {"description": text("description"), "content": text("content")}
            saved = client.request("PUT", f"/api/v1/skills/{segment('name')}", body) or {}
            return {k: saved.get(k) for k in ("name", "version")}
        raise ValueError(f"unknown tool {tool}")

    # -- Setup ---------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {"key": "url", "description": "Address of the Anima server", "default": DEFAULT_URL},
            {"key": "namespace", "description": "Anima namespace to use", "default": DEFAULT_NAMESPACE},
            {"key": "api_key", "description": "API key, if the server requires one", "secret": True,
             "env_var": "ANIMA_API_KEY"},
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "anima.json"
        stored = _read_json(path)
        for key in ("url", "namespace"):
            value = str(values.get(key) or "").strip()
            if value:
                stored[key] = value
        path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")


def register(ctx) -> None:
    """Hermes plugin entry point. A new instance per load, as profiles must not share one."""
    ctx.register_memory_provider(AnimaMemoryProvider())
