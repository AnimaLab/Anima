#!/usr/bin/env python3
"""LoCoMo benchmark evaluation harness for anima.

Evaluates anima's memory system against the LoCoMo benchmark:
  1. Downloads the LoCoMo dataset (locomo10.json)
  2. Ingests conversations into a running anima server
  3. Retrieves memories and generates answers via LLM
  4. Scores answers with an LLM judge
  5. Reports per-category and overall accuracy

Usage:
    python run.py [--anima-url URL] [--llm-model MODEL] [--judge-model MODEL]

Requires: .env file with OPENAI_API_KEY (Groq key) or set env vars directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import httpx
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Thinking mode: THINK_EXTRA for answer generation, NO_THINK_EXTRA for auxiliary
# tasks (reflection, synthesis, judging) where thinking wastes tokens.
THINK_EXTRA = {"chat_template_kwargs": {"enable_thinking": True}}
NO_THINK_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def strip_thinking(text: str | None) -> str:
    """Strip thinking artifacts from model output (Qwen3/3.5).

    Handles None content (can happen with thinking mode when all output
    goes to reasoning tokens and content is empty).
    """
    if text is None:
        return ""
    # XML-style <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Unclosed <think> blocks (truncated)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    # Plain-text "Thinking Process:" blocks that fill the entire response
    text = re.sub(r"^Thinking Process:.*", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Retry helper for transient API errors
# ---------------------------------------------------------------------------

async def retry_api_call(coro_fn, max_retries=5, base_delay=2.0):
    """Retry an async API call with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(k in err_str for k in [
                "connection refused", "reset before headers", "502", "503",
                "504", "rate_limit", "overloaded", "timeout", "internal"
            ])
            if not is_transient or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            console.print(f"[yellow]API error (attempt {attempt+1}/{max_retries}), retrying in {delay:.0f}s: {str(e)[:80]}[/yellow]")
            await asyncio.sleep(delay)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCOMO_DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

CATEGORY_NAMES = {
    1: "Single-hop",
    2: "Multi-hop",
    3: "Temporal",
    4: "Open-domain",
    5: "Adversarial",
}

EXTRACT_SYSTEM_PROMPT = """\
You are a research assistant. Given a question and conversation excerpts,
extract ALL specific facts that could help answer the question.

CRITICAL: Preserve EXACT details from the text:
- Specific names, places, countries, cities
- Exact numbers (how many children, how many times, etc.)
- Specific objects described (e.g. "a cup with a dog face", not just "pottery")
- Exact words on signs, posters, or artworks
- Book titles, song names, event names

For dates: convert relative references to absolute dates.
"yesterday" in [8 May 2023] = 7 May 2023.
"last Friday" in [15 July 2023] = Friday BEFORE 15 July (NOT 15 July itself).
"last year" in [2023] = 2022.

GROUNDING RULE: When extracting a specific named entity (game name, organization, brand),
verify it appears in the provided text before including it. If only a category is mentioned
(e.g. "a board game") but no specific name is given, write "[name not specified]".
Do NOT substitute a well-known example from your general knowledge.

Output a numbered list. Be thorough and specific — never summarize when exact details exist."""

ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about past conversations.
You will be given extracted facts from conversation history, then a question.

Important rules:
- Answer based on the provided facts. TRUST the facts — do not contradict them.
- For INFERENCE questions ("would X", "is X likely", "what might X", "does X probably"):
  commit to a best-guess inference from the person's interests, values, and indirect evidence.
  Say "Likely yes", "Probably around 30", "Most likely X because Y" — not "I don't know".
  Indirect evidence counts: "in school" → age ≤ 30; "only dogs gave him joy + actively dating"
  → was lonely before meeting someone; "mentions beach frequently" → lives near beach.
- For FACTUAL questions (what/when/where/who): answer only from the provided facts.
  If no relevant fact is present, say "I don't know".
- Use ABSOLUTE dates (e.g. "7 May 2023"), never relative ones."""

JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating how well a generated answer matches a gold
reference answer. Score from 0.0 to 1.0 in 0.1 increments ONLY.
Allowed values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

- 1.0 = Fully correct. Conveys the same essential information as the gold answer.
  Semantically equivalent responses count (e.g. "Caesar salad" matches "Caesar salad with chicken").
  Different date formats for the same date count (e.g. "May 7th" vs "7 May").
  Paraphrases count (e.g. "she thinks it's amazing" matches "expressed pride and support").
  Same date in relative vs absolute form counts (e.g. "9-10 September" matches "the weekend before 13 September").
  IMPORTANT: Compute dates yourself. "weekend of 14-15 October" IS "the weekend before 20 October". "July 14" IS "the Friday before July 15".
  If dates are within 1-2 days of each other, they are effectively equivalent (score 0.8+). "20 May" ≈ "the Sunday before 25 May" since both refer to the same weekend.
  If the answer includes the correct date among multiple dates, score 0.7+.
- 0.7–0.9 = Mostly correct. Right answer with minor missing details or slight imprecision.
  Extra correct information beyond the gold answer should NOT lower the score.
  If the answer includes the gold answer's key facts plus additional correct details, score 0.8+.
- 0.4–0.6 = Partially correct. Right general topic but missing key details, or correct
  in some parts but wrong in others.
- 0.1–0.3 = Mostly wrong. Touches the right topic but the answer itself is incorrect or misleading.
- 0.0 = Completely wrong. Contradicts, fundamentally misses, or is unrelated to the gold answer.
  Only use 0.0 if the answer is ENTIRELY wrong or says "I don't know" when the gold has a specific answer.

Respond with a JSON object: {"reasoning": "<one sentence>", "score": <float 0.0–1.0>}
Only output the JSON object, nothing else."""

REFLECTION_PROMPT = """\
You are analyzing a conversation session between two people.
Extract ALL key facts, events, decisions, preferences, plans, and relationships mentioned.

Rules:
- Write one concise factual statement per line
- Include specific names, dates, places, and details
- Convert relative time references to absolute dates using the session timestamp
- Capture both speakers' information
- Include emotions, opinions, and plans — not just facts
- Do NOT add information not present in the conversation

Output ONLY the list of facts, one per line. No numbering, no bullets, no headers."""

console = Console()

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


async def send_discord(content: str, embed: dict | None = None) -> None:
    """Send a message to Discord via webhook. Silently ignores failures."""
    if not DISCORD_WEBHOOK_URL:
        return
    payload: dict = {}
    if embed:
        payload["embeds"] = [embed]
    else:
        payload["content"] = content
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(DISCORD_WEBHOOK_URL, json=payload)
    except Exception:
        pass


def _conv_summary_embed(sample_id: str, results: list[dict], conv_idx: int, total_convs: int) -> dict:
    """Build a Discord embed for a completed conversation evaluation."""
    cat_stats: dict[str, dict] = {}
    for r in results:
        name = r["category_name"]
        cat_stats.setdefault(name, {"correct": 0, "total": 0, "score_sum": 0.0})
        cat_stats[name]["total"] += 1
        if r["llm_score"] >= 0.5:
            cat_stats[name]["correct"] += 1
        cat_stats[name]["score_sum"] += r["llm_score"]

    total = len(results)
    correct = sum(s["correct"] for s in cat_stats.values())
    avg_score = sum(r["llm_score"] for r in results) / total if total else 0

    fields = []
    for name, s in sorted(cat_stats.items()):
        avg = s["score_sum"] / s["total"] if s["total"] else 0
        fields.append({"name": name, "value": f"{s['correct']}/{s['total']} ({avg:.3f})", "inline": True})

    color = 0x2ECC71 if avg_score >= 0.7 else 0xF39C12 if avg_score >= 0.5 else 0xE74C3C
    return {
        "title": f"Conversation {conv_idx}/{total_convs}: {sample_id}",
        "description": f"**{correct}/{total}** correct | LLM score: **{avg_score:.3f}**",
        "color": color,
        "fields": fields,
    }


def _final_summary_embed(summary: dict, config: dict) -> dict:
    """Build a Discord embed for the final benchmark summary."""
    fields = []
    for cat_name, stats in summary.get("category_breakdown", {}).items():
        fields.append({
            "name": cat_name,
            "value": f"Acc: {stats['accuracy']:.1%} | Score: {stats['llm_score']:.3f}",
            "inline": True,
        })
    fields.append({"name": "Avg answer time", "value": f"{summary['avg_answer_time_s']:.1f}s", "inline": True})
    fields.append({"name": "Avg judge time", "value": f"{summary['avg_judge_time_s']:.1f}s", "inline": True})

    score = summary["overall_llm_score"]
    color = 0x2ECC71 if score >= 0.7 else 0xF39C12 if score >= 0.5 else 0xE74C3C
    return {
        "title": "LoCoMo Benchmark Complete",
        "description": (
            f"**Overall: {summary['total_correct']}/{summary['total_questions']}** "
            f"({summary['overall_accuracy']:.1%}) | LLM score: **{score:.3f}**\n"
            f"Model: `{config.get('llm_model', '?')}`"
        ),
        "color": color,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Session index (for context expansion)
# ---------------------------------------------------------------------------


class SessionIndex:
    """In-memory index of conversation turns grouped by session for context expansion."""

    def __init__(self):
        # {namespace: {session_key: [content_string, ...]}}
        self._index: dict[str, dict[str, list[str]]] = {}

    def add(self, namespace: str, session_key: str, content: str) -> None:
        self._index.setdefault(namespace, {}).setdefault(session_key, []).append(content)

    def get_window(self, namespace: str, session_key: str, content: str, window: int) -> list[str]:
        """Get a window of turns around the given content in its session."""
        sessions = self._index.get(namespace, {})
        turns = sessions.get(session_key, [])
        if not turns:
            return [content]
        # Find the turn in the session
        try:
            idx = turns.index(content)
        except ValueError:
            return [content]
        start = max(0, idx - window)
        end = min(len(turns), idx + window + 1)
        return turns[start:end]

    def get_session(self, namespace: str, session_key: str) -> list[str]:
        """Get all turns in a session."""
        return self._index.get(namespace, {}).get(session_key, [])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def download_dataset(data_path: Path) -> None:
    """Download locomo10.json if not present."""
    if data_path.exists():
        console.print(f"[dim]Dataset already exists at {data_path}[/dim]")
        return
    console.print(f"Downloading LoCoMo dataset to {data_path} ...")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LOCOMO_DATA_URL, str(data_path))
    console.print("[green]Download complete.[/green]")


def load_dataset(data_path: Path) -> list[dict]:
    """Load and return the LoCoMo dataset."""
    with open(data_path) as f:
        return json.load(f)


def extract_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """Extract (session_key, timestamp, turns) from a conversation object, sorted numerically."""
    sessions = []
    for key in conversation.keys():
        if key.startswith("session_") and not key.endswith(("_date_time", "_observation", "_summary")):
            ts_key = f"{key}_date_time"
            timestamp = conversation.get(ts_key, "")
            turns = conversation[key]
            if isinstance(turns, list):
                sessions.append((key, timestamp, turns))
    # Sort numerically: session_1, session_2, ..., session_10, session_11
    sessions.sort(key=lambda s: int(s[0].split("_")[1]))
    return sessions


# ---------------------------------------------------------------------------
# Anima client
# ---------------------------------------------------------------------------


class AnimaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=600.0)

    async def health(self) -> bool:
        try:
            r = await self.http.get("/health")
            return r.status_code == 200
        except httpx.ConnectError:
            return False

    async def wait_for_processor_idle(self, poll_interval: float = 5.0, timeout: float = 1800.0) -> None:
        """Poll /api/v1/processor/status until the background processor is idle."""
        import asyncio
        elapsed = 0.0
        while elapsed < timeout:
            try:
                r = await self.http.get("/api/v1/processor/status")
                if r.status_code == 200:
                    data = r.json()
                    if data.get("idle", True):
                        return
                    q = data.get("queue_depth", 0)
                    f = data.get("in_flight", 0)
                    console.print(f"[dim]  Processor busy: {q} queued, {f} in-flight — waiting...[/dim]")
            except Exception:
                pass
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        console.print("[yellow]  Processor idle-wait timed out after {timeout}s — proceeding anyway[/yellow]")

    async def add_memory(
        self,
        namespace: str,
        content: str,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        memory_type: str = "event",
    ) -> dict:
        r = await self.http.post(
            "/api/v1/memories",
            headers={"X-Anima-Namespace": namespace},
            json={
                "content": content,
                "metadata": metadata or {},
                "consolidate": False,
                "tags": tags or [],
                "memory_type": memory_type,
            },
        )
        r.raise_for_status()
        return r.json()

    async def add_memories_batch(
        self,
        namespace: str,
        items: list[dict],
        reflect: bool = False,
    ) -> dict:
        r = await self.http.post(
            "/api/v1/memories/batch",
            headers={"X-Anima-Namespace": namespace},
            json={
                "items": items,
                "reflect": reflect,
            },
        )
        r.raise_for_status()
        return r.json()

    async def search(
        self,
        namespace: str,
        query: str,
        limit: int = 20,
        search_mode: str = "hybrid",
    ) -> dict:
        r = await self.http.post(
            "/api/v1/memories/search",
            headers={"X-Anima-Namespace": namespace},
            json={
                "query": query,
                "limit": limit,
                "search_mode": search_mode,
            },
        )
        r.raise_for_status()
        return r.json()

    async def ask(
        self,
        namespace: str,
        question: str,
        search_limit: int = 20,
        max_results: int = 20,
        llm: dict | None = None,
        memory_types: list[str] | None = None,
    ) -> dict:
        import asyncio as _asyncio
        body = {
            "question": question,
            "search_limit": search_limit,
            "max_results": max_results,
        }
        if llm:
            body["llm"] = llm
        if memory_types:
            body["memory_types"] = memory_types
        for attempt in range(5):
            r = await self.http.post(
                "/api/v1/ask",
                headers={"X-Anima-Namespace": namespace},
                json=body,
            )
            if r.status_code == 500 and attempt < 4:
                await _asyncio.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()

    async def close(self):
        await self.http.aclose()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def ingest_conversation(
    anima: AnimaClient,
    sample: dict,
    namespace_prefix: str,
    session_index: SessionIndex,
    progress: Progress,
) -> int:
    """Ingest per-turn memories with timestamp prefix. Returns turn count."""
    sample_id = sample["sample_id"]
    namespace = f"{namespace_prefix}/{sample_id}"
    conversation = sample["conversation"]
    sessions = extract_sessions(conversation)

    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    total_turns = sum(len(turns) for _, _, turns in sessions)
    task = progress.add_task(f"  {sample_id}", total=total_turns)

    count = 0
    batch_items: list[dict] = []
    print(f"  [DEBUG] {sample_id}: {len(sessions)} sessions, {total_turns} turns", flush=True)
    for session_key, timestamp, turns in sessions:
        for turn in turns:
            speaker_raw = turn.get("speaker", "")
            if speaker_raw == "speaker_a":
                speaker = speaker_a
            elif speaker_raw == "speaker_b":
                speaker = speaker_b
            else:
                speaker = speaker_raw

            text = turn.get("text", "")
            if not text.strip():
                progress.advance(task)
                continue

            # Append image captions so visual context is preserved
            blip = turn.get("blip_caption", "")
            if blip:
                text = f"{text} [Shared image: {blip}]"

            # Prepend timestamp so the LLM always knows when this was said
            content = f"[{timestamp}] {speaker}: {text}" if timestamp else f"{speaker}: {text}"
            metadata = {
                "session": session_key,
                "dia_id": turn.get("dia_id", ""),
                "speaker": speaker,
                "timestamp": timestamp,
            }
            tags = ["locomo", session_key]

            # Track in session index for context expansion
            session_index.add(namespace, session_key, content)
            batch_items.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "tags": tags,
                }
            )
            if len(batch_items) >= 64:
                try:
                    # During ingest we skip server-side reflection here; explicit reflect pass follows.
                    print(f"  [DEBUG] {sample_id}: sending batch of {len(batch_items)}...", flush=True)
                    await anima.add_memories_batch(namespace, batch_items, reflect=False)
                    print(f"  [DEBUG] {sample_id}: batch done, total={count}", flush=True)
                except Exception as exc:
                    print(f"  [DEBUG] {sample_id}: batch failed ({exc}), falling back", flush=True)
                    # Fallback for older servers that don't expose /api/v1/memories/batch.
                    for item in batch_items:
                        await anima.add_memory(
                            namespace,
                            item["content"],
                            item.get("metadata"),
                            item.get("tags"),
                        )
                batch_items.clear()
            count += 1
            progress.advance(task)

    if batch_items:
        try:
            await anima.add_memories_batch(namespace, batch_items, reflect=False)
        except Exception:
            for item in batch_items:
                await anima.add_memory(
                    namespace,
                    item["content"],
                    item.get("metadata"),
                    item.get("tags"),
                )

    return count


async def ingest_all(
    anima: AnimaClient,
    dataset: list[dict],
    namespace_prefix: str,
    session_index: SessionIndex,
    conversation_ids: list[str] | None = None,
) -> None:
    """Ingest all (or selected) conversations."""
    samples = dataset
    if conversation_ids:
        samples = [s for s in dataset if s["sample_id"] in conversation_ids]

    console.print(f"\n[bold]Ingesting {len(samples)} conversations...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        total = 0
        for sample in samples:
            n = await ingest_conversation(anima, sample, namespace_prefix, session_index, progress)
            total += n
    console.print(f"[green]Ingested {total} memories across {len(samples)} conversations.[/green]")


def _rebuild_session_index(
    dataset: list[dict],
    namespace_prefix: str,
    session_index: SessionIndex,
    conversation_ids: list[str] | None = None,
) -> None:
    """Rebuild session index from dataset (used when --skip-ingest)."""
    samples = dataset
    if conversation_ids:
        samples = [s for s in dataset if s["sample_id"] in conversation_ids]
    for sample in samples:
        sample_id = sample["sample_id"]
        namespace = f"{namespace_prefix}/{sample_id}"
        conversation = sample["conversation"]
        sessions = extract_sessions(conversation)
        speaker_a = conversation.get("speaker_a", "Speaker A")
        speaker_b = conversation.get("speaker_b", "Speaker B")
        for session_key, timestamp, turns in sessions:
            for turn in turns:
                speaker_raw = turn.get("speaker", "")
                if speaker_raw == "speaker_a":
                    speaker = speaker_a
                elif speaker_raw == "speaker_b":
                    speaker = speaker_b
                else:
                    speaker = speaker_raw
                text = turn.get("text", "")
                if text.strip():
                    content = f"[{timestamp}] {speaker}: {text}" if timestamp else f"{speaker}: {text}"
                    session_index.add(namespace, session_key, content)


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


async def reflect_conversation(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    sample: dict,
    namespace_prefix: str,
    session_index: SessionIndex,
    llm_model: str,
    progress: Progress,
    sem: asyncio.Semaphore,
) -> int:
    """Generate reflection memories from session summaries. Returns count of reflections."""
    sample_id = sample["sample_id"]
    namespace = f"{namespace_prefix}/{sample_id}"
    conversation = sample["conversation"]
    sessions = extract_sessions(conversation)

    task = progress.add_task(f"  {sample_id} (reflect)", total=len(sessions))
    count = 0

    async def _reflect_session(session_key, timestamp, turns):
        nonlocal count
        async with sem:
            session_turns = session_index.get_session(namespace, session_key)
            if not session_turns:
                progress.advance(task)
                return

            session_text = "\n".join(session_turns)
            prompt = f"""Session timestamp: {timestamp}

{session_text}"""

            try:
                response = await openai_client.chat.completions.create(
                    model=llm_model,
                    temperature=0.1,
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": REFLECTION_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body=NO_THINK_EXTRA,
                )
                facts = strip_thinking(response.choices[0].message.content)
            except Exception as e:
                console.print(f"[yellow]Reflection failed for {session_key}: {e}[/yellow]")
                progress.advance(task)
                return

            # Ingest each fact as a reflection memory
            for line in facts.split("\n"):
                line = line.strip().lstrip("•-*0123456789. ")
                if not line or len(line) < 10:
                    continue
                content = f"[{timestamp}] {line}" if timestamp else line
                await anima.add_memory(
                    namespace, content,
                    metadata={"session": session_key, "timestamp": timestamp},
                    tags=["locomo", session_key, "reflection"],
                    memory_type="reflection",
                )
                count += 1

            progress.advance(task)

    await asyncio.gather(*[
        _reflect_session(sk, ts, t) for sk, ts, t in sessions
    ])

    return count


async def reflect_all(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    dataset: list[dict],
    namespace_prefix: str,
    session_index: SessionIndex,
    llm_model: str,
    conversation_ids: list[str] | None = None,
) -> None:
    """Run reflection pass on all conversations."""
    samples = dataset
    if conversation_ids:
        samples = [s for s in dataset if s["sample_id"] in conversation_ids]

    console.print(f"\n[bold]Reflecting on {len(samples)} conversations...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        sem = asyncio.Semaphore(32)

        async def _reflect_one(sample):
            return await reflect_conversation(
                anima, openai_client, sample, namespace_prefix,
                session_index, llm_model, progress, sem,
            )

        results = await asyncio.gather(*[_reflect_one(s) for s in samples])
        total = sum(results)
    console.print(f"[green]Generated {total} reflection memories.[/green]")


# ---------------------------------------------------------------------------
# Temporal synthesis — cross-session state tracking
# ---------------------------------------------------------------------------

TEMPORAL_SYNTHESIS_PROMPT = """\
You are analyzing facts extracted from a long conversation spanning multiple sessions.
Your job: identify EVOLVING STATES or CONTRADICTIONS across time and produce temporal summaries.

Look for patterns like:
- Status changes: "X had a job (March) → X lost their job (August)"
- Relationship changes: "X was dating Y → X broke up with Y"
- Location changes: "X lived in A → X moved to B"
- Opinion/preference changes: "X liked Y → X no longer likes Y"
- Health/condition changes: "X was healthy → X got diagnosed with Z"

For each state change found, produce a summary capturing the LATEST known state AND the transition.

Output JSON only (no markdown):
{"summaries":[{"content":"As of [date], [person] [latest state]. Previously [old state].","event_date":"YYYY-MM-DD","importance":8}]}

Rules:
- Only produce summaries for genuine state CHANGES (not static facts).
- Use the most recent date from the source facts as event_date.
- If no state changes found, return {"summaries":[]}.
- Max 10 summaries per conversation.
- Be specific: include names, dates, details.

REFLECTED FACTS:
"""


async def temporal_synthesis_conversation(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    sample: dict,
    namespace_prefix: str,
    llm_model: str,
    sem: asyncio.Semaphore,
) -> int:
    """Detect state changes across sessions and produce temporal summaries."""
    sample_id = sample["sample_id"]
    namespace = f"{namespace_prefix}/{sample_id}"

    # Fetch all memories for this conversation (filter reflection-tagged ones)
    async with sem:
        all_reflected = []
        offset = 0
        while True:
            r = await anima.http.get(
                "/api/v1/memories",
                headers={"X-Anima-Namespace": namespace},
                params={"limit": 200, "offset": offset},
            )
            r.raise_for_status()
            data = r.json()
            memories = data.get("memories", [])
            for m in memories:
                tags = m.get("tags", [])
                if "reflection" in tags:
                    all_reflected.append(m)
            if len(memories) < 200:
                break
            offset += 200

        if len(all_reflected) < 5:
            console.print(f"[dim]  {sample_id}: only {len(all_reflected)} reflected facts, skipping synthesis[/dim]")
            return 0  # Too few facts to synthesize

        console.print(f"[dim]  {sample_id}: synthesizing from {len(all_reflected)} reflected facts[/dim]")
        # Format facts with dates for the LLM
        sorted_facts = sorted(all_reflected, key=lambda m: m.get("created_at", ""))
        fact_lines = [
            f"- [{m.get('created_at', '')[:10]}] {m['content']}"
            for m in sorted_facts
        ]

        # Truncate to fit context window (~30K tokens ≈ ~90K chars, leave margin)
        MAX_CHARS = 80000
        facts_text = "\n".join(fact_lines)
        if len(facts_text) > MAX_CHARS:
            # Keep most important facts — prioritize diversity by sampling evenly
            step = max(1, len(fact_lines) // (MAX_CHARS // 80))  # ~80 chars per fact avg
            sampled = fact_lines[::step] if step > 1 else fact_lines
            facts_text = "\n".join(sampled)
            if len(facts_text) > MAX_CHARS:
                facts_text = facts_text[:MAX_CHARS]
            console.print(f"[dim]  {sample_id}: truncated to {len(sampled)} facts for context window[/dim]")

        try:
            response = await openai_client.chat.completions.create(
                model=llm_model,
                temperature=0.1,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": TEMPORAL_SYNTHESIS_PROMPT},
                    {"role": "user", "content": facts_text},
                ],
                extra_body=NO_THINK_EXTRA,
            )
            raw = strip_thinking(response.choices[0].message.content)
        except Exception as e:
            console.print(f"[yellow]Temporal synthesis failed for {sample_id}: {e}[/yellow]")
            return 0

    # Parse JSON
    try:
        raw = raw[raw.index("{"):raw.rindex("}") + 1]
        parsed = json.loads(raw)
        summaries = parsed.get("summaries", [])
    except (json.JSONDecodeError, ValueError):
        return 0

    # Ingest temporal summaries as deduced memories with high importance
    count = 0
    for s in summaries[:10]:
        content = s.get("content", "").strip()
        if not content or len(content) < 15:
            continue
        event_date = s.get("event_date")
        await anima.add_memory(
            namespace, content,
            metadata={"tier": 3, "event_date": event_date, "temporal_synthesis": True},
            tags=["locomo", "temporal_synthesis", "deduced"],
            memory_type="deduced",
        )
        count += 1

    return count


async def temporal_synthesis_all(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    dataset: list[dict],
    namespace_prefix: str,
    llm_model: str,
    conversation_ids: list[str] | None = None,
) -> None:
    """Run temporal synthesis on all conversations."""
    samples = dataset
    if conversation_ids:
        samples = [s for s in dataset if s["sample_id"] in conversation_ids]

    console.print(f"\n[bold]Running temporal synthesis on {len(samples)} conversations...[/bold]")
    sem = asyncio.Semaphore(10)
    results = await asyncio.gather(*[
        temporal_synthesis_conversation(
            anima, openai_client, sample, namespace_prefix, llm_model, sem,
        )
        for sample in samples
    ])
    total = sum(results)
    console.print(f"[green]Generated {total} temporal synthesis memories.[/green]")


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

QUERY_EXPAND_PROMPT = """\
Given a question about a past conversation between friends, generate 3 short
keyword search queries (2-4 words each). Think about what specific words or
phrases the speakers would have actually said in casual conversation.

For each query, try a DIFFERENT angle:
1. The direct topic keywords
2. Related/synonym terms a speaker might use casually
3. The answer/outcome keywords (guess what the answer might contain)

Output ONLY the 3 queries, one per line. No numbering, no explanation."""


async def expand_query(
    openai_client: AsyncOpenAI,
    question: str,
    llm_model: str,
) -> list[str]:
    """Generate alternative search queries for better retrieval."""
    try:
        response = await openai_client.chat.completions.create(
            model=llm_model,
            temperature=0.3,
            max_tokens=512,
            messages=[
                {"role": "system", "content": QUERY_EXPAND_PROMPT},
                {"role": "user", "content": question},
            ],
            extra_body=NO_THINK_EXTRA,
        )
        raw = strip_thinking(response.choices[0].message.content)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
        queries = [q.strip().lstrip("0123456789.-) ") for q in raw.split("\n") if q.strip()]
        return queries[:3]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


async def answer_question(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    namespace: str,
    question: str,
    llm_model: str,
    search_limit: int,
    session_index: SessionIndex | None = None,
    context_window: int = 0,
) -> tuple[str, list[dict], int]:
    """Retrieve context from anima and generate an answer. Returns (answer, retrieved, tokens_used)."""
    # Multi-query search: original + expanded queries
    queries = [question]
    extra_queries = await expand_query(openai_client, question, llm_model)
    queries.extend(extra_queries)

    # Search with all queries and merge results (deduplicate by content)
    seen_content: set[str] = set()
    results: list[dict] = []
    for q in queries:
        if not q or not q.strip():
            continue
        try:
            # Search hybrid first
            search_result = await anima.search(namespace, q, limit=search_limit)
            for r in search_result.get("results", []):
                content = r.get("content", "")
                if content not in seen_content:
                    seen_content.add(content)
                    results.append(r)
            # Also search keyword-only (bypasses vector similarity threshold)
            kw_result = await anima.search(namespace, q, limit=search_limit // 2, search_mode="keyword")
            for r in kw_result.get("results", []):
                content = r.get("content", "")
                if content not in seen_content:
                    seen_content.add(content)
                    results.append(r)
        except Exception as e:
            console.print(f"[yellow]Search failed for query '{q[:50]}': {e}[/yellow]")

    if not results:
        return "I don't know.", [], 0

    # Build raw context from top results
    context_parts = []
    for i, r in enumerate(results[:100], 1):
        content = r.get("content", "")
        context_parts.append(f"{i}. {content}")
    raw_context = "\n".join(context_parts)

    # Step 1: Extract relevant facts (with thinking for date math)
    extract_prompt = f"""Question: {question}

Conversation excerpts:
{raw_context}

Extract the specific facts needed to answer this question:"""

    tokens_used = 0

    extract_resp = await retry_api_call(lambda: openai_client.chat.completions.create(
        model=llm_model,
        temperature=0.1,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": extract_prompt},
        ],
        extra_body=NO_THINK_EXTRA,
    ))
    tokens_used += extract_resp.usage.total_tokens if extract_resp.usage else 0
    extracted = strip_thinking(extract_resp.choices[0].message.content)

    # Step 2: Answer from extracted facts (concise, no thinking needed)
    answer_prompt = f"""Extracted facts:
{extracted}

Question: {question}

Answer concisely:"""

    response = await retry_api_call(lambda: openai_client.chat.completions.create(
        model=llm_model,
        temperature=0.1,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": answer_prompt},
        ],
        extra_body=THINK_EXTRA,  # Thinking ON for answer generation
    ))
    tokens_used += response.usage.total_tokens if response.usage else 0
    answer = strip_thinking(response.choices[0].message.content)
    return answer, results, tokens_used, raw_context


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


async def judge_adversarial(
    judge_client: AsyncOpenAI,
    question: str,
    adversarial_answer: str,
    generated_answer: str,
    judge_model: str,
) -> dict:
    """Judge adversarial question: correct if the system refuses to answer or says it doesn't know.
    Wrong if it hallucinates the adversarial_answer."""
    # Quick heuristic: check if answer is an IDK response
    gen_lower = generated_answer.lower().strip()
    idk_patterns = [
        "i don't know", "i do not know", "no information", "not mentioned",
        "no relevant", "cannot determine", "can't determine", "unable to determine",
        "not discussed", "no evidence", "not enough information", "unclear",
        "no data", "not available", "cannot find", "can't find",
        "no record", "not specified", "i'm not sure", "i am not sure",
        "there is no", "there's no", "no mention",
    ]
    is_idk = any(p in gen_lower for p in idk_patterns)

    if is_idk:
        return {"score": 1.0, "label": "CORRECT", "reasoning": "System correctly refused to answer (IDK)."}

    # Use LLM judge to check if the generated answer contains the hallucinated content
    prompt = f"""Question: {question}
Hallucinated answer (what a wrong system would say): {adversarial_answer}
Generated answer: {generated_answer}

The question is adversarial — it asks about something that was never discussed in the conversations.
A CORRECT system should say "I don't know" or refuse to answer.
A WRONG system would hallucinate and provide the adversarial answer or similar fabricated details.

Score 1.0 if the system correctly refuses/says it doesn't know.
Score 0.0 if the system confidently provides the hallucinated answer or fabricated details.
Score 0.5 if the system partially hedges but still provides some fabricated content.

Respond with JSON: {{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

    response = await retry_api_call(lambda: judge_client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are a precise evaluation judge. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        extra_body=NO_THINK_EXTRA,
    ))

    raw = strip_thinking(response.choices[0].message.content)
    try:
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        result = json.loads(raw)
    except json.JSONDecodeError:
        score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if score_match:
            result = {"reasoning": raw, "score": float(score_match.group(1))}
        else:
            result = {"reasoning": raw, "score": 0.5}

    score = max(0.0, min(1.0, float(result.get("score", 0.5))))
    score = round(score * 10.0) / 10.0
    result["score"] = score
    result["label"] = "CORRECT" if score >= 0.5 else "WRONG"
    return result


async def judge_answer(
    judge_client: AsyncOpenAI,
    question: str,
    gold_answer: str,
    generated_answer: str,
    judge_model: str,
) -> dict:
    """Judge answer quality on a 0.0-1.0 scale with 0.1 increments."""
    prompt = f"""Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Evaluate whether the generated answer is correct based on the gold answer."""

    response = await retry_api_call(lambda: judge_client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_body=NO_THINK_EXTRA,
    ))

    raw = strip_thinking(response.choices[0].message.content)
    # Parse JSON from response
    try:
        # Handle potential markdown code blocks
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract score or label
        score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if score_match:
            result = {"reasoning": raw, "score": float(score_match.group(1))}
        elif "CORRECT" in raw.upper():
            result = {"reasoning": raw, "score": 1.0}
        else:
            result = {"reasoning": raw, "score": 0.0}

    # Normalise: support both old label format and score format
    if "score" not in result:
        result["score"] = 1.0 if result.get("label", "WRONG") == "CORRECT" else 0.0
    score = max(0.0, min(1.0, float(result["score"])))
    # Force 0.1 increments for judge stability/reproducibility.
    score = round(score * 10.0) / 10.0
    result["score"] = score
    # Derive binary label only for backward compatibility with existing scripts.
    result["label"] = "CORRECT" if result["score"] >= 0.5 else "WRONG"

    return result


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


async def evaluate(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    dataset: list[dict],
    namespace_prefix: str,
    llm_model: str,
    judge_model: str,
    search_limit: int,
    conversation_ids: list[str] | None = None,
    session_index: SessionIndex | None = None,
    context_window: int = 0,
    use_ask_api: bool = False,
    judge_client: AsyncOpenAI | None = None,
    question_limit: int = 0,
    output_path: str | None = None,
    memory_types: list[str] | None = None,
    skip_adversarial: bool = False,
) -> list[dict]:
    """Run QA evaluation across all conversations."""
    samples = dataset
    if conversation_ids:
        samples = [s for s in dataset if s["sample_id"] in conversation_ids]

    all_results = []
    allowed_cats = (1, 2, 3, 4) if skip_adversarial else (1, 2, 3, 4, 5)
    total_questions = sum(
        len([q for q in s.get("qa", []) if q.get("category", 5) in allowed_cats])
        for s in samples
    )
    if question_limit > 0:
        total_questions = min(total_questions, question_limit)

    mode_label = "via /api/v1/ask" if use_ask_api else "via Python pipeline"
    console.print(f"\n[bold]Evaluating {total_questions} questions across {len(samples)} conversations ({mode_label})...[/bold]")

    # Prepare output directory: results/<run-name>/ with one file per conversation
    output_dir = None
    if output_path:
        output_dir = Path(output_path).parent / Path(output_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build flat list of (conv_idx, sample, qa) work items
    work_items = []
    for conv_idx, sample in enumerate(samples, 1):
        for qa in sample.get("qa", []):
            category = qa.get("category", 5)
            if category not in allowed_cats:
                continue
            is_adversarial = category == 5
            if is_adversarial and skip_adversarial:
                continue
            work_items.append((conv_idx, sample, qa))
    if question_limit > 0:
        work_items = work_items[:question_limit]
    total_questions = len(work_items)

    sem = asyncio.Semaphore(64)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=total_questions)

        async def _eval_one(conv_idx: int, sample: dict, qa: dict) -> dict:
            async with sem:
                sample_id = sample["sample_id"]
                namespace = f"{namespace_prefix}/{sample_id}"
                category = qa.get("category", 5)
                is_adversarial = category == 5
                question = qa["question"]
                gold_answer = qa.get("adversarial_answer", "") if is_adversarial else qa["answer"]

                # Generate answer
                t0 = time.time()
                tokens_used = 0
                context_text = ""
                if use_ask_api:
                    ask_llm = {
                        "base_url": str(openai_client.base_url).rstrip("/"),
                        "model": llm_model,
                        "api_key": openai_client.api_key or "",
                    }
                    ask_result = await anima.ask(namespace, question, search_limit=search_limit, max_results=search_limit, llm=ask_llm, memory_types=memory_types)
                    generated_answer = ask_result.get("answer", "I don't know.")
                    retrieved = ask_result.get("memories_referenced", [])
                else:
                    generated_answer, retrieved, tokens_used, context_text = await answer_question(
                        anima, openai_client, namespace, question, llm_model, search_limit,
                        session_index=session_index, context_window=context_window,
                    )
                answer_time = time.time() - t0

                # Judge answer
                t0 = time.time()
                if is_adversarial:
                    judgment = await judge_adversarial(
                        judge_client or openai_client, question, gold_answer, generated_answer, judge_model,
                    )
                else:
                    judgment = await judge_answer(
                        judge_client or openai_client, question, gold_answer, generated_answer, judge_model,
                    )
                judge_time = time.time() - t0

                result = {
                    "sample_id": sample_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": generated_answer,
                    "category": category,
                    "category_name": CATEGORY_NAMES[category],
                    "label": judgment.get("label", "WRONG"),
                    "llm_score": float(judgment.get("score", 0.0) or 0.0),
                    "reasoning": judgment.get("reasoning", ""),
                    "answer_time_s": round(answer_time, 2),
                    "judge_time_s": round(judge_time, 2),
                    "num_retrieved": len(retrieved),
                    "tokens_used": tokens_used,
                    "context": context_text,
                }
                progress.advance(task)
                return result

        all_results = await asyncio.gather(*[
            _eval_one(ci, s, qa) for ci, s, qa in work_items
        ])
        all_results = list(all_results)

    # Save per-conversation files and send discord webhooks
    if output_dir or True:  # always group for discord
        from collections import defaultdict
        conv_groups: dict[str, list[dict]] = defaultdict(list)
        for r in all_results:
            conv_groups[r["sample_id"]].append(r)

        sample_order = {s["sample_id"]: i for i, s in enumerate(samples, 1)}
        for sample_id, conv_results in conv_groups.items():
            if output_dir and conv_results:
                conv_file = output_dir / f"{sample_id}.json"
                with open(conv_file, "w") as f:
                    json.dump(conv_results, f, indent=2)
            conv_idx = sample_order.get(sample_id, 0)
            embed = _conv_summary_embed(sample_id, conv_results, conv_idx, len(samples))
            await send_discord("", embed=embed)

    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[dict]) -> dict:
    """Print and return accuracy metrics."""
    # Per-category stats
    category_stats: dict[int, dict] = {}
    for r in results:
        cat = r["category"]
        score = float(r.get("llm_score", 0.0) or 0.0)
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0, "llm_score_sum": 0.0}
        category_stats[cat]["total"] += 1
        if score >= 0.5:
            category_stats[cat]["correct"] += 1
        category_stats[cat]["llm_score_sum"] += score

    # Overall
    total_correct = sum(s["correct"] for s in category_stats.values())
    total_questions = sum(s["total"] for s in category_stats.values())
    overall_accuracy = total_correct / total_questions if total_questions else 0
    overall_llm_score = sum(s["llm_score_sum"] for s in category_stats.values()) / total_questions if total_questions else 0

    # Display table
    table = Table(title="LoCoMo Benchmark Results")
    table.add_column("Category", style="bold")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("LLM Score", justify="right")

    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        acc = s["correct"] / s["total"] if s["total"] else 0
        avg_score = s["llm_score_sum"] / s["total"] if s["total"] else 0
        table.add_row(
            CATEGORY_NAMES.get(cat, f"Cat {cat}"),
            str(s["correct"]),
            str(s["total"]),
            f"{acc:.1%}",
            f"{avg_score:.3f}",
        )

    table.add_section()
    table.add_row(
        "[bold]Overall[/bold]",
        str(total_correct),
        str(total_questions),
        f"[bold]{overall_accuracy:.1%}[/bold]",
        f"[bold]{overall_llm_score:.3f}[/bold]",
    )

    console.print()
    console.print(table)

    # Comparison against public benchmark writeups.
    console.print()
    console.print("[dim]Reference LLM-alignment scores from public benchmark writeups:[/dim]")
    console.print(f"[dim]  Nemori:    0.811 (public README)[/dim]")
    console.print(f"[dim]  Anima:      {overall_llm_score:.3f} (thresholded acc@0.5: {overall_accuracy:.1%})[/dim]")

    # Avg response time and tokens
    avg_answer = sum(r["answer_time_s"] for r in results) / len(results) if results else 0
    avg_judge = sum(r["judge_time_s"] for r in results) / len(results) if results else 0
    avg_tokens = sum(r.get("tokens_used", 0) for r in results) / len(results) if results else 0
    console.print(f"\n[dim]Avg answer time: {avg_answer:.2f}s | Avg judge time: {avg_judge:.2f}s | Avg tokens/question: {avg_tokens:.0f}[/dim]")

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_llm_score": round(overall_llm_score, 4),
        "category_breakdown": {
            CATEGORY_NAMES[cat]: {
                "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0,
                "llm_score": round(s["llm_score_sum"] / s["total"], 4) if s["total"] else 0,
            }
            for cat, s in category_stats.items()
        },
        "total_correct": total_correct,
        "total_questions": total_questions,
        "avg_answer_time_s": round(avg_answer, 2),
        "avg_judge_time_s": round(avg_judge, 2),
        "avg_tokens_per_question": round(avg_tokens, 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for anima")
    parser.add_argument(
        "--anima-url",
        dest="anima_url",
        default="http://127.0.0.1:3000",
        help="Anima server URL",
    )
    parser.add_argument("--data-path", default="data/locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--namespace-prefix", default="benchmark/locomo", help="Namespace prefix")
    parser.add_argument("--llm-base-url", default="https://api.groq.com/openai/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--llm-model", default="llama-3.3-70b-versatile", help="LLM model for answer generation")
    parser.add_argument("--reflect-model", default=None, help="LLM model for reflection (defaults to --llm-model)")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b", help="LLM model for judging")
    parser.add_argument("--judge-base-url", default=None, help="Base URL for judge LLM (defaults to --llm-base-url)")
    parser.add_argument("--search-limit", type=int, default=50, help="Number of memories to retrieve")
    parser.add_argument("--output", default="results.json", help="Output file for detailed results")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion step")
    parser.add_argument("--no-reflect", action="store_true", help="Skip reflection pass")
    parser.add_argument("--reflect-only", action="store_true", help="Run reflection even with --skip-ingest")
    parser.add_argument("--context-window", type=int, default=2, help="Turns before/after each hit to include (0=disabled)")
    parser.add_argument("--dry-run", action="store_true", help="Test retrieval on 5 questions without LLM calls")
    parser.add_argument("--use-ask-api", action="store_true", help="Use server /api/v1/ask endpoint instead of Python pipeline")
    parser.add_argument("--conversations", help="Comma-separated sample_ids to evaluate (default: all)")
    parser.add_argument("--question-limit", type=int, default=0, help="Limit number of questions to evaluate (0=all)")
    parser.add_argument("--skip-adversarial", action="store_true", default=True, help="Skip adversarial (category 5) questions")
    parser.add_argument("--include-adversarial", action="store_true", help="Include adversarial (category 5) questions")
    parser.add_argument("--memory-types", help="Comma-separated memory types to include in /ask (e.g. event,reflected,deduced)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    conversation_ids = args.conversations.split(",") if args.conversations else None

    # Step 1: Download dataset
    download_dataset(data_path)
    dataset = load_dataset(data_path)
    console.print(f"Loaded {len(dataset)} conversations from {data_path}")

    # Check anima connectivity
    anima = AnimaClient(args.anima_url)
    if not await anima.health():
        console.print(f"[red]Cannot connect to anima at {args.anima_url}[/red]")
        console.print("[red]Start the server: cargo run --release -- config.default.toml[/red]")
        sys.exit(1)
    console.print(f"[green]Connected to anima at {args.anima_url}[/green]")

    # Build session index (needed for context expansion and reflection)
    session_index = SessionIndex()

    # Step 2: Ingest conversations
    if not args.skip_ingest:
        await ingest_all(anima, dataset, args.namespace_prefix, session_index, conversation_ids)
    else:
        console.print("[dim]Skipping ingestion (--skip-ingest)[/dim]")
        # Rebuild session index from dataset even when skipping ingestion
        _rebuild_session_index(dataset, args.namespace_prefix, session_index, conversation_ids)

    # Step 2b: Reflection pass
    openai_client = AsyncOpenAI(base_url=args.llm_base_url)
    judge_base_url = args.judge_base_url or args.llm_base_url
    judge_client = AsyncOpenAI(base_url=judge_base_url) if judge_base_url != args.llm_base_url else openai_client
    should_reflect = (not args.no_reflect and not args.skip_ingest) or args.reflect_only
    if should_reflect:
        reflect_model = args.reflect_model or args.llm_model
        await reflect_all(
            anima, openai_client, dataset, args.namespace_prefix,
            session_index, reflect_model, conversation_ids,
        )
    elif args.no_reflect:
        console.print("[dim]Skipping reflection (--no-reflect)[/dim]")

    # Step 2c: Temporal synthesis — detect state changes across sessions
    if should_reflect:
        reflect_model = args.reflect_model or args.llm_model
        await temporal_synthesis_all(
            anima, openai_client, dataset, args.namespace_prefix,
            reflect_model, conversation_ids,
        )

    # Wait for server-side background processor (reflection + deduction) to finish
    # before evaluation, so tier-2/tier-3 memories are available for retrieval.
    console.print("\n[bold]Waiting for background processor to become idle...[/bold]")
    await anima.wait_for_processor_idle()
    console.print("[green]Background processor idle — proceeding to evaluation.[/green]")

    # Dry run: test retrieval without LLM calls
    if args.dry_run:
        samples = dataset
        if conversation_ids:
            samples = [s for s in dataset if s["sample_id"] in conversation_ids]
        console.print("\n[bold]Dry run: testing retrieval on 5 questions...[/bold]\n")
        test_qs = []
        for s in samples:
            for qa in s.get("qa", []):
                if qa.get("category", 5) in (1, 2, 3, 4):
                    test_qs.append((s["sample_id"], qa))
                    if len(test_qs) >= 5:
                        break
            if len(test_qs) >= 5:
                break
        for sample_id, qa in test_qs:
            ns = f"{args.namespace_prefix}/{sample_id}"
            result = await anima.search(ns, qa["question"], limit=args.search_limit)
            hits = result.get("results", [])
            console.print(f"[bold]Q:[/bold] {qa['question']}")
            console.print(f"[dim]Gold: {qa['answer']}[/dim]")
            console.print(f"[dim]Retrieved: {len(hits)} memories[/dim]")
            if hits:
                console.print(f"[dim]Top hit (score={hits[0].get('score', 0):.3f}):[/dim]")
                console.print(f"[dim]  {hits[0].get('content', '')[:120]}[/dim]")
            else:
                console.print("[red]  No results![/red]")
            console.print()
        await anima.close()
        return

    # Step 3 & 4: Answer questions and judge
    memory_types = args.memory_types.split(",") if args.memory_types else None
    results = await evaluate(
        anima, openai_client, dataset, args.namespace_prefix,
        args.llm_model, args.judge_model, args.search_limit, conversation_ids,
        session_index=session_index, context_window=args.context_window,
        use_ask_api=args.use_ask_api, judge_client=judge_client,
        question_limit=args.question_limit, output_path=args.output,
        memory_types=memory_types, skip_adversarial=not args.include_adversarial,
    )

    # Step 5: Report
    summary = print_report(results)

    # Save summary results
    output_path = Path(args.output)
    from datetime import datetime, timezone
    config = {
        "anima_url": args.anima_url,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "judge_base_url": judge_base_url,
        "judge_model": args.judge_model,
        "search_limit": args.search_limit,
        "search_mode": "hybrid",
        "namespace_prefix": args.namespace_prefix,
    }
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "config": config,
        "results": results,
    }
    # Save combined summary to the output dir
    output_dir = output_path.parent / output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[green]Results saved to {output_dir}/[/green]")

    # Discord: final summary
    await send_discord("", embed=_final_summary_embed(summary, config))

    await anima.close()


if __name__ == "__main__":
    asyncio.run(main())
