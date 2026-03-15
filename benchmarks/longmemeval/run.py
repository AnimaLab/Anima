#!/usr/bin/env python3
"""LongMemEval benchmark evaluation harness for Anima.

Evaluates Anima's memory system against the LongMemEval benchmark (ICLR 2025):
  1. Loads the LongMemEval-S dataset (500 questions, ~40 sessions each)
  2. Ingests chat sessions into a running Anima server
  3. Retrieves memories and generates answers via LLM
  4. Scores answers with an LLM judge (binary yes/no per question type)
  5. Reports per-category and overall accuracy

Usage:
    python run.py [--anima-url URL] [--llm-model MODEL] [--judge-model MODEL]

Requires: .env file with OPENAI_API_KEY or set env vars directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import httpx
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# ---------------------------------------------------------------------------
# Retry helper
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

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
)

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

# LongMemEval uses type-specific judge prompts (from the paper)
JUDGE_PROMPTS = {
    "single-session-user": (
        "You are evaluating whether a chat assistant correctly recalled information from past conversations.\n"
        "The question asks about something the USER said in a past session.\n"
        "Reference answer: {answer}\n"
        "Model response: {hypothesis}\n\n"
        "Please answer yes if the response contains the correct answer or equivalent information. "
        "Otherwise, answer no. Only output yes or no."
    ),
    "single-session-assistant": (
        "You are evaluating whether a chat assistant correctly recalled information from past conversations.\n"
        "The question asks about something the ASSISTANT said or recommended in a past session.\n"
        "Reference answer: {answer}\n"
        "Model response: {hypothesis}\n\n"
        "Please answer yes if the response contains the correct answer or equivalent information. "
        "Otherwise, answer no. Only output yes or no."
    ),
    "single-session-preference": (
        "You are evaluating whether a chat assistant correctly recalled the user's personal preference.\n"
        "Reference answer: {answer}\n"
        "Model response: {hypothesis}\n\n"
        "Please answer yes if the response correctly recalls and utilizes the user's personal information. "
        "Otherwise, answer no. Only output yes or no."
    ),
    "multi-session": (
        "You are evaluating whether a chat assistant correctly synthesized information across multiple sessions.\n"
        "Reference answer: {answer}\n"
        "Model response: {hypothesis}\n\n"
        "Please answer yes if the response contains the correct answer or equivalent information. "
        "Otherwise, answer no. Only output yes or no."
    ),
    "temporal-reasoning": (
        "You are evaluating whether a chat assistant correctly performed temporal reasoning over past sessions.\n"
        "Reference answer: {answer}\n"
        "Model response: {hypothesis}\n\n"
        "Please answer yes if the response contains the correct answer. "
        "Do not penalize off-by-one errors for the number of days. "
        "Otherwise, answer no. Only output yes or no."
    ),
    "knowledge-update": (
        "You are evaluating whether a chat assistant correctly handled a knowledge update.\n"
        "Reference answer: {answer}\n"
        "Model response: {hypothesis}\n\n"
        "Please answer yes if the response contains the correct updated answer. "
        "It is acceptable if the response also mentions the prior outdated information, "
        "as long as the current answer is correct. "
        "Otherwise, answer no. Only output yes or no."
    ),
}

ANSWER_SYSTEM_PROMPT = """\
You are a helpful chat assistant with memory of past conversations.
You will be given retrieved facts from conversation history, then a question.

Important rules:
- Answer based ONLY on the provided facts. Do not make up information.
- If no relevant fact is present, say "I don't have that information from our past conversations."
- Be concise and direct.
- For temporal questions, compute dates carefully."""

REFLECTION_PROMPT = """\
You are analyzing a chat session between a user and an assistant.
Extract ALL key facts, decisions, preferences, recommendations, and information discussed.

Rules:
- Write one concise factual statement per line
- Include specific names, dates, places, numbers, and details
- Capture what the user said AND what the assistant recommended/said
- Include user preferences, opinions, and personal information
- Do NOT add information not present in the conversation

Output ONLY the list of facts, one per line. No numbering, no bullets, no headers."""

console = Console()

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


async def send_discord(content: str = "", embed: dict | None = None) -> None:
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


def _progress_embed(done: int, total: int, correct: int, result: dict) -> dict:
    """Build a Discord embed for a completed question."""
    qtype = result["question_type"]
    label = "correct" if result["correct"] else "wrong"
    color = 0x2ECC71 if result["correct"] else 0xE74C3C
    return {
        "title": f"LongMemEval [{done}/{total}] — {label}",
        "description": (
            f"**Q:** {result['question'][:200]}\n"
            f"**Expected:** {result['answer'][:200]}\n"
            f"**Got:** {result['hypothesis'][:200]}"
        ),
        "color": color,
        "fields": [
            {"name": "Type", "value": qtype, "inline": True},
            {"name": "Running accuracy", "value": f"{correct}/{done} ({correct/done:.1%})", "inline": True},
        ],
    }


def _final_summary_embed(summary: dict, config: dict) -> dict:
    """Build a Discord embed for the final benchmark summary."""
    fields = []
    for qtype, stats in summary.get("type_breakdown", {}).items():
        fields.append({
            "name": qtype,
            "value": f"{stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})",
            "inline": True,
        })

    acc = summary["overall_accuracy"]
    color = 0x2ECC71 if acc >= 0.7 else 0xF39C12 if acc >= 0.5 else 0xE74C3C
    return {
        "title": "LongMemEval Benchmark Complete",
        "description": (
            f"**Overall: {summary['total_correct']}/{summary['total_questions']}** "
            f"({acc:.1%})\n"
            f"Model: `{config.get('llm_model', '?')}`"
        ),
        "color": color,
        "fields": fields,
    }


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

    async def add_memories_batch(self, namespace: str, items: list[dict], reflect: bool = False) -> dict:
        r = await self.http.post(
            "/api/v1/memories/batch",
            headers={"X-Anima-Namespace": namespace},
            json={"items": items, "reflect": reflect},
        )
        r.raise_for_status()
        return r.json()

    async def add_memory(self, namespace: str, content: str, metadata: dict | None = None,
                         tags: list[str] | None = None, memory_type: str = "event") -> dict:
        r = await self.http.post(
            "/api/v1/memories",
            headers={"X-Anima-Namespace": namespace},
            json={"content": content, "metadata": metadata or {}, "consolidate": False,
                  "tags": tags or [], "memory_type": memory_type},
        )
        r.raise_for_status()
        return r.json()

    async def search(self, namespace: str, query: str, limit: int = 20, search_mode: str = "hybrid") -> dict:
        r = await self.http.post(
            "/api/v1/memories/search",
            headers={"X-Anima-Namespace": namespace},
            json={"query": query, "limit": limit, "search_mode": search_mode},
        )
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self.http.aclose()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def download_dataset(data_path: Path) -> None:
    """Download longmemeval_s.json if not present."""
    if data_path.exists():
        console.print(f"[dim]Dataset already exists at {data_path}[/dim]")
        return
    import urllib.request
    console.print(f"Downloading LongMemEval-S dataset to {data_path} ...")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATASET_URL, str(data_path))
    console.print("[green]Download complete.[/green]")


def load_dataset(data_path: Path) -> list[dict]:
    with open(data_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ingestion — each question has its own haystack of ~40-50 sessions
# ---------------------------------------------------------------------------


async def ingest_question(
    anima: AnimaClient,
    entry: dict,
    namespace: str,
    reflect: bool,
    openai_client: AsyncOpenAI | None,
    reflect_model: str | None,
    progress: Progress,
) -> int:
    """Ingest all haystack sessions for a single question. Returns turn count."""
    sessions = entry["haystack_sessions"]
    dates = entry.get("haystack_dates", [])
    session_ids = entry.get("haystack_session_ids", [])

    total_turns = sum(len(s) for s in sessions)
    task = progress.add_task(f"  Q:{entry['question_id']}", total=total_turns)

    count = 0
    batch_items: list[dict] = []

    for idx, session in enumerate(sessions):
        date = dates[idx] if idx < len(dates) else ""
        sid = session_ids[idx] if idx < len(session_ids) else f"session_{idx}"

        session_text_parts = []
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if not content:
                progress.advance(task)
                continue

            text = f"[{date}] {role}: {content}" if date else f"{role}: {content}"
            session_text_parts.append(text)

            batch_items.append({
                "content": text,
                "metadata": {"session_id": sid, "timestamp": date, "role": role},
                "tags": ["longmemeval", sid],
            })
            if len(batch_items) >= 64:
                try:
                    await anima.add_memories_batch(namespace, batch_items, reflect=False)
                except Exception:
                    for item in batch_items:
                        await anima.add_memory(namespace, item["content"], item.get("metadata"), item.get("tags"))
                batch_items.clear()
            count += 1
            progress.advance(task)

        # Reflect on this session if enabled
        if reflect and openai_client and reflect_model and session_text_parts:
            session_text = "\n".join(session_text_parts)
            try:
                response = await openai_client.chat.completions.create(
                    model=reflect_model,
                    temperature=0.1,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": REFLECTION_PROMPT},
                        {"role": "user", "content": f"Session timestamp: {date}\n\n{session_text}"},
                    ],
                )
                facts = response.choices[0].message.content.strip()
                facts = re.sub(r"<think>.*?</think>\s*", "", facts, flags=re.DOTALL).strip()
                facts = re.sub(r"<think>.*", "", facts, flags=re.DOTALL).strip()
                for line in facts.split("\n"):
                    line = line.strip().lstrip("•-*0123456789. ")
                    if not line or len(line) < 10:
                        continue
                    fact_content = f"[{date}] {line}" if date else line
                    batch_items.append({
                        "content": fact_content,
                        "metadata": {"session_id": sid, "timestamp": date},
                        "tags": ["longmemeval", sid, "reflection"],
                        "memory_type": "reflection",
                    })
            except Exception as e:
                console.print(f"[yellow]Reflection failed for session {sid}: {e}[/yellow]")

    if batch_items:
        try:
            await anima.add_memories_batch(namespace, batch_items, reflect=False)
        except Exception:
            for item in batch_items:
                await anima.add_memory(namespace, item["content"], item.get("metadata"), item.get("tags"))

    return count


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

QUERY_EXPAND_PROMPT = """\
Given a question about past chat conversations, generate 3 short keyword search queries (2-4 words each).
Think about what specific words the user or assistant would have actually said.

For each query, try a DIFFERENT angle:
1. The direct topic keywords
2. Related/synonym terms
3. The answer/outcome keywords

Output ONLY the 3 queries, one per line. No numbering, no explanation."""


async def expand_query(openai_client: AsyncOpenAI, question: str, llm_model: str) -> list[str]:
    try:
        response = await openai_client.chat.completions.create(
            model=llm_model,
            temperature=0.3,
            max_tokens=100,
            messages=[
                {"role": "system", "content": QUERY_EXPAND_PROMPT},
                {"role": "user", "content": question + " /no_think"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
        queries = [q.strip().lstrip("0123456789.-) ") for q in raw.split("\n") if q.strip()]
        return queries[:3]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


async def answer_question(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    namespace: str,
    question: str,
    llm_model: str,
    search_limit: int,
) -> tuple[str, int]:
    """Retrieve context and generate answer. Returns (answer, tokens_used)."""
    queries = [question]
    extra = await expand_query(openai_client, question, llm_model)
    queries.extend(extra)

    seen: set[str] = set()
    results: list[dict] = []
    for q in queries:
        for mode in ("hybrid", "keyword"):
            lim = search_limit if mode == "hybrid" else search_limit // 2
            search_result = await anima.search(namespace, q, limit=lim, search_mode=mode)
            for r in search_result.get("results", []):
                content = r.get("content", "")
                if content not in seen:
                    seen.add(content)
                    results.append(r)

    if not results:
        return "I don't have that information from our past conversations.", 0

    context_parts = [f"{i}. {r.get('content', '')}" for i, r in enumerate(results[:100], 1)]
    raw_context = "\n".join(context_parts)

    prompt = f"""Retrieved conversation excerpts:
{raw_context}

Question: {question}

Answer concisely: /no_think"""

    tokens_used = 0
    response = await retry_api_call(lambda: openai_client.chat.completions.create(
        model=llm_model,
        temperature=0.1,
        max_tokens=512,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    ))
    tokens_used += response.usage.total_tokens if response.usage else 0
    answer = response.choices[0].message.content.strip()
    answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL).strip()
    answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL).strip()
    return answer, tokens_used


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


async def judge_answer(
    judge_client: AsyncOpenAI,
    entry: dict,
    hypothesis: str,
    judge_model: str,
) -> bool:
    """Judge if hypothesis is correct using type-specific prompt. Returns True/False."""
    qtype = entry["question_type"]
    prompt_template = JUDGE_PROMPTS.get(qtype, JUDGE_PROMPTS["single-session-user"])
    prompt = prompt_template.format(answer=entry["answer"], hypothesis=hypothesis)

    response = await retry_api_call(lambda: judge_client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    ))

    raw = response.choices[0].message.content.strip().lower()
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    return "yes" in raw


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate(
    anima: AnimaClient,
    openai_client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    dataset: list[dict],
    config: dict,
) -> dict:
    """Run full evaluation: ingest → answer → judge for each question."""
    namespace_prefix = config["namespace_prefix"]
    llm_model = config["llm_model"]
    judge_model = config["judge_model"]
    search_limit = config["search_limit"]
    reflect = config.get("reflect", False)
    reflect_model = config.get("reflect_model")
    output_path = Path(config["output"])

    # Create output directory: results/<run-name>/
    output_dir = output_path.parent / output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    total_correct = 0

    console.print(f"\n[bold]Evaluating {len(dataset)} questions...[/bold]\n")

    for i, entry in enumerate(dataset):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        namespace = f"{namespace_prefix}/{qid}"

        console.print(f"\n[bold cyan]Question {i+1}/{len(dataset)}[/bold cyan] [{qtype}] {question[:80]}...")

        # 1. Ingest haystack
        t0 = time.time()
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                       BarColumn(), TaskProgressColumn(), console=console) as progress:
            turns = await ingest_question(
                anima, entry, namespace, reflect,
                openai_client if reflect else None,
                reflect_model, progress,
            )
        ingest_time = time.time() - t0
        console.print(f"  Ingested {turns} turns in {ingest_time:.1f}s")

        # Wait for processor
        await anima.wait_for_processor_idle()

        # 2. Answer
        t0 = time.time()
        hypothesis, tokens = await answer_question(
            anima, openai_client, namespace, question, llm_model, search_limit,
        )
        answer_time = time.time() - t0
        console.print(f"  Answer ({answer_time:.1f}s): {hypothesis[:120]}")

        # 3. Judge
        t0 = time.time()
        correct = await judge_answer(judge_client, entry, hypothesis, judge_model)
        judge_time = time.time() - t0

        if correct:
            total_correct += 1
        label = "[green]CORRECT[/green]" if correct else "[red]WRONG[/red]"
        console.print(f"  Judge: {label}  (expected: {entry['answer'][:120]})")

        result = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer": entry["answer"],
            "hypothesis": hypothesis,
            "correct": correct,
            "ingest_time_s": round(ingest_time, 2),
            "answer_time_s": round(answer_time, 2),
            "judge_time_s": round(judge_time, 2),
            "tokens_used": tokens,
        }
        all_results.append(result)

        # Save per-question result
        q_file = output_dir / f"{qid}.json"
        with open(q_file, "w") as f:
            json.dump(result, f, indent=2)

        # Discord notification
        await send_discord(embed=_progress_embed(i + 1, len(dataset), total_correct, result))

    # Build summary
    type_breakdown: dict[str, dict] = {}
    for r in all_results:
        qt = r["question_type"]
        type_breakdown.setdefault(qt, {"correct": 0, "total": 0})
        type_breakdown[qt]["total"] += 1
        if r["correct"]:
            type_breakdown[qt]["correct"] += 1
    for stats in type_breakdown.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0

    summary = {
        "benchmark": "LongMemEval-S",
        "total_questions": len(all_results),
        "total_correct": total_correct,
        "overall_accuracy": total_correct / len(all_results) if all_results else 0,
        "type_breakdown": type_breakdown,
        "config": config,
        "avg_answer_time_s": sum(r["answer_time_s"] for r in all_results) / len(all_results) if all_results else 0,
        "avg_judge_time_s": sum(r["judge_time_s"] for r in all_results) / len(all_results) if all_results else 0,
        "results": all_results,
    }

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"\n[green]Summary saved to {summary_file}[/green]")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_results(summary: dict) -> None:
    table = Table(title="LongMemEval Results")
    table.add_column("Question Type", style="cyan")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accuracy", justify="right")

    for qtype, stats in sorted(summary.get("type_breakdown", {}).items()):
        acc = stats["accuracy"]
        style = "green" if acc >= 0.7 else "yellow" if acc >= 0.5 else "red"
        table.add_row(qtype, str(stats["correct"]), str(stats["total"]), f"[{style}]{acc:.1%}[/{style}]")

    overall = summary["overall_accuracy"]
    style = "green" if overall >= 0.7 else "yellow" if overall >= 0.5 else "red"
    table.add_row("OVERALL", str(summary["total_correct"]), str(summary["total_questions"]),
                   f"[bold {style}]{overall:.1%}[/bold {style}]", style="bold")

    console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for Anima")
    parser.add_argument("--anima-url", default="http://localhost:3000")
    parser.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--reflect", action="store_true", help="Enable reflection pass on sessions")
    parser.add_argument("--reflect-model", default=None, help="Model for reflection (defaults to --llm-model)")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--judge-base-url", default=None, help="Base URL for judge model (defaults to --llm-base-url)")
    parser.add_argument("--search-limit", type=int, default=50)
    parser.add_argument("--namespace-prefix", default="longmemeval")
    parser.add_argument("--output", default="results/longmemeval.json")
    parser.add_argument("--data-path", default="data/longmemeval_s.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    parser.add_argument("--skip-types", nargs="*", default=[], help="Question types to skip")
    args = parser.parse_args()

    config = {
        "llm_model": args.llm_model,
        "reflect_model": args.reflect_model or args.llm_model,
        "judge_model": args.judge_model,
        "search_limit": args.search_limit,
        "namespace_prefix": args.namespace_prefix,
        "output": args.output,
        "reflect": args.reflect,
    }

    # Download and load dataset
    data_path = Path(args.data_path)
    download_dataset(data_path)
    dataset = load_dataset(data_path)
    console.print(f"Loaded {len(dataset)} questions from {data_path}")

    # Filter by type if requested
    if args.skip_types:
        dataset = [e for e in dataset if e["question_type"] not in args.skip_types]
        console.print(f"After filtering: {len(dataset)} questions")

    # Limit if requested
    if args.limit > 0:
        dataset = dataset[:args.limit]
        console.print(f"Limited to {len(dataset)} questions")

    # Init clients
    anima = AnimaClient(args.anima_url)
    if not await anima.health():
        console.print("[red]Anima server not reachable![/red]")
        sys.exit(1)
    console.print(f"[green]Anima server OK at {args.anima_url}[/green]")

    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    openai_client = AsyncOpenAI(api_key=api_key, base_url=args.llm_base_url)
    judge_base_url = args.judge_base_url or args.llm_base_url
    judge_client = AsyncOpenAI(api_key=api_key, base_url=judge_base_url)

    summary = await evaluate(anima, openai_client, judge_client, dataset, config)
    print_results(summary)

    # Final Discord notification
    await send_discord(embed=_final_summary_embed(summary, config))

    await anima.close()


if __name__ == "__main__":
    asyncio.run(main())
