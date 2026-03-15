#!/usr/bin/env python3
"""Compare /ask answers from two anima servers side-by-side on LoCoMo questions."""

import asyncio
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import httpx

# Config
BGE_URL = "http://127.0.0.1:3000"
QWEN_URL = "http://127.0.0.1:3001"
BGE_NS = "benchmark/locomo-bgem3-1024d-v3/conv-26"
QWEN_NS = "benchmark/locomo-qwen3emb06b-qwen32b/conv-26"
SEARCH_LIMIT = 100
LOCOMO_DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

CATEGORY_NAMES = {1: "Single-hop", 2: "Multi-hop", 3: "Temporal", 4: "Open-domain", 5: "Adversarial"}

# Judge
JUDGE_SYSTEM = """You are an impartial judge evaluating how well a generated answer matches a gold
reference answer. Score from 0.0 to 1.0 in 0.1 increments ONLY.
Allowed values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

- 1.0 = Fully correct.
- 0.7-0.9 = Mostly correct.
- 0.4-0.6 = Partially correct.
- 0.1-0.3 = Mostly wrong.
- 0.0 = Completely wrong.

Respond with a JSON object: {"reasoning": "<one sentence>", "score": <float 0.0-1.0>}
Only output the JSON object, nothing else."""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")
GROQ_KEY = os.environ.get("OPENAI_API_KEY", "")
JUDGE_MODEL = "llama-3.3-70b-versatile"

# Use llama for answering (no thinking token issues on Groq)
LLM_CONFIG = {
    "base_url": "https://api.groq.com/openai/v1",
    "model": "llama-3.3-70b-versatile",
    "api_key": None,  # filled at runtime
}

# Rate limiting for Groq
GROQ_SEMAPHORE = asyncio.Semaphore(3)  # max 3 concurrent Groq calls


async def ask_server(base_url: str, namespace: str, question: str) -> str:
    """Call /ask on a server and return the answer."""
    llm = {**LLM_CONFIG, "api_key": GROQ_KEY}
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        try:
            r = await client.post(
                "/api/v1/ask",
                headers={"X-Anima-Namespace": namespace},
                json={"question": question, "search_limit": SEARCH_LIMIT, "llm": llm},
            )
            if r.status_code == 200:
                data = r.json()
                answer = data.get("answer", "")
                # Strip think blocks
                answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL).strip()
                answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL).strip()
                return answer or "I don't know."
            else:
                return f"[ERROR {r.status_code}]"
        except Exception as e:
            return f"[ERROR: {e}]"


async def judge_answer(client: AsyncOpenAI, question: str, gold: str, answer: str) -> tuple[float, str]:
    """Judge an answer, return (score, reasoning)."""
    prompt = f"Question: {question}\nGold answer: {gold}\nGenerated answer: {answer}"
    async with GROQ_SEMAPHORE:
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0.0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
            m = re.search(r'"score"\s*:\s*([\d.]+)', raw)
            if m:
                score = float(m.group(1))
                rm = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
                reasoning = rm.group(1) if rm else ""
                return score, reasoning
        except Exception as e:
            print(f"  [judge error: {e}]", file=sys.stderr)
            await asyncio.sleep(2)  # backoff on rate limit
    return 0.0, "judge_error"


async def main():
    # Load dataset
    data_path = Path(__file__).parent / "data" / "locomo10.json"
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LoCoMo dataset to {data_path}")
        urllib.request.urlretrieve(LOCOMO_DATA_URL, str(data_path))
    with open(data_path) as f:
        conversations = json.load(f)

    # Find conv-26
    conv = None
    for c in conversations:
        if c.get("sample_id") == "conv-26":
            conv = c
            break
    if not conv:
        print("conv-26 not found!")
        return

    all_questions = conv.get("qa", [])
    # Filter to categories 1-4 (skip adversarial=5), matching run.py
    questions = [q for q in all_questions if q.get("category", 5) in (1, 2, 3, 4)]
    print(f"Found {len(questions)} questions in conv-26 (filtered from {len(all_questions)})\n")

    judge = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)

    results = []
    cat_stats = {}  # category -> {bge_correct, qwen_correct, total}

    for i, q in enumerate(questions):
        qtext = q["question"]
        gold = str(q["answer"])
        cat = q.get("category", 0)
        cat_name = CATEGORY_NAMES.get(cat, f"Cat-{cat}")

        # Ask both servers concurrently
        bge_ans, qwen_ans = await asyncio.gather(
            ask_server(BGE_URL, BGE_NS, qtext),
            ask_server(QWEN_URL, QWEN_NS, qtext),
        )

        # Judge both concurrently
        (bge_score, bge_reason), (qwen_score, qwen_reason) = await asyncio.gather(
            judge_answer(judge, qtext, gold, bge_ans),
            judge_answer(judge, qtext, gold, qwen_ans),
        )

        bge_ok = "OK" if bge_score >= 0.5 else "WRONG"
        qwen_ok = "OK" if qwen_score >= 0.5 else "WRONG"

        # Track per-category
        if cat_name not in cat_stats:
            cat_stats[cat_name] = {"bge_correct": 0, "qwen_correct": 0, "total": 0}
        cat_stats[cat_name]["total"] += 1
        if bge_score >= 0.5:
            cat_stats[cat_name]["bge_correct"] += 1
        if qwen_score >= 0.5:
            cat_stats[cat_name]["qwen_correct"] += 1

        # Print all results
        marker = ""
        if bge_ok != qwen_ok:
            marker = " <<<< DISAGREE"

        print(f"[{i+1:3d}/{len(questions)}] {cat_name:12s} | BGE={bge_score:.1f} {bge_ok:5s} | Qwen3={qwen_score:.1f} {qwen_ok:5s}{marker}")
        if marker:
            print(f"     Q: {qtext[:80]}")
            print(f"     Gold: {gold[:80]}")
            print(f"     BGE:  {bge_ans[:80]}")
            print(f"     Qwen: {qwen_ans[:80]}")
            print()

        results.append({
            "idx": i,
            "category": cat_name,
            "question": qtext,
            "gold": gold,
            "bge_answer": bge_ans,
            "qwen_answer": qwen_ans,
            "bge_score": bge_score,
            "qwen_score": qwen_score,
        })

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Category':15s} | {'BGE-M3 1024d':>14s} | {'Qwen3-Emb 512d':>14s} | {'Delta':>8s}")
    print("-" * 70)
    bge_total_ok = 0
    qwen_total_ok = 0
    total = 0
    for cat_name in ["Single-hop", "Multi-hop", "Temporal", "Open-domain"]:
        s = cat_stats.get(cat_name, {"bge_correct": 0, "qwen_correct": 0, "total": 0})
        bge_pct = 100 * s["bge_correct"] / s["total"] if s["total"] else 0
        qwen_pct = 100 * s["qwen_correct"] / s["total"] if s["total"] else 0
        delta = qwen_pct - bge_pct
        print(f"{cat_name:15s} | {s['bge_correct']:3d}/{s['total']:3d} ({bge_pct:5.1f}%) | {s['qwen_correct']:3d}/{s['total']:3d} ({qwen_pct:5.1f}%) | {delta:+6.1f}%")
        bge_total_ok += s["bge_correct"]
        qwen_total_ok += s["qwen_correct"]
        total += s["total"]
    print("-" * 70)
    bge_pct = 100 * bge_total_ok / total if total else 0
    qwen_pct = 100 * qwen_total_ok / total if total else 0
    delta = qwen_pct - bge_pct
    print(f"{'Overall':15s} | {bge_total_ok:3d}/{total:3d} ({bge_pct:5.1f}%) | {qwen_total_ok:3d}/{total:3d} ({qwen_pct:5.1f}%) | {delta:+6.1f}%")

    # Save
    out_path = Path(__file__).parent / "results" / "compare_bge_vs_qwen3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
