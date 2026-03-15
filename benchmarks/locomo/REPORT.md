# Anima LoCoMo Benchmark Report

This is a curated public summary of benchmark behavior. Raw experiment outputs
and per-run artifact dumps are intentionally kept outside the public repo.

**Date**: 2026-03-03
**Pipeline**: Extract (multi-query → retrieve → extract facts → synthesize answer)
**Configuration**: qwen/qwen3-32b (LLM) · llama-3.3-70b-versatile (judge, 0–1 score) · search-limit 100 · BGE-M3 embeddings

---

## Results — All 10 Conversations (1,540 questions)

| Category     | Correct | Total | Accuracy | LLM Score |
|--------------|---------|-------|----------|-----------|
| Single-hop   | 230     | 282   | 81%      | 0.816     |
| Multi-hop    | 275     | 321   | 85%      | 0.857     |
| Temporal     | 68      | 96    | **70%**  | 0.708     |
| Open-domain  | 757     | 841   | 90%      | 0.900     |
| **Overall**  | **1330**| **1540**| **86%** | **0.864** |

**Avg tokens/question**: ~6,300 (extract + answer calls, 3 convs instrumented)

> A separate all-10-conversation run later reached **87.73%** (`1351 / 1540`)
> with `Qwen/Qwen3.5-27B` as both answer model and judge at
> `search-limit = 50`. See [`FULL-RUNS.md`](FULL-RUNS.md). The table above
> remains the curated reference run for this report's own configuration.

### Per-conversation breakdown

| Conv       | Single | Multi | Temporal | Open | Accuracy | LLM Score |  N  |
|------------|--------|-------|----------|------|----------|-----------|-----|
| conv-26    | 87%    | 97%   | 84%      | 88%  | 90%      | 0.901     | 152 |
| conv-30    | 72%    | 88%   | —        | 79%  | 81%      | 0.815     |  81 |
| conv-41    | 87%    | 88%   | 62%      | 93%  | 89%      | 0.895     | 152 |
| conv-42    | 70%    | 85%   | 54%      | 90%  | 83%      | 0.792*    | 199 |
| conv-43    | 77%    | 84%   | 71%      | 88%  | 84%      | 0.848     | 178 |
| conv-44    | 83%    | 70%   | 42%      | 91%  | 82%      | 0.829     | 123 |
| conv-47    | 85%    | 79%   | 61%      | 97%  | 88%      | 0.887     | 150 |
| conv-48    | 80%    | 92%   | 60%      | 88%  | 86%      | 0.835*    | 191 |
| conv-49    | 89%    | 75%   | 100%     | 87%  | 86%      | 0.818*    | 156 |
| conv-50    | 78%    | 87%   | 85%      | 89%  | 86%      | 0.867     | 158 |

\* LLM score from 0–1 judge; convs without * used binary label (score = 0 or 1)

### Reference comparison (binary accuracy)

| System      | Overall | Single-hop | Multi-hop | Temporal | Open-domain |
|-------------|---------|------------|-----------|----------|-------------|
| Anima       | **86%** | 81%        | 85%       | 70%      | 90%         |
| Honcho      | 89.9%   | —          | —         | —        | —           |
| Backboard   | 90.0%   | —          | —         | —        | —           |

> Nemori is omitted from the binary-accuracy table above because its current
> public README reports LoCoMo as `BLEU 0.4487`, `F1 0.5196`, and
> `LLM alignment 0.8110`, not a directly comparable binary-accuracy percentage.
> Honcho's `89.9%` comes from its public site, and Backboard's `90.0%` comes
> from its public LoCoMo benchmark repository.

### LLM Score comparison (continuous 0–1)

| System      | LLM Score | Judge model  |
|-------------|-----------|--------------|
| Anima       | **0.864** | llama-3.3-70b|
| Nemori      | 0.811     | not specified in README |

> Note: judge models differ, and Nemori's public README reports `LLM alignment`
> rather than a binary-accuracy figure, so this comparison should be treated as
> directional rather than strictly apples-to-apples.

---

## System Configuration

### Nemori-inspired features (implemented 2026-03)

1. **Predict-calibrate consolidation** — stores only novel claims not already predicted by existing memories. Fixed to protect direct identity labels and specific named entities (no_change threshold tightened).
2. **Semantic batching in processor** — memories clustered by topic coherence (cosine ≥ 0.60, max 10) before LLM reflection instead of arbitrary chunks of 10.
3. **Tiered retrieval weighting** — tier-2 (reflected) +0.025, tier-3 (deduced) +0.050 score bonus over raw tier-1.
4. **Date metadata indexing** — `event_date` ISO 8601 extracted during reflection, stored in metadata for temporal filtering.

### Search config used for historical benchmark runs

```toml
[search]
rrf_k              = 60.0
weight_vector      = 0.6
weight_keyword     = 0.4
temporal_weight    = 0.2
temporal_lambda    = 0.001
similarity_threshold = 0.85
min_vector_similarity = 0.55
tier_boost         = 0.05
```

### Token efficiency vs search-limit

| search-limit | Tokens/Q | Accuracy (conv-26) | LLM Score |
|---|---|---|---|
| 50  | ~3,842 | 84.9% | 0.819 |
| 100 | ~6,293 | 90.1% | 0.901 |

Search-limit 100 is used for all benchmark runs (5pp accuracy gain worth the 40% token cost).

---

## Temporal Failure Analysis (28 wrong / 96 total = 30% error rate)

### Failure taxonomy

| Type | Count | Examples |
|------|-------|---------|
| **Missing retrieval** — fact stored but not surfaced | ~8 | Indiana visit, Florida visit, Alaska internship state, Under Armour endorsement |
| **Inference/external knowledge** — answer requires reasoning beyond stored facts | ~9 | Hatha Yoga, Voyageurs NP, Minnesota state, Good Sports charity, Star Wars Ireland locations |
| **Hedging failure** — LLM says "I don't know" when evidence supports inference | ~6 | Jolene's age ("in school" → ≤30), James in CT, Dave's shop size, Deborah's friend status |
| **Contradiction** — LLM contradicts correctly stored fact | ~5 | James lonely before Samantha, John & James studied together, Mafia vs Among Us |

### Root causes

**1. External knowledge gap (~9 failures)**
Category 3 "temporal" questions in LoCoMo include inference questions whose gold answers require external world knowledge (e.g. *"What yoga type builds core strength?"* → Hatha Yoga; *"Which national park near Minnesota?"* → Voyageurs). Anima has no query-time web retrieval in the evaluation pipeline, so these questions are essentially unanswerable from stored memories alone.

**2. Hedging LLM (~6 failures)**
qwen3-32b over-hedges when evidence is indirect. Gold answers like *"Likely yes"* or *"Probably no more than 30"* require the LLM to commit to a probabilistic inference. The extract prompt should explicitly instruct the model to reason from available evidence and commit to a best estimate rather than defaulting to "I don't know."

**3. Retrieval miss (~8 failures)**
State/location facts (Indiana, Florida, Alaska) are stored but not retrieved because the question ("What state did Nate visit?") doesn't lexically match the memory content ("Nate mentioned visiting Miami"). Hybrid search favors semantic similarity but location names are under-represented at consolidation time. `event_date` metadata helps for dates but not geography.

**4. Hallucination/contradiction (~5 failures)**
Some answers directly contradict stored facts (Among Us vs Mafia; James was not lonely). These are LLM reasoning errors during synthesis, not retrieval failures.

### Improvement directions

| Direction | Expected gain | Effort |
|-----------|--------------|--------|
| Extract prompt: instruct model to commit to inference, avoid "I don't know" | ~4–6 failures → 0.5 score | Low |
| Add location/geography extraction to reflection (like `event_date` for states/places) | ~5–8 failures | Medium |
| Web search at query time for external-knowledge questions | ~6–9 failures | Medium |
| Re-ranker pass to surface exact-match location facts buried in RRF ranking | ~3–5 failures | Medium |
