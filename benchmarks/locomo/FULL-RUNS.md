# LoCoMo Full-Run Summaries

This page tracks notable LoCoMo runs across all 10 conversations
(1,540 questions). It complements [`REPORT.md`](REPORT.md), which remains the
main curated public narrative for the March 3 benchmark configuration.

## Highest Broad Accuracy Run

This full-dataset run reached the highest broad LoCoMo accuracy currently
published in the Anima benchmark materials.

**Date**: 2026-03-09  
**Pipeline**: raw + reflected retrieval  
**Answer model**: `Qwen/Qwen3.5-27B`  
**Judge model**: `Qwen/Qwen3.5-27B`  
**Search limit**: `50`  
**Search mode**: `hybrid`

| Metric | Value |
| --- | --- |
| Total correct | `1351 / 1540` |
| Overall accuracy | **87.73%** |
| Overall LLM judge score | **0.8412** |
| Avg tokens / question | `~5,958` |
| Avg answer time | `160.66s` |
| Avg judge time | `23.32s` |

### Category Breakdown

| Category | Accuracy | LLM Score |
| --- | --- | --- |
| Single-hop | `86.88%` | `0.7837` |
| Multi-hop | `84.11%` | `0.8259` |
| Temporal | `54.17%` | `0.5333` |
| Open-domain | `93.22%` | `0.9015` |

## Comparison With The Curated Report Run

[`REPORT.md`](REPORT.md) documents a different all-10-conversation run:

- `86.0%` overall accuracy
- `0.864` LLM judge score
- `qwen/qwen3-32b` as answer model
- `llama-3.3-70b-versatile` as judge
- `search-limit = 100`

The run on this page achieved higher binary accuracy, but under a different
model/judge/search-limit combination. Treat `87.73%` as the current best broad
accuracy figure, and `0.864` as the best published broad judge score.
