# Anima Roadmap PRD

Retrieval engine upgrades (borrowed from Qdrant's architecture, adapted for SQLite) and data safety features.

---

## 1. Pre-filtering (instead of post-filtering)

**Problem:** Vector search is global — `search_vectors()` scans all embeddings, then each result is checked one-by-one against `namespace LIKE ? AND status = 'active'` (store.rs:1985-1988). With many namespaces, most candidates are discarded after being retrieved.

**Qdrant's approach:** Payload-based filtering happens *before* or *during* ANN search, so only matching vectors are ever considered.

**How to adopt (pick one):**

- **Per-namespace vec0 tables.** Partition vectors by namespace at write time. Search only queries the relevant table. Pros: clean separation, fast. Cons: cross-namespace search requires multi-table union.
- **Pre-filter IDs via SQL, then vector search a subset.** Query `SELECT id FROM memories WHERE namespace LIKE ? AND status = 'active'`, then search vectors only within that ID set. Requires sqlite-vec support for ID-constrained search or a temp table join.
- **Joined query.** If sqlite-vec supports WHERE clauses on the virtual table join, filter inline. Needs benchmarking.

**Impact:** High — eliminates wasted vector comparisons, especially with many namespaces.
**Effort:** Medium — requires changes to `vector.rs` and `store.rs` search paths.

---

## 2. Sparse Vectors (hybrid at the vector level)

**Problem:** Hybrid search fuses two separate systems — sqlite-vec for dense cosine similarity + FTS5 for BM25 keyword matching. The fusion is a weighted sum after the fact. FTS5 uses basic Porter stemming, which misses semantic keyword importance.

**Qdrant's approach:** Store a sparse vector (SPLADE or BM42) alongside the dense vector. Both are searched in vector space, enabling learned keyword-semantic matching instead of raw term frequency.

**How to adopt:**

- Generate sparse vectors from memory content using a learned sparse model (SPLADE, BGE-M3 sparse, or similar).
- Store as a second embedding column or a separate sparse vec0 table.
- Fuse dense + sparse scores at the vector level before combining with temporal decay.
- Could eventually replace FTS5, or complement it for queries where BM25 and sparse vectors disagree.

**Impact:** High — better keyword-semantic matching, especially for paraphrased queries.
**Effort:** High — requires a sparse embedding model, new storage schema, and fusion logic changes.

---

## 3. Scalar Quantization (f32 to int8)

**Problem:** Full f32 embeddings stored as BLOBs. 512-dim = 2,048 bytes per memory. Index size grows linearly.

**Qdrant's approach:** Scalar quantization (f32 to int8, 4x smaller), binary quantization (f32 to 1-bit, 32x smaller). Search uses quantized vectors with oversampling, then rescores top candidates against full-precision vectors.

**How to adopt:**

- Create a second vec0 table (`vec_memories_q`) with int8 quantized vectors for fast ANN search. The existing `vec_memories` table uses `float[512]` and cannot store int8 directly. Verify sqlite-vec supports `int8[N]` column type — if not, store quantized vectors as raw BLOBs and compute distances manually.
- Keep full f32 vectors in `memories.embedding` (already stored there) and in `vec_memories` for rescoring.
- Oversample: fetch 3-5x candidates from quantized index, rescore against f32 originals.
- The cross-encoder reranker already runs after retrieval, so this adds a cheap pre-filter stage.

**Impact:** Medium — 4x index size reduction with minimal recall loss. More relevant at 10k+ memories.
**Effort:** Low — quantization is simple math (min-max scaling to int8), vec0 table recreation.

---

## 4. Multi-vector (named vectors per memory)

**Problem:** One embedding per memory. A long memory and its short summary embed very differently — factual queries match summaries better, contextual queries match full content better.

**Qdrant's approach:** Multiple named vectors per point (e.g., `content`, `summary`, `title`). Search targets the vector most relevant to the query type.

**How to adopt:**

- Embed both the raw content and a reflected/summarized version (already produced by the consolidation pipeline).
- Store summary embeddings in a second vec0 table (`vec_memories_summary`).
- Short factual queries (detected by length/intent) search against summary vectors.
- Long contextual queries search against content vectors.
- Or search both and take the better match per memory.

**Impact:** Medium — improves precision for factual lookups where full-content embeddings are noisy.
**Effort:** Medium — second vec0 table, embedding at consolidation time, query routing logic.

---

## 5. Discovery / Recommendation Mode

**Problem:** Search is always query-to-nearest-neighbors. No way to say "find things like A but not like B."

**Qdrant's approach:** Discovery API takes positive and negative examples, searching for vectors close to positives and far from negatives.

**How to adopt:**

- New search mode: `POST /api/v1/memories/discover` with `positive_ids` and `negative_ids`.
- Compute a target vector: average of positive embeddings minus average of negative embeddings.
- Search for nearest neighbors to the target vector.
- Use cases:
  - Consolidation: when a contradiction is found, use newer memory as positive and superseded as negative to find other stale memories.
  - "More like this" in the web UI.
  - Agent use: "find memories related to X but not about Y."

**Impact:** Medium — enables new retrieval patterns, especially useful for consolidation quality.
**Effort:** Low — vector arithmetic + existing search infrastructure.

---

## 6. Result Grouping (deduplication by field)

**Problem:** Search can return multiple memories from the same episode or about the same topic, flooding results with near-duplicates.

**Qdrant's approach:** Group results by a payload field, returning only the best-scoring result per group.

**How to adopt:**

- Add optional `group_by` parameter to search: `episode_id`, `category`, or `source`.
- After scoring, group results by the field and keep only the top result per group.
- Return group metadata (count of memories in each group) so clients know there's more.
- Keeps result diversity high without requiring clients to deduplicate.

**Impact:** Medium — immediate UX improvement for search results, especially with episodic memories.
**Effort:** Low — post-processing step on scored results, no schema changes.

---

## 7. Oversampling + Two-Stage Rescoring

**Problem:** The reranker (cross-encoder) is expensive (~200ms). It only sees the top-N candidates from retrieval. If retrieval misses a good candidate, the reranker can't save it.

**Qdrant's approach:** Prefetch a larger candidate set with a cheap method (quantized vectors), then rescore with a more expensive method (full vectors or cross-encoder).

**How to adopt:**

- **Depends on Section 3 (Scalar Quantization).** Only worthwhile once quantized search is cheap.
- Currently `candidate_limit = limit * 5`. Increase to `limit * 10` when quantization is available (cheap with int8).
- Add an intermediate rescoring step between vector search and cross-encoder reranking:
  1. Quantized search: fetch 50 candidates (cheap)
  2. Full-precision rescore: narrow to 20 candidates (moderate)
  3. Cross-encoder rerank: narrow to final top-N (expensive)
- Three-stage pipeline: cheap recall, then precision, then understanding.

**Impact:** Low-Medium — marginal recall improvement when combined with quantization.
**Effort:** Low — pipeline restructuring, no new models needed.

---

## 8. Cursor-Based Pagination (Scroll API)

**Problem:** Listing memories uses `OFFSET/LIMIT`, which gets slower as offset grows (SQLite rescans).

**Qdrant's approach:** Scroll API uses a cursor (last seen ID + sort value) for efficient forward-only pagination.

**How to adopt:**

- Add `cursor` parameter to `GET /api/v1/memories` (encoded as `{sort_value}:{id}`).
- Query: `WHERE (sort_col, id) > (cursor_val, cursor_id) ORDER BY sort_col, id LIMIT ?`.
- Return `next_cursor` in response for the next page.
- Keep `offset/limit` for backward compatibility but mark as deprecated for large datasets.

**Impact:** Low — only matters with thousands of memories per namespace.
**Effort:** Low — SQL query change + response field.

---

## 9. Memory Backup & Restore (Settings Page) — P1 SHIPPED

**Problem:** Memories are stored in a single SQLite file. Users have no way to export, back up, or restore their memories from the UI. If the database is lost or corrupted, all memories are gone.

### P1 — Manual export/import from Settings UI — SHIPPED

**Shipped:**

- **`GET /api/v1/backup?format=sqlite|json`** endpoint. JSON format exports all memories in the current namespace as a `BackupEnvelope` (version, exported_at, namespace, memory_count, memories array). Excludes embeddings. SQLite format performs a WAL checkpoint then serves the raw database file as a download.
- **`POST /api/v1/restore`** endpoint. Accepts a JSON backup envelope with `mode=merge|replace`. Merge mode skips duplicates by content hash. Replace mode soft-deletes all existing memories in the namespace first. Re-embeds all imported memories in batches of 50.
- **Namespace-scoped export.** Export is filtered by the `X-Anima-Namespace` header — exports only the active namespace's memories.
- **Settings page UI.** "Backup & Restore" section in Advanced Settings with: JSON and SQLite download buttons, JSON import via file picker with confirmation dialog, memory count display, success/error feedback messages, loading spinners during operations.

**Not yet shipped (deferred to P2 or future):**

- SQLite restore from UI (currently export-only for SQLite; restore requires manual file replacement on server).
- Progress bar for large JSON imports (currently shows spinner, no percentage).
- Replace mode is not exposed in the UI (API supports it, but UI defaults to merge).
- Security gating — endpoints share the same auth as other API endpoints (none currently; must be gated if auth is added).

### P2 — Scheduled backups (future)

- Config option `[backup]` with `interval` (e.g., `"daily"`) and `path` (e.g., `"./backups/"`).
- Keeps last N backups with rotation.
- `GET /api/v1/backup/status` — returns last backup time, size, and scheduled next backup.

**Impact:** High — data safety is table stakes for a "your memories belong to you" product.
**Effort:** Low-Medium — SQLite backup API is built-in, JSON export/import requires serialization + batch re-embedding.

---

## Priority Order

Section numbers match the body above.

| Section | Feature | Impact | Effort | Priority |
|---------|---------|--------|--------|----------|
| 1 | Pre-filtering | High | Medium | P0 |
| 2 | Sparse vectors | High | High | P1 |
| 3 | Scalar quantization | Medium | Low | P1 |
| 6 | Result grouping | Medium | Low | P1 |
| 9 | Backup & restore (manual export/import) | High | Low-Medium | **SHIPPED** |
| 4 | Multi-vector | Medium | Medium | P2 |
| 5 | Discovery mode | Medium | Low | P2 |
| 7 | Oversampling + rescoring | Low-Medium | Low | P2 |
| 9 | Backup & restore (scheduled) | Low | Low | P2 |
| 8 | Cursor pagination | Low | Low | P3 |

---

## Design Constraints

- **No external services.** All features must work within SQLite + embedded models. No Qdrant, no Pinecone, no network dependencies.
- **Single-file database.** Everything stays in `anima.db`. No splitting data across storage systems.
- **Backward compatible.** Existing databases must migrate without data loss. New features are opt-in or auto-enabled with safe defaults.
- **Local-first.** All features must work fully offline. Cloud embedding backends are optional, not required.
