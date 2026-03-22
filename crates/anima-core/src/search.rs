use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::temporal::{apply_temporal_weight, exponential_decay};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Hybrid,
    Vector,
    Keyword,
}

impl Default for SearchMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Configuration for the hybrid scorer.
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// RRF constant (default: 60.0)
    pub rrf_k: f64,
    /// Weight for vector results in RRF (default: 0.6)
    pub weight_vector: f64,
    /// Weight for keyword results in RRF (default: 0.4)
    pub weight_keyword: f64,
    /// How much temporal decay influences the final score (0.0-1.0, default: 0.2)
    pub temporal_weight: f64,
    /// Exponential decay rate (default: 0.001, half-life ~29 days)
    pub lambda: f64,
    /// How much access frequency boosts score (default: 0.02)
    /// bonus = access_weight * ln(1 + access_count), added to score
    pub access_weight: f64,
    /// How much importance boosts score (default: 0.03)
    /// bonus = importance_weight * (importance - 5) / 5, added to score
    pub importance_weight: f64,
    /// Bonus per tier level above tier-1 (default: 0.0, disabled).
    /// tier 1 (raw) → +0.00, tier 2 (reflected) → +0.025, tier 3 (deduced) → +0.05
    pub tier_boost: f64,
    /// Minimum cosine similarity to include a vector result (default: 0.35).
    /// Filters noise from unrelated/gibberish queries.
    pub min_vector_similarity: f64,
    /// Minimum score spread to accept vector results (default: 0.055).
    /// If all vector results are within this range, they're considered noise.
    /// Set to 0.0 to disable the spread check.
    pub min_score_spread: f64,
    /// Maximum memory tier to include in search results (1-4, default: 4 = all).
    /// 1=raw only, 2=raw+reflected, 3=+deduced, 4=+induced.
    pub max_tier: i32,
    /// Optional ISO 8601 date filter (inclusive start).
    pub date_start: Option<String>,
    /// Optional ISO 8601 date filter (inclusive end).
    pub date_end: Option<String>,
    /// Per-category decay lambda overrides. Key = category name, value = lambda.
    /// Categories not in this map fall back to `self.lambda`.
    pub category_lambdas: HashMap<String, f64>,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            rrf_k: 10.0,
            weight_vector: 0.6,
            weight_keyword: 0.4,
            temporal_weight: 0.2,
            lambda: 0.001,
            access_weight: 0.02,
            importance_weight: 0.03,
            tier_boost: 0.0,
            min_vector_similarity: 0.55,
            min_score_spread: 0.055,
            max_tier: 4,
            date_start: None,
            date_end: None,
            category_lambdas: HashMap::new(),
        }
    }
}

/// A candidate from a single retrieval source.
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    pub memory_id: String,
    pub rank: usize, // 1-indexed
}

/// Final scored result with component scores for transparency.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredResult {
    pub memory_id: String,
    pub score: f64,
    pub vector_score: Option<f64>,
    pub keyword_score: Option<f64>,
    pub temporal_score: Option<f64>,
}

/// Hybrid scorer using Reciprocal Rank Fusion + temporal decay.
pub struct HybridScorer {
    pub config: ScorerConfig,
}

impl HybridScorer {
    pub fn new(config: ScorerConfig) -> Self {
        Self { config }
    }

    /// Fuse vector and keyword results using RRF + temporal decay.
    ///
    /// - `vector_results`: (memory_id, raw_similarity) ordered by similarity (best first)
    /// - `keyword_results`: (memory_id, raw_bm25_score) ordered by score (best first)
    /// - `timestamps`: updated_at timestamps for temporal decay
    /// - `now`: current time
    ///
    /// In hybrid mode, uses RRF to fuse both ranked lists.
    /// In single-source modes (vector-only or keyword-only), uses raw scores
    /// directly to preserve the actual similarity information.
    pub fn fuse(
        &self,
        vector_results: &[(String, f64)],
        keyword_results: &[(String, f64)],
        timestamps: &HashMap<String, DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> Vec<ScoredResult> {
        let has_vec = !vector_results.is_empty();
        let has_kw = !keyword_results.is_empty();

        // Single-source mode: use raw scores directly (no RRF)
        if has_vec && !has_kw {
            return self.score_single_source(vector_results, timestamps, now, true);
        }
        if has_kw && !has_vec {
            return self.score_single_source(keyword_results, timestamps, now, false);
        }
        if !has_vec && !has_kw {
            return vec![];
        }

        // Hybrid mode: weighted raw-score combination.
        // Vector scores are cosine similarity [0,1]. BM25 scores are unbounded
        // so we normalize them to [0,1] using the max BM25 score in this batch.
        let vec_raw: HashMap<String, f64> = vector_results.iter().cloned().collect();
        let kw_raw: HashMap<String, f64> = keyword_results.iter().cloned().collect();

        let max_bm25 = keyword_results.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
        let norm_bm25 = if max_bm25 > 0.0 { max_bm25 } else { 1.0 };

        // Collect all unique memory IDs
        let mut all_ids: HashMap<String, ()> = HashMap::new();
        for (id, _) in vector_results.iter().chain(keyword_results.iter()) {
            all_ids.entry(id.clone()).or_default();
        }

        let wv = self.config.weight_vector;
        let wk = self.config.weight_keyword;

        let mut results: Vec<ScoredResult> = all_ids
            .into_keys()
            .map(|id| {
                let vs = vec_raw.get(&id).copied().unwrap_or(0.0);
                let ks_norm = kw_raw.get(&id).copied().unwrap_or(0.0) / norm_bm25;
                let raw = wv * vs + wk * ks_norm;
                let decay = self.compute_decay(&id, timestamps, now);
                let final_score = apply_temporal_weight(raw, decay, self.config.temporal_weight);

                ScoredResult {
                    memory_id: id.clone(),
                    score: final_score,
                    vector_score: vec_raw.get(&id).copied(),
                    keyword_score: kw_raw.get(&id).copied(),
                    temporal_score: Some(decay),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Score results from a single source using raw scores directly.
    /// `is_vector` indicates whether these are vector (cosine sim [0,1]) or keyword (BM25, unbounded) results.
    fn score_single_source(
        &self,
        results_with_scores: &[(String, f64)],
        timestamps: &HashMap<String, DateTime<Utc>>,
        now: DateTime<Utc>,
        is_vector: bool,
    ) -> Vec<ScoredResult> {
        if results_with_scores.is_empty() {
            return vec![];
        }

        // For vector: cosine similarity is already in [0, 1], use directly.
        // For keyword: BM25 scores are unbounded, normalize by max.
        let max_raw = if is_vector {
            1.0 // cosine similarity max
        } else {
            let m = results_with_scores.iter()
                .map(|(_, s)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            if m > 0.0 { m } else { 1.0 }
        };

        let mut results: Vec<ScoredResult> = results_with_scores.iter()
            .map(|(id, raw_score)| {
                let normalized = (raw_score / max_raw).min(1.0);
                let decay = self.compute_decay(id, timestamps, now);
                let final_score = apply_temporal_weight(normalized, decay, self.config.temporal_weight);

                ScoredResult {
                    memory_id: id.clone(),
                    score: final_score.min(1.0),
                    vector_score: if is_vector { Some(normalized) } else { None },
                    keyword_score: if !is_vector { Some(normalized) } else { None },
                    temporal_score: Some(decay),
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn compute_decay(&self, id: &str, timestamps: &HashMap<String, DateTime<Utc>>, now: DateTime<Utc>) -> f64 {
        timestamps
            .get(id)
            .map(|ts| {
                let age_hours = (now - *ts).num_seconds() as f64 / 3600.0;
                exponential_decay(age_hours.max(0.0), self.config.lambda)
            })
            .unwrap_or(1.0)
    }

    /// Apply access frequency, importance, and tier boosts as post-processing.
    /// Boosts are additive (not multiplicative) to prevent low-relevance
    /// high-importance results from overtaking genuinely relevant ones.
    /// Final scores are clamped to [0, 1].
    pub fn apply_boosts(
        &self,
        results: &mut Vec<ScoredResult>,
        access_counts: &HashMap<String, u64>,
        importances: &HashMap<String, i32>,
        tiers: &HashMap<String, i32>,
    ) {
        if self.config.access_weight == 0.0
            && self.config.importance_weight == 0.0
            && self.config.tier_boost == 0.0
        {
            return;
        }

        for r in results.iter_mut() {
            // Access frequency boost: small additive bonus, logarithmic
            // 0 accesses → +0.0, 10 accesses → +0.048, 100 accesses → +0.092
            let access_bonus = if self.config.access_weight > 0.0 {
                let count = access_counts.get(&r.memory_id).copied().unwrap_or(0);
                self.config.access_weight * (1.0 + count as f64).ln()
            } else {
                0.0
            };

            // Importance boost: small additive bonus centered at 5 (neutral)
            // importance=5 → +0.0, importance=10 → +0.03, importance=1 → -0.024
            let importance_bonus = if self.config.importance_weight > 0.0 {
                let imp = importances.get(&r.memory_id).copied().unwrap_or(5);
                self.config.importance_weight * (imp as f64 - 5.0) / 5.0
            } else {
                0.0
            };

            // Tier boost: reflected (tier 2) and deduced (tier 3) memories are higher signal
            // tier 1 → +0.000, tier 2 → +0.025, tier 3 → +0.050 (with default tier_boost=0.05)
            let tier_bonus = if self.config.tier_boost > 0.0 {
                let tier = tiers.get(&r.memory_id).copied().unwrap_or(1).max(1);
                self.config.tier_boost * (tier as f64 - 1.0) / 2.0
            } else {
                0.0
            };

            r.score = (r.score + access_bonus + importance_bonus + tier_bonus).clamp(0.0, 1.0);
        }

        // Re-sort after boosting
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn default_scorer() -> HybridScorer {
        HybridScorer::new(ScorerConfig::default())
    }

    fn timestamps_now(ids: &[&str]) -> (HashMap<String, DateTime<Utc>>, DateTime<Utc>) {
        let now = Utc::now();
        let ts: HashMap<String, DateTime<Utc>> = ids.iter().map(|id| (id.to_string(), now)).collect();
        (ts, now)
    }

    /// Helper: convert a list of IDs into scored tuples with descending scores.
    /// Score starts at 0.9 and decreases by 0.05 for each rank.
    fn scored(ids: &[&str]) -> Vec<(String, f64)> {
        ids.iter().enumerate()
            .map(|(i, id)| (id.to_string(), 0.9 - i as f64 * 0.05))
            .collect()
    }

    // ═══════════════════════════════════════════════════════════
    // HYBRID MODE TESTS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn hybrid_basic_ranking() {
        // a=vec1/kw3, b=vec2/kw1, c=vec3/kw2
        // b gets best combined rank (high in both)
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b", "c"]);
        let vector = scored(&["a", "b", "c"]);
        let keyword = scored(&["b", "c", "a"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].memory_id, "b", "b should rank first (vec2+kw1)");
        assert!(results.iter().all(|r| r.score > 0.0 && r.score <= 1.0));
    }

    #[test]
    fn hybrid_all_scores_in_0_1_range() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b", "c", "d", "e"]);
        let vector = scored(&["a", "b", "c", "d", "e"]);
        let keyword = scored(&["e", "d", "c", "b", "a"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0,
                "score {} out of [0,1] for {}", r.score, r.memory_id);
            if let Some(vs) = r.vector_score {
                assert!(vs >= 0.0 && vs <= 1.0,
                    "vector_score {} out of [0,1] for {}", vs, r.memory_id);
            }
            if let Some(ks) = r.keyword_score {
                assert!(ks >= 0.0 && ks <= 1.0,
                    "keyword_score {} out of [0,1] for {}", ks, r.memory_id);
            }
        }
    }

    #[test]
    fn hybrid_rank1_in_both_gets_max_score() {
        // "a" is rank 1 in both vector and keyword → should get ~1.0 score
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b"]);
        let vector = scored(&["a", "b"]);
        let keyword = scored(&["a", "b"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results[0].memory_id, "a");
        assert!(results[0].score > 0.90,
            "rank-1 in both should score high, got {}", results[0].score);
    }

    #[test]
    fn hybrid_results_sorted_descending() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b", "c", "d"]);
        let vector = scored(&["a", "b", "c", "d"]);
        let keyword = scored(&["d", "c", "b", "a"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score,
                "results not sorted: {} ({}) before {} ({})",
                w[0].memory_id, w[0].score, w[1].memory_id, w[1].score);
        }
    }

    #[test]
    fn hybrid_no_duplicate_ids() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b", "c"]);
        let vector = scored(&["a", "b", "c"]);
        let keyword = scored(&["b", "a", "c"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        let ids: Vec<&str> = results.iter().map(|r| r.memory_id.as_str()).collect();
        let mut unique = ids.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(ids.len(), unique.len(), "duplicate IDs in results");
    }

    #[test]
    fn hybrid_disjoint_results_union() {
        // Vector and keyword return completely different IDs
        let scorer = default_scorer();
        let (mut ts, now) = timestamps_now(&["v1", "v2"]);
        ts.insert("k1".into(), now);
        ts.insert("k2".into(), now);

        let vector = scored(&["v1", "v2"]);
        let keyword = scored(&["k1", "k2"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results.len(), 4, "should contain union of both result sets");
        let v1 = results.iter().find(|r| r.memory_id == "v1").unwrap();
        assert!(v1.vector_score.is_some());
        assert!(v1.keyword_score.is_none());
        let k1 = results.iter().find(|r| r.memory_id == "k1").unwrap();
        assert!(k1.keyword_score.is_some());
        assert!(k1.vector_score.is_none());
    }

    // ═══════════════════════════════════════════════════════════
    // VECTOR-ONLY MODE TESTS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn vector_only_preserves_ranking() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b", "c"]);
        let vector = scored(&["a", "b", "c"]);
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&vector, &empty, &ts, now);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].memory_id, "a");
        assert_eq!(results[1].memory_id, "b");
        assert_eq!(results[2].memory_id, "c");
    }

    #[test]
    fn vector_only_no_keyword_scores() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b"]);
        let vector = scored(&["a", "b"]);
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&vector, &empty, &ts, now);

        for r in &results {
            assert!(r.vector_score.is_some(), "{} missing vector_score", r.memory_id);
            assert!(r.keyword_score.is_none(), "{} should have no keyword_score", r.memory_id);
        }
    }

    #[test]
    fn vector_only_score_math() {
        let scorer = HybridScorer::new(ScorerConfig::default());
        let (ts, now) = timestamps_now(&["a"]);
        let vector = vec![("a".to_string(), 0.9)];
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&vector, &empty, &ts, now);

        // Raw cosine sim 0.9 used directly, with minimal temporal decay
        assert!((results[0].score - 0.9).abs() < 0.01,
            "vector-only should use raw similarity, got {:.4}", results[0].score);
        assert!((results[0].vector_score.unwrap() - 0.9).abs() < 0.01);
    }

    // ═══════════════════════════════════════════════════════════
    // KEYWORD-ONLY MODE TESTS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn keyword_only_preserves_ranking() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b", "c"]);
        let empty: Vec<(String, f64)> = vec![];
        let keyword = scored(&["a", "b", "c"]);

        let results = scorer.fuse(&empty, &keyword, &ts, now);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].memory_id, "a");
        assert_eq!(results[1].memory_id, "b");
        assert_eq!(results[2].memory_id, "c");
    }

    #[test]
    fn keyword_only_no_vector_scores() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["a", "b"]);
        let empty: Vec<(String, f64)> = vec![];
        let keyword = scored(&["a", "b"]);

        let results = scorer.fuse(&empty, &keyword, &ts, now);

        for r in &results {
            assert!(r.keyword_score.is_some(), "{} missing keyword_score", r.memory_id);
            assert!(r.vector_score.is_none(), "{} should have no vector_score", r.memory_id);
        }
    }

    #[test]
    fn keyword_only_score_math() {
        let scorer = HybridScorer::new(ScorerConfig::default());
        let (ts, now) = timestamps_now(&["a"]);
        let empty: Vec<(String, f64)> = vec![];
        let keyword = vec![("a".to_string(), 5.0)];

        let results = scorer.fuse(&empty, &keyword, &ts, now);

        assert!(results[0].score > 0.99,
            "keyword-only rank-1 should score ~1.0, got {:.4}", results[0].score);
    }

    // ═══════════════════════════════════════════════════════════
    // TEMPORAL DECAY TESTS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn temporal_zero_age_no_penalty() {
        let cfg = ScorerConfig { temporal_weight: 1.0, ..Default::default() };
        let scorer = HybridScorer::new(cfg);
        let (ts, now) = timestamps_now(&["a"]);
        let vector = scored(&["a"]);
        let keyword = scored(&["a"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results[0].temporal_score, Some(1.0));
        assert!(results[0].score > 0.90, "fresh result should score high, got {}", results[0].score);
    }

    #[test]
    fn temporal_old_result_penalized() {
        let cfg = ScorerConfig { temporal_weight: 1.0, lambda: 0.001, ..Default::default() };
        let scorer = HybridScorer::new(cfg);
        let now = Utc::now();
        let mut ts = HashMap::new();
        ts.insert("old".into(), now - Duration::hours(720));
        ts.insert("new".into(), now);

        let vector = scored(&["old", "new"]);
        let keyword = scored(&["old", "new"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        let old = results.iter().find(|r| r.memory_id == "old").unwrap();
        let new = results.iter().find(|r| r.memory_id == "new").unwrap();

        assert!(old.temporal_score.unwrap() < 0.5, "30-day decay should be < 0.5");
        assert!(new.score > old.score, "newer result should score higher");
    }

    #[test]
    fn temporal_decay_flips_ranking() {
        // "old" has better raw score but is very old → "new" should win
        let cfg = ScorerConfig {
            temporal_weight: 0.9,
            lambda: 0.01,
            ..Default::default()
        };
        let scorer = HybridScorer::new(cfg);
        let now = Utc::now();
        let mut ts = HashMap::new();
        ts.insert("old".into(), now - Duration::hours(500));
        ts.insert("new".into(), now);

        // old has higher raw score but is much older
        let vector = vec![("old".to_string(), 0.9), ("new".to_string(), 0.85)];
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&vector, &empty, &ts, now);

        assert_eq!(results[0].memory_id, "new",
            "recency should flip ranking: new={:.4} vs old={:.4}",
            results.iter().find(|r| r.memory_id == "new").unwrap().score,
            results.iter().find(|r| r.memory_id == "old").unwrap().score);
    }

    #[test]
    fn temporal_weight_zero_ignores_age() {
        let cfg = ScorerConfig { temporal_weight: 0.0, ..Default::default() };
        let scorer = HybridScorer::new(cfg);
        let now = Utc::now();
        let mut ts = HashMap::new();
        ts.insert("old".into(), now - Duration::hours(10_000));
        ts.insert("new".into(), now);

        let vector = vec![("old".to_string(), 0.9), ("new".to_string(), 0.85)];
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&vector, &empty, &ts, now);

        assert_eq!(results[0].memory_id, "old");
    }

    #[test]
    fn temporal_missing_timestamp_gets_no_penalty() {
        let cfg = ScorerConfig { temporal_weight: 1.0, ..Default::default() };
        let scorer = HybridScorer::new(cfg);
        let now = Utc::now();
        let ts: HashMap<String, DateTime<Utc>> = HashMap::new();

        let vector = vec![("a".to_string(), 0.9)];
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&vector, &empty, &ts, now);

        assert_eq!(results[0].temporal_score, Some(1.0),
            "missing timestamp should default to decay=1.0");
    }

    // ═══════════════════════════════════════════════════════════
    // RRF WEIGHT CONFIGURATION TESTS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn rrf_weights_affect_relative_importance() {
        let cfg = ScorerConfig {
            weight_vector: 0.9,
            weight_keyword: 0.1,
            temporal_weight: 0.0,
            ..Default::default()
        };
        let scorer = HybridScorer::new(cfg);
        let (ts, now) = timestamps_now(&["a", "b"]);
        let vector = scored(&["a", "b"]);
        let keyword = scored(&["b", "a"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results[0].memory_id, "a",
            "with heavy vector weight, vec-rank1 should dominate");
    }

    #[test]
    fn weight_vector_heavy_favors_vector_rank() {
        let now = Utc::now();
        let (ts, _) = timestamps_now(&["a", "b"]);

        let vec_heavy = ScorerConfig { weight_vector: 0.9, weight_keyword: 0.1, temporal_weight: 0.0, ..Default::default() };
        let kw_heavy = ScorerConfig { weight_vector: 0.1, weight_keyword: 0.9, temporal_weight: 0.0, ..Default::default() };

        let scorer_vec = HybridScorer::new(vec_heavy);
        let scorer_kw = HybridScorer::new(kw_heavy);

        // "a" is rank-1 vector, "b" is rank-1 keyword
        let vector = vec![("a".to_string(), 0.95), ("b".to_string(), 0.5)];
        let keyword = vec![("b".to_string(), 10.0), ("a".to_string(), 2.0)];

        let results_vec = scorer_vec.fuse(&vector, &keyword, &ts, now);
        let results_kw = scorer_kw.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results_vec[0].memory_id, "a", "vector-heavy should rank 'a' first");
        assert_eq!(results_kw[0].memory_id, "b", "keyword-heavy should rank 'b' first");
    }

    // ═══════════════════════════════════════════════════════════
    // NORMALIZATION TESTS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn normalization_vector_uses_raw_similarity() {
        let scorer = HybridScorer::new(ScorerConfig::default());
        let (ts, now) = timestamps_now(&["a"]);
        let vector = vec![("a".to_string(), 0.85)];
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&vector, &empty, &ts, now);

        // Vector scores use raw cosine similarity directly (already [0,1])
        assert!((results[0].vector_score.unwrap() - 0.85).abs() < 0.01,
            "vector should use raw similarity, got {}", results[0].vector_score.unwrap());
    }

    #[test]
    fn normalization_keyword_rank1_is_1() {
        let scorer = HybridScorer::new(ScorerConfig::default());
        let (ts, now) = timestamps_now(&["a"]);
        let empty: Vec<(String, f64)> = vec![];
        let keyword = vec![("a".to_string(), 5.0)];

        let results = scorer.fuse(&empty, &keyword, &ts, now);

        assert!((results[0].keyword_score.unwrap() - 1.0).abs() < 0.01,
            "keyword rank-1 should normalize to 1.0, got {}", results[0].keyword_score.unwrap());
    }

    #[test]
    fn normalization_both_rank1_combined_is_1() {
        let cfg = ScorerConfig { temporal_weight: 0.0, ..Default::default() };
        let scorer = HybridScorer::new(cfg);
        let (ts, now) = timestamps_now(&["a"]);
        let vector = scored(&["a"]);
        let keyword = scored(&["a"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        // With weighted raw scores: 0.6 * 0.9 (vector) + 0.4 * 1.0 (normalized BM25) = 0.94
        assert!(results[0].score > 0.90,
            "rank-1 in both with no temporal should score high, got {}", results[0].score);
    }

    #[test]
    fn normalization_single_mode_scores() {
        let cfg = ScorerConfig { temporal_weight: 0.0, ..Default::default() };
        let scorer = HybridScorer::new(cfg);
        let (ts, now) = timestamps_now(&["a"]);
        let empty: Vec<(String, f64)> = vec![];

        // Vector: raw cosine sim used directly
        let results = scorer.fuse(&vec![("a".to_string(), 0.8)], &empty, &ts, now);
        assert!((results[0].score - 0.8).abs() < 0.01,
            "vector-only should use raw similarity, got {:.4}", results[0].score);

        // Keyword: BM25 normalized by max → rank-1 = 1.0
        let results = scorer.fuse(&empty, &vec![("a".to_string(), 5.0)], &ts, now);
        assert!(results[0].score > 0.99,
            "keyword-only rank-1 should be ~1.0, got {:.4}", results[0].score);
    }

    // ═══════════════════════════════════════════════════════════
    // EDGE CASES
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn empty_inputs_returns_empty() {
        let scorer = default_scorer();
        let now = Utc::now();
        let ts: HashMap<String, DateTime<Utc>> = HashMap::new();
        let empty: Vec<(String, f64)> = vec![];

        let results = scorer.fuse(&empty, &empty, &ts, now);
        assert!(results.is_empty());
    }

    #[test]
    fn single_result_in_both() {
        let scorer = default_scorer();
        let (ts, now) = timestamps_now(&["only"]);
        let vector = scored(&["only"]);
        let keyword = scored(&["only"]);

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, "only");
        assert!(results[0].vector_score.is_some());
        assert!(results[0].keyword_score.is_some());
        assert!(results[0].temporal_score.is_some());
    }

    #[test]
    fn large_result_set() {
        let scorer = default_scorer();
        let now = Utc::now();
        let ids: Vec<String> = (0..100).map(|i| format!("mem_{i}")).collect();
        let ts: HashMap<String, DateTime<Utc>> = ids.iter().map(|id| (id.clone(), now)).collect();
        let vector: Vec<(String, f64)> = ids.iter().enumerate()
            .map(|(i, id)| (id.clone(), 0.9 - i as f64 * 0.005))
            .collect();
        let keyword: Vec<(String, f64)> = ids.iter().rev().enumerate()
            .map(|(i, id)| (id.clone(), 0.9 - i as f64 * 0.005))
            .collect();

        let results = scorer.fuse(&vector, &keyword, &ts, now);

        assert_eq!(results.len(), 100);
        for r in &results {
            assert!(r.score > 0.0 && r.score <= 1.0, "invalid score {}", r.score);
        }
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
}
