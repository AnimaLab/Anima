//! Briefing: the small block of memory a harness puts in front of a model.
//!
//! Two modes share one pipeline:
//!
//! - `session`: who the user is, what matters for the current project, and
//!   which skills exist. Sent once, at session start.
//! - `prompt`: memories that match one user message and are not already in
//!   the session briefing. Sent with each message.
//!
//! The pipeline is rank, collapse same-topic memories, then cut to a
//! character budget. It uses no LLM, so it is fast enough to run on every
//! prompt and gives the same output for the same store.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use anima_core::memory::Memory;
use anima_core::namespace::Namespace;

use crate::app::{AppError, AppState, ExtractNamespace};
use crate::dto::SearchRequest;

/// Categories that describe the user rather than a task, with how much each
/// counts. Routines matter less often than identity and preferences.
const STANDING: &[(&str, f64)] = &[("identity", 1.0), ("preference", 1.0), ("routine", 0.8)];

/// Cosine similarity above which two memories are about the same thing.
/// With BGE-M3, restatements and contradictions of one fact score around
/// 0.70 and unrelated personal facts stay under 0.60.
const SAME_TOPIC: f32 = 0.68;
/// Hybrid search scores under this are noise: unrelated memories score 0.
const MIN_SEARCH_SCORE: f64 = 0.02;

const SESSION_MEMORY_BUDGET: usize = 1400;
const SESSION_SKILL_BUDGET: usize = 700;
const PROMPT_BUDGET: usize = 600;
const PROMPT_MAX_LINES: usize = 4;
const MAX_LINE: usize = 220;
const MAX_NOTE: usize = 90;
const MAX_SKILL_DESCRIPTION: usize = 110;

#[derive(Debug, Deserialize)]
pub struct BriefingRequest {
    /// "session" (default) or "prompt".
    #[serde(default)]
    pub mode: Option<String>,
    /// Name of the project the session is about, usually the working folder.
    #[serde(default)]
    pub project: Option<String>,
    /// The user's message. Required in prompt mode.
    #[serde(default)]
    pub query: Option<String>,
    /// Character budget for memories. Defaults depend on the mode.
    #[serde(default)]
    pub budget_chars: Option<usize>,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct BriefingMemory {
    pub id: String,
    pub content: String,
    pub category: String,
    /// "profile", "project" or "prompt".
    pub section: String,
    /// A same-topic memory that lost to this one, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub also_on_record: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BriefingSkill {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Serialize)]
pub struct BriefingResponse {
    /// Ready to place in a model's context. Empty when there is nothing to say.
    pub text: String,
    pub memories: Vec<BriefingMemory>,
    pub skills: Vec<BriefingSkill>,
    /// Candidate memories left out by the budget.
    pub omitted: usize,
    pub elapsed_ms: f64,
}

/// A memory competing for a place in the briefing.
#[derive(Debug, Clone)]
struct Candidate {
    id: String,
    content: String,
    category: String,
    section: &'static str,
    confidence: f64,
    updated_at: DateTime<Utc>,
    /// Higher is more worth showing. Comparable only within a section.
    rank: f64,
    also_on_record: Option<String>,
}

fn flatten(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    let cut: String = flat.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut.trim_end())
}

/// How much a standing memory deserves a line: importance and confidence
/// first, then how often it was used and how fresh it is.
fn standing_rank(m: &Memory, weight: f64, now: DateTime<Utc>) -> f64 {
    let importance = (m.importance.clamp(1, 10) as f64) / 10.0;
    let access = ((1.0 + m.access_count as f64).ln() / 51f64.ln()).min(1.0);
    let age_days = (now - m.updated_at).num_seconds().max(0) as f64 / 86_400.0;
    let recency = (-age_days / 180.0).exp();
    weight * (0.35 * importance + 0.30 * m.confidence.clamp(0.0, 1.0) + 0.15 * access + 0.20 * recency)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

fn normalized(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Keep one memory per topic. The winner is the more trusted one, then the
/// more recent. A loser that says something different is kept as a short
/// note on the winner, so a contradiction is visible instead of silently
/// resolved; a plain restatement is dropped.
fn collapse_same_topic(mut candidates: Vec<Candidate>, embeddings: &HashMap<String, Vec<f32>>) -> Vec<Candidate> {
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.updated_at.cmp(&a.updated_at))
    });
    let mut kept: Vec<Candidate> = Vec::new();
    for c in candidates {
        let winner = embeddings.get(&c.id).and_then(|e| {
            kept.iter_mut().find(|k| {
                embeddings.get(&k.id).is_some_and(|ke| cosine(e, ke) >= SAME_TOPIC)
            })
        });
        match winner {
            Some(w) => {
                let (a, b) = (normalized(&w.content), normalized(&c.content));
                let restatement = a == b || a.contains(&b) || b.contains(&a);
                if !restatement && w.also_on_record.is_none() {
                    w.also_on_record = Some(flatten(&c.content, MAX_NOTE));
                }
                // The topic keeps the best rank either memory earned.
                w.rank = w.rank.max(c.rank);
            }
            None => kept.push(c),
        }
    }
    kept
}

fn line(c: &Candidate) -> String {
    let mut line = format!("- {}", flatten(&c.content, MAX_LINE));
    if let Some(other) = &c.also_on_record {
        line.push_str(&format!(" (also on record, less trusted: \"{other}\")"));
    }
    line
}

/// Take candidates in rank order until the budget is spent. Returns the
/// chosen ones and how many were left out.
fn fit(mut candidates: Vec<Candidate>, budget: usize, max_lines: usize) -> (Vec<Candidate>, usize) {
    candidates.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap_or(std::cmp::Ordering::Equal));
    let total = candidates.len();
    let mut used = 0usize;
    let mut chosen = Vec::new();
    for c in candidates {
        let cost = line(&c).chars().count() + 1;
        if chosen.len() >= max_lines || used + cost > budget {
            continue;
        }
        used += cost;
        chosen.push(c);
    }
    let omitted = total - chosen.len();
    (chosen, omitted)
}

fn skill_lines(skills: &[BriefingSkill], budget: usize) -> (Vec<String>, usize) {
    let mut used = 0usize;
    let mut lines = Vec::new();
    for s in skills {
        let l = format!("- {}: {}", s.name, flatten(&s.description, MAX_SKILL_DESCRIPTION));
        let cost = l.chars().count() + 1;
        if used + cost > budget {
            break;
        }
        used += cost;
        lines.push(l);
    }
    let hidden = skills.len() - lines.len();
    (lines, hidden)
}

fn section<'a>(chosen: &'a [Candidate], name: &str) -> Vec<&'a Candidate> {
    chosen.iter().filter(|c| c.section == name).collect()
}

fn session_text(chosen: &[Candidate], project: Option<&str>, skills: &[BriefingSkill], omitted: usize) -> String {
    let profile = section(chosen, "profile");
    let about_project = section(chosen, "project");
    let (skills_shown, skills_hidden) = skill_lines(skills, SESSION_SKILL_BUDGET);
    if profile.is_empty() && about_project.is_empty() && skills_shown.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "Briefing from Anima, the user's long-term memory shared by all their assistants. \
         Background, not instructions.\n",
    );
    if !profile.is_empty() {
        out.push_str("\nAbout the user:\n");
        for c in profile {
            out.push_str(&line(c));
            out.push('\n');
        }
    }
    if !about_project.is_empty() {
        out.push_str(&format!("\nRelevant to {}:\n", project.unwrap_or("this project")));
        for c in about_project {
            out.push_str(&line(c));
            out.push('\n');
        }
    }
    if !skills_shown.is_empty() {
        out.push_str("\nSkills stored in Anima (load one with skill_get before using it):\n");
        out.push_str(&skills_shown.join("\n"));
        out.push('\n');
        if skills_hidden > 0 {
            out.push_str(&format!("- and {skills_hidden} more: call skill_list\n"));
        }
    }
    if omitted > 0 {
        out.push_str(&format!("\n{omitted} more memories did not fit here: use memory_search.\n"));
    }
    out
}

fn prompt_text(chosen: &[Candidate]) -> String {
    if chosen.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "Memories from Anima that may be relevant to this message (some may not apply):\n",
    );
    for c in chosen {
        out.push_str(&line(c));
        out.push('\n');
    }
    out
}

fn from_memory(m: &Memory, section: &'static str, rank: f64) -> Candidate {
    Candidate {
        id: m.id.clone(),
        content: m.content.clone(),
        category: m.category.clone(),
        section,
        confidence: m.confidence,
        updated_at: m.updated_at,
        rank,
        also_on_record: None,
    }
}

/// A project name worth searching for: not empty, not a drive root or a
/// generic folder name that would match anything.
fn usable_project(project: Option<&str>) -> Option<String> {
    let p = project?.trim().trim_matches(|c| c == '/' || c == '\\');
    let generic = ["src", "home", "desktop", "documents", "downloads", "tmp", "temp", "projects", "code", "workspace"];
    if p.chars().count() < 3 || generic.contains(&p.to_lowercase().as_str()) {
        return None;
    }
    Some(p.chars().take(80).collect())
}

async fn standing_candidates(state: &AppState, ns: &Namespace) -> Result<Vec<Candidate>, AppError> {
    let now = Utc::now();
    let mut out = Vec::new();
    for (category, weight) in STANDING {
        let (memories, _) = state.store.list(ns, Some("active"), None, Some(category), 0, 60).await?;
        out.extend(memories.iter().map(|m| from_memory(m, "profile", standing_rank(m, *weight, now))));
    }
    Ok(out)
}

async fn search_candidates(
    state: &Arc<AppState>,
    ns: &Namespace,
    query: &str,
    section: &'static str,
) -> Result<Vec<Candidate>, AppError> {
    let req: SearchRequest = serde_json::from_value(serde_json::json!({ "query": query, "limit": 8 }))
        .map_err(|e| AppError::Internal(e.to_string()))?;
    let found = crate::handlers::search_memories(State(state.clone()), ExtractNamespace(ns.clone()), Json(req))
        .await?
        .0;
    let mut out = Vec::new();
    for r in found.results.into_iter().filter(|r| r.score >= MIN_SEARCH_SCORE) {
        let updated_at = DateTime::parse_from_rfc3339(&r.updated_at)
            .map(|t| t.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        out.push(Candidate {
            id: r.id,
            content: r.content,
            category: r.category,
            section,
            confidence: r.confidence,
            updated_at,
            rank: r.score,
            also_on_record: None,
        });
    }
    Ok(out)
}

async fn embeddings_for(state: &AppState, candidates: &[Candidate]) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    for c in candidates {
        if let Ok(named) = state.store.get_memory_embeddings(&c.id).await {
            let picked = named.get("content").cloned().or_else(|| named.into_values().next());
            if let Some(e) = picked {
                out.insert(c.id.clone(), e);
            }
        }
    }
    out
}

/// The profile part of a session briefing. Prompt mode recomputes it to
/// avoid repeating what the session already received.
async fn profile_selection(state: &AppState, ns: &Namespace, budget: usize) -> Result<(Vec<Candidate>, usize), AppError> {
    let candidates = standing_candidates(state, ns).await?;
    let embeddings = embeddings_for(state, &candidates).await;
    Ok(fit(collapse_same_topic(candidates, &embeddings), budget, usize::MAX))
}

fn to_response_memories(chosen: &[Candidate]) -> Vec<BriefingMemory> {
    chosen
        .iter()
        .map(|c| BriefingMemory {
            id: c.id.clone(),
            content: c.content.clone(),
            category: c.category.clone(),
            section: c.section.to_string(),
            also_on_record: c.also_on_record.clone(),
        })
        .collect()
}

pub async fn briefing(
    State(state): State<Arc<AppState>>,
    ExtractNamespace(ns): ExtractNamespace,
    Json(req): Json<BriefingRequest>,
) -> Result<Json<BriefingResponse>, AppError> {
    let start = std::time::Instant::now();
    let elapsed = |s: std::time::Instant| s.elapsed().as_secs_f64() * 1000.0;

    if req.mode.as_deref() == Some("prompt") {
        let query = req.query.as_deref().map(str::trim).unwrap_or("");
        if query.is_empty() {
            return Err(AppError::BadRequest("prompt mode needs a query".into()));
        }
        let (in_session, _) = profile_selection(&state, &ns, SESSION_MEMORY_BUDGET).await?;
        let seen: HashSet<&str> = in_session.iter().map(|c| c.id.as_str()).collect();
        let candidates: Vec<Candidate> = search_candidates(&state, &ns, query, "prompt")
            .await?
            .into_iter()
            .filter(|c| !seen.contains(c.id.as_str()))
            .collect();
        let embeddings = embeddings_for(&state, &candidates).await;
        let (chosen, omitted) = fit(
            collapse_same_topic(candidates, &embeddings),
            req.budget_chars.unwrap_or(PROMPT_BUDGET),
            PROMPT_MAX_LINES,
        );
        return Ok(Json(BriefingResponse {
            text: prompt_text(&chosen),
            memories: to_response_memories(&chosen),
            skills: vec![],
            omitted,
            elapsed_ms: elapsed(start),
        }));
    }

    let budget = req.budget_chars.unwrap_or(SESSION_MEMORY_BUDGET);
    let project = usable_project(req.project.as_deref());

    // The profile gets first call on the budget; the project gets at least a third.
    let mut candidates = standing_candidates(&state, &ns).await?;
    let profile_ids: HashSet<String> = candidates.iter().map(|c| c.id.clone()).collect();
    if let Some(p) = &project {
        // A search that fails must not cost the user their profile.
        if let Ok(found) = search_candidates(&state, &ns, p, "project").await {
            candidates.extend(found.into_iter().filter(|c| !profile_ids.contains(&c.id)));
        }
    }
    let embeddings = embeddings_for(&state, &candidates).await;
    let collapsed = collapse_same_topic(candidates, &embeddings);
    let (profile, about_project): (Vec<Candidate>, Vec<Candidate>) =
        collapsed.into_iter().partition(|c| c.section == "profile");
    let project_budget = if about_project.is_empty() { 0 } else { budget / 3 };
    let (mut chosen, omitted_profile) = fit(profile, budget - project_budget, usize::MAX);
    let (chosen_project, omitted_project) = fit(about_project, project_budget, 5);
    chosen.extend(chosen_project);

    let skills: Vec<BriefingSkill> = crate::skills::all_skills(&state, ns.as_str())
        .await?
        .into_iter()
        .map(|(s, _)| BriefingSkill { name: s.name, description: s.description })
        .collect();
    let omitted = omitted_profile + omitted_project;

    Ok(Json(BriefingResponse {
        text: session_text(&chosen, project.as_deref(), &skills, omitted),
        memories: to_response_memories(&chosen),
        skills,
        omitted,
        elapsed_ms: elapsed(start),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, content: &str, confidence: f64, days_old: i64, rank: f64) -> Candidate {
        Candidate {
            id: id.into(),
            content: content.into(),
            category: "preference".into(),
            section: "profile",
            confidence,
            updated_at: Utc::now() - chrono::Duration::days(days_old),
            rank,
            also_on_record: None,
        }
    }

    fn embeddings(list: &[(&str, [f32; 2])]) -> HashMap<String, Vec<f32>> {
        list.iter().map(|(id, v)| (id.to_string(), v.to_vec())).collect()
    }

    #[test]
    fn contradiction_keeps_the_more_trusted_memory_and_notes_the_other() {
        let candidates = vec![
            cand("python", "Favorite programming language is Python.", 0.7, 1, 0.5),
            cand("rust", "My favorite programming language is Rust.", 1.0, 30, 0.4),
            cand("hiking", "I love hiking on weekends.", 1.0, 5, 0.3),
        ];
        let e = embeddings(&[("python", [1.0, 0.05]), ("rust", [1.0, 0.0]), ("hiking", [0.0, 1.0])]);
        let kept = collapse_same_topic(candidates, &e);
        let ids: Vec<&str> = kept.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"rust") && ids.contains(&"hiking") && !ids.contains(&"python"));
        let rust = kept.iter().find(|c| c.id == "rust").unwrap();
        assert_eq!(rust.also_on_record.as_deref(), Some("Favorite programming language is Python."));
        assert_eq!(rust.rank, 0.5, "the topic keeps the best rank");
        assert!(line(rust).contains("also on record, less trusted"));
    }

    #[test]
    fn equal_trust_prefers_the_newer_memory() {
        let candidates = vec![cand("old", "Lives in Paris.", 1.0, 400, 0.1), cand("new", "Lives in Lyon.", 1.0, 2, 0.1)];
        let e = embeddings(&[("old", [1.0, 0.0]), ("new", [1.0, 0.0])]);
        let kept = collapse_same_topic(candidates, &e);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "new");
        assert_eq!(kept[0].also_on_record.as_deref(), Some("Lives in Paris."));
    }

    #[test]
    fn restatements_are_dropped_without_a_note() {
        let candidates = vec![cand("a", "Prefers dark mode.", 1.0, 1, 0.1), cand("b", "prefers dark mode", 0.9, 1, 0.1)];
        let e = embeddings(&[("a", [1.0, 0.0]), ("b", [1.0, 0.0])]);
        let kept = collapse_same_topic(candidates, &e);
        assert_eq!(kept.len(), 1);
        assert!(kept[0].also_on_record.is_none());
    }

    #[test]
    fn memories_without_embeddings_are_never_merged() {
        let candidates = vec![cand("a", "One.", 1.0, 1, 0.1), cand("b", "Two.", 1.0, 1, 0.1)];
        assert_eq!(collapse_same_topic(candidates, &HashMap::new()).len(), 2);
    }

    #[test]
    fn budget_takes_the_best_ranked_and_counts_the_rest() {
        let candidates: Vec<Candidate> = (0..10)
            .map(|i| cand(&format!("m{i}"), &"x".repeat(98), 1.0, 1, i as f64))
            .collect();
        let (chosen, omitted) = fit(candidates, 310, usize::MAX);
        assert_eq!(chosen.len(), 3, "three 101-character lines fit in 310");
        assert_eq!(omitted, 7);
        assert_eq!(chosen[0].id, "m9");
        let (capped, _) = fit((0..10).map(|i| cand(&format!("m{i}"), "short", 1.0, 1, i as f64)).collect(), 10_000, 4);
        assert_eq!(capped.len(), 4);
    }

    #[test]
    fn long_lines_are_cut() {
        let c = cand("a", &"word ".repeat(200), 1.0, 1, 1.0);
        assert!(line(&c).chars().count() <= MAX_LINE + 2);
        assert!(line(&c).ends_with('…'));
    }

    #[test]
    fn rank_rewards_importance_confidence_use_and_freshness() {
        let now = Utc::now();
        let mut base = Memory::new("default".into(), "x".into(), None, vec![], None);
        base.confidence = 0.5;
        let r0 = standing_rank(&base, 1.0, now);
        let mut m = base.clone();
        m.importance = 9;
        assert!(standing_rank(&m, 1.0, now) > r0);
        let mut m = base.clone();
        m.confidence = 1.0;
        assert!(standing_rank(&m, 1.0, now) > r0);
        let mut m = base.clone();
        m.access_count = 20;
        assert!(standing_rank(&m, 1.0, now) > r0);
        let mut m = base.clone();
        m.updated_at = now - chrono::Duration::days(700);
        assert!(standing_rank(&m, 1.0, now) < r0);
        assert!(standing_rank(&base, 0.8, now) < r0);
    }

    #[test]
    fn session_text_has_sections_and_is_empty_when_nothing_to_say() {
        assert_eq!(session_text(&[], Some("anima"), &[], 0), "");
        let mut project = cand("p", "Billing service is being migrated to Rust.", 1.0, 1, 1.0);
        project.section = "project";
        let chosen = vec![cand("a", "Prefers concise answers.", 1.0, 1, 1.0), project];
        let skills = vec![BriefingSkill { name: "deploy-api".into(), description: "Ship the API to production.".into() }];
        let text = session_text(&chosen, Some("billing"), &skills, 3);
        assert!(text.contains("About the user:\n- Prefers concise answers."));
        assert!(text.contains("Relevant to billing:\n- Billing service"));
        assert!(text.contains("- deploy-api: Ship the API to production."));
        assert!(text.contains("3 more memories did not fit"));
    }

    #[test]
    fn skill_index_respects_its_budget() {
        let skills: Vec<BriefingSkill> = (0..40)
            .map(|i| BriefingSkill { name: format!("skill-{i}"), description: "d".repeat(300) })
            .collect();
        let (lines, hidden) = skill_lines(&skills, SESSION_SKILL_BUDGET);
        assert!(lines.len() < 10 && hidden == 40 - lines.len());
        assert!(lines.iter().all(|l| l.chars().count() <= MAX_SKILL_DESCRIPTION + 14));
    }

    #[test]
    fn generic_folder_names_are_not_projects() {
        assert_eq!(usable_project(Some("anima")).as_deref(), Some("anima"));
        for p in ["", "src", "Desktop", "C:", "/"] {
            assert!(usable_project(Some(p)).is_none(), "{p:?}");
        }
        assert!(usable_project(None).is_none());
    }
}
