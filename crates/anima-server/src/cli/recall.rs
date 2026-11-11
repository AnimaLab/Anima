//! `recall`: print memories for a harness hook.
//!
//! A harness runs this at session start, or on each prompt, and adds what it
//! prints to the model's context. A hook must never get in the user's way,
//! so every failure (server down, slow, bad input) prints nothing and exits 0.
//!
//! With `--prompt`, the hook's JSON payload is read from stdin and its
//! `prompt` field is searched. Without it, the standing profile (identity and
//! preferences) is printed.

use std::io::Read;
use std::time::Duration;

use serde_json::{json, Value};

use super::client::Api;
use super::Options;

const TIMEOUT: Duration = Duration::from_secs(4);
/// Below this relevance a search hit is more likely noise than context.
/// Hybrid scores are small: unrelated memories score 0, loosely related ones
/// around 0.03, strong matches 0.2 and up.
const MIN_SCORE: f64 = 0.02;
const MAX_LINE_CHARS: usize = 400;

pub async fn run(opts: &Options) -> anyhow::Result<()> {
    let Ok(api) = Api::from_options(opts, TIMEOUT) else {
        return Ok(());
    };
    let text = if opts.has("--prompt") {
        for_prompt(&api).await
    } else {
        profile(&api).await
    };
    if let Some(text) = text {
        print!("{text}");
    }
    Ok(())
}

fn one_line(content: &str) -> String {
    let flat = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= MAX_LINE_CHARS {
        return flat;
    }
    let cut: String = flat.chars().take(MAX_LINE_CHARS).collect();
    format!("{cut}…")
}

fn bullets<'a>(items: impl Iterator<Item = &'a Value>) -> Vec<String> {
    items
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .map(|c| format!("- {}", one_line(c)))
        .collect()
}

/// The prompt text from a hook payload, when it is worth searching for.
fn prompt_from_payload(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let prompt = value.get("prompt")?.as_str()?.trim();
    // Slash commands and one-word replies carry nothing to search for.
    if prompt.starts_with('/') || prompt.chars().count() < 12 {
        return None;
    }
    Some(prompt.chars().take(2000).collect())
}

fn format_block(title: &str, lines: &[String]) -> Option<String> {
    if lines.is_empty() {
        return None;
    }
    Some(format!("{title}\n{}\n", lines.join("\n")))
}

async fn for_prompt(api: &Api) -> Option<String> {
    let mut payload = String::new();
    std::io::stdin().read_to_string(&mut payload).ok()?;
    let prompt = prompt_from_payload(&payload)?;
    let found = api
        .post("/api/v1/memories/search", &json!({ "query": prompt, "limit": 6, "search_mode": "hybrid" }))
        .await
        .ok()?;
    let hits = found.get("results")?.as_array()?;
    let relevant = hits
        .iter()
        .filter(|m| m.get("score").and_then(Value::as_f64).unwrap_or(0.0) >= MIN_SCORE);
    format_block(
        "Memories from Anima that may be relevant to this message (stored earlier by the user or their assistants; some may not apply):",
        &bullets(relevant),
    )
}

async fn profile(api: &Api) -> Option<String> {
    let mut lines = Vec::new();
    for category in ["identity", "preference"] {
        let query = [("category", category), ("status", "active"), ("limit", "15")];
        let Ok(listed) = api
            .json(api.request(reqwest::Method::GET, "/api/v1/memories").query(&query))
            .await
        else {
            return None;
        };
        if let Some(memories) = listed.get("memories").and_then(Value::as_array) {
            lines.extend(bullets(memories.iter()));
        }
    }
    format_block(
        "What Anima, the user's long-term memory, holds about them and their preferences. \
         More is available through the anima memory tools:",
        &lines,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_read_from_the_hook_payload() {
        let payload = r#"{"hook_event_name":"UserPromptSubmit","prompt":"  how do I deploy the api?  "}"#;
        assert_eq!(prompt_from_payload(payload).as_deref(), Some("how do I deploy the api?"));
    }

    #[test]
    fn short_prompts_commands_and_garbage_are_skipped() {
        assert!(prompt_from_payload(r#"{"prompt":"yes"}"#).is_none());
        assert!(prompt_from_payload(r#"{"prompt":"/compact the conversation"}"#).is_none());
        assert!(prompt_from_payload("not json").is_none());
        assert!(prompt_from_payload(r#"{"source":"startup"}"#).is_none());
    }

    #[test]
    fn long_content_is_flattened_and_cut() {
        assert_eq!(one_line("a\n  b\tc"), "a b c");
        let long = "x".repeat(MAX_LINE_CHARS + 50);
        let cut = one_line(&long);
        assert_eq!(cut.chars().count(), MAX_LINE_CHARS + 1);
        assert!(cut.ends_with('…'));
    }

    #[test]
    fn empty_results_print_nothing() {
        assert!(format_block("title", &[]).is_none());
        assert_eq!(format_block("t", &["- a".into()]).as_deref(), Some("t\n- a\n"));
    }
}
