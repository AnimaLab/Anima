//! `recall`: print a briefing for a harness hook.
//!
//! A harness runs this at session start, or on each prompt, and adds what it
//! prints to the model's context. A hook must never get in the user's way,
//! so every failure (server down, slow, bad input) prints nothing and exits 0.
//!
//! Without flags it prints the session briefing: the user's profile, what is
//! relevant to the current project, and the list of stored skills. With
//! `--prompt` it prints the memories matching the prompt in the hook payload.
//! The selection itself happens on the server (`POST /api/v1/briefing`).

use std::io::{IsTerminal, Read};
use std::time::Duration;

use serde_json::{json, Value};

use super::client::Api;
use super::Options;

const TIMEOUT: Duration = Duration::from_secs(4);

pub async fn run(opts: &Options) -> anyhow::Result<()> {
    let Ok(api) = Api::from_options(opts, TIMEOUT) else {
        return Ok(());
    };
    let payload = read_payload();
    let request = if opts.has("--prompt") {
        match prompt_from_payload(&payload) {
            Some(prompt) => json!({ "mode": "prompt", "query": prompt }),
            None => return Ok(()),
        }
    } else {
        json!({ "mode": "session", "project": project(opts, &payload) })
    };
    if let Ok(briefing) = api.post("/api/v1/briefing", &request).await {
        if let Some(text) = briefing.get("text").and_then(Value::as_str) {
            print!("{text}");
        }
    }
    Ok(())
}

/// The hook payload on stdin. Run by hand from a terminal there is none, and
/// waiting for one would hang.
fn read_payload() -> String {
    let mut stdin = std::io::stdin();
    if stdin.is_terminal() {
        return String::new();
    }
    let mut payload = String::new();
    let _ = stdin.read_to_string(&mut payload);
    payload
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

/// Last folder of a path. Both separators are accepted on every platform,
/// since the path comes from the harness and not from this OS's own APIs.
fn folder_name(path: &str) -> Option<String> {
    path.trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .map(str::to_string)
        .filter(|n| !n.is_empty() && !n.ends_with(':'))
}

/// The project a session is about: `--project=<name>`, else the working
/// folder the harness reported, else the current folder.
fn project(opts: &Options, payload: &str) -> Option<String> {
    if let Some(name) = opts.flags.iter().find_map(|f| f.strip_prefix("--project=")) {
        return Some(name.to_string()).filter(|n| !n.is_empty());
    }
    let from_payload = serde_json::from_str::<Value>(payload)
        .ok()
        .and_then(|v| v.get("cwd").and_then(Value::as_str).and_then(folder_name));
    from_payload.or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|d| folder_name(&d.to_string_lossy()))
    })
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
    fn project_comes_from_the_flag_then_the_payload_folder() {
        let payload = r#"{"cwd":"C:\\Users\\me\\code\\billing-api\\","source":"startup"}"#;
        let none = Options::default();
        assert_eq!(project(&none, payload).as_deref(), Some("billing-api"));
        assert_eq!(project(&none, r#"{"cwd":"/home/me/work/anima"}"#).as_deref(), Some("anima"));
        let flagged = Options { flags: vec!["--project=other".into()], ..Default::default() };
        assert_eq!(project(&flagged, payload).as_deref(), Some("other"));
    }
}
