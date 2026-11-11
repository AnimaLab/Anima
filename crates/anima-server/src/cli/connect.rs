//! `connect` / `disconnect`: register Anima with an AI harness.
//!
//! For each harness this writes up to three things, all idempotent and all
//! removable with `disconnect`:
//!
//! 1. an MCP server entry that runs `anima-server mcp`,
//! 2. an instruction block telling the model when to read and write memory,
//! 3. where the harness supports it, hooks that inject memories automatically.
//!
//! The server address and API key go to `~/.anima/client.json`, never into
//! the harness files.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use super::client::{resolve, ClientSettings, DEFAULT_URL};
use super::Options;

const SERVER_NAME: &str = "anima";
const BLOCK_START: &str = "<!-- anima:start -->";
const BLOCK_END: &str = "<!-- anima:end -->";
/// Marks hook commands as ours so they can be replaced or removed.
const HOOK_MARK: &str = "--hook=anima";

const INSTRUCTIONS: &str = "\
## Long-term memory (Anima)

You have persistent memory through the `anima` MCP server. It is shared by every
assistant and machine the user works with, so what you save here is what the
next assistant will know.

- Before starting a task, call `memory_search` with the topic to recall relevant
  preferences, decisions and context.
- When the user states a durable fact, preference, decision or correction, save
  it with `memory_add`: one fact per memory, written so it stands on its own.
- If a stored memory is wrong or outdated, fix it with `memory_update` instead of
  adding a contradicting one.
- Do not store secrets, credentials or details that only matter to the current task.";

const HARNESSES: &[&str] = &["claude", "claude-desktop", "codex", "cursor", "hermes"];

struct Ctx<'a> {
    home: PathBuf,
    exe: String,
    connect: bool,
    dry_run: bool,
    hooks: bool,
    instructions: bool,
    report: &'a mut Vec<String>,
}

impl Ctx<'_> {
    /// Write `new` to `path` unless it already holds it. The first `connect`
    /// to touch a file keeps a copy of the original next to it.
    fn write(&mut self, path: &Path, new: &str, what: &str) -> anyhow::Result<()> {
        let old = std::fs::read_to_string(path).ok();
        if old.as_deref() == Some(new) {
            self.report.push(format!("  unchanged  {} ({what})", path.display()));
            return Ok(());
        }
        if !self.dry_run {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            if let (Some(old), true) = (&old, self.connect) {
                let backup = PathBuf::from(format!("{}.before-anima", path.display()));
                if !backup.exists() {
                    std::fs::write(&backup, old)?;
                }
            }
            std::fs::write(path, new)?;
        }
        let verb = if self.dry_run { "would write" } else { "wrote     " };
        self.report.push(format!("  {verb} {} ({what})", path.display()));
        Ok(())
    }

    /// Edit a JSON file through `edit`. A missing file starts as `{}`; an
    /// unreadable one is left alone.
    fn edit_json(&mut self, path: &Path, what: &str, edit: impl FnOnce(&mut Map<String, Value>)) -> anyhow::Result<()> {
        let mut root = match std::fs::read_to_string(path) {
            Ok(s) if !s.trim().is_empty() => serde_json::from_str::<Value>(&s)
                .map_err(|e| anyhow::anyhow!("{} is not valid JSON, left untouched: {e}", path.display()))?,
            Ok(_) | Err(_) if !self.connect && !path.exists() => return Ok(()),
            _ => json!({}),
        };
        let obj = root
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("{} is not a JSON object, left untouched", path.display()))?;
        edit(obj);
        self.write(path, &(serde_json::to_string_pretty(&root)? + "\n"), what)
    }

    fn edit_text(&mut self, path: &Path, what: &str, edit: impl FnOnce(&str) -> String) -> anyhow::Result<()> {
        if !self.connect && !path.exists() {
            return Ok(());
        }
        let old = std::fs::read_to_string(path).unwrap_or_default();
        self.write(path, &edit(&old), what)
    }

    fn mcp_entry(&self, typed: bool) -> Value {
        let mut entry = Map::new();
        if typed {
            entry.insert("type".into(), json!("stdio"));
        }
        entry.insert("command".into(), json!(self.exe));
        entry.insert("args".into(), json!(["mcp"]));
        Value::Object(entry)
    }

    /// Add or remove `mcpServers.anima` in a JSON config.
    fn mcp_json(&mut self, path: &Path, typed: bool) -> anyhow::Result<()> {
        let (connect, entry) = (self.connect, self.mcp_entry(typed));
        self.edit_json(path, "MCP server", |root| {
            if connect {
                let servers = root.entry("mcpServers").or_insert_with(|| json!({}));
                if let Some(servers) = servers.as_object_mut() {
                    servers.insert(SERVER_NAME.into(), entry);
                }
            } else if let Some(servers) = root.get_mut("mcpServers").and_then(Value::as_object_mut) {
                servers.remove(SERVER_NAME);
            }
        })
    }

    fn instruction_file(&mut self, path: &Path) -> anyhow::Result<()> {
        if !self.instructions {
            return Ok(());
        }
        let connect = self.connect;
        self.edit_text(path, "instructions", |old| {
            let rest = remove_block(old);
            if connect {
                upsert_block(&rest)
            } else {
                rest
            }
        })
    }
}

// ── Marked text blocks ────────────────────────────────────────────

fn remove_block(text: &str) -> String {
    let (Some(start), Some(end)) = (text.find(BLOCK_START), text.find(BLOCK_END)) else {
        return text.to_string();
    };
    if end < start {
        return text.to_string();
    }
    let before = text[..start].trim_end();
    let after = text[end + BLOCK_END.len()..].trim_start();
    match (before.is_empty(), after.is_empty()) {
        (true, true) => String::new(),
        (true, false) => after.to_string(),
        (false, true) => format!("{before}\n"),
        (false, false) => format!("{before}\n\n{after}"),
    }
}

fn upsert_block(text: &str) -> String {
    let rest = text.trim_end();
    let block = format!("{BLOCK_START}\n{INSTRUCTIONS}\n{BLOCK_END}\n");
    if rest.is_empty() {
        block
    } else {
        format!("{rest}\n\n{block}")
    }
}

// ── Codex: TOML ───────────────────────────────────────────────────

fn is_anima_toml_header(line: &str) -> bool {
    let header = line.trim();
    header == "[mcp_servers.anima]" || header.starts_with("[mcp_servers.anima.")
}

/// Drop the `[mcp_servers.anima]` table and its sub-tables, keeping the rest
/// of the file byte for byte.
fn remove_toml_server(text: &str) -> String {
    let mut out = Vec::new();
    let mut skipping = false;
    for line in text.lines() {
        if line.trim_start().starts_with('[') {
            skipping = is_anima_toml_header(line);
        }
        if !skipping {
            out.push(line);
        }
    }
    let joined = out.join("\n");
    let trimmed = joined.trim_end();
    if trimmed.is_empty() { String::new() } else { format!("{trimmed}\n") }
}

fn upsert_toml_server(text: &str, exe: &str) -> String {
    let rest = remove_toml_server(text);
    // JSON string escaping is valid TOML basic-string escaping for paths.
    let table = format!("[mcp_servers.anima]\ncommand = {}\nargs = [\"mcp\"]\n", json!(exe));
    if rest.is_empty() { table } else { format!("{rest}\n{table}") }
}

// ── Hermes: YAML ──────────────────────────────────────────────────

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Remove the `anima:` entry under the top-level `mcp_servers:` key.
fn remove_yaml_server(text: &str) -> String {
    let mut out = Vec::new();
    let mut in_servers = false;
    let mut skipping = false;
    for line in text.lines() {
        let blank = line.trim().is_empty();
        if !blank && indent_of(line) == 0 {
            in_servers = line.trim_end() == "mcp_servers:";
            skipping = false;
        } else if in_servers && !blank && indent_of(line) <= 2 {
            skipping = line.trim() == "anima:";
        }
        if !skipping {
            out.push(line);
        }
    }
    let joined = out.join("\n");
    let trimmed = joined.trim_end();
    if trimmed.is_empty() { String::new() } else { format!("{trimmed}\n") }
}

fn upsert_yaml_server(text: &str, exe: &str) -> String {
    let rest = remove_yaml_server(text);
    let entry = format!(
        "  anima:\n    command: '{}'\n    args: [\"mcp\"]",
        exe.replace('\'', "''")
    );
    let mut lines: Vec<String> = rest.lines().map(str::to_string).collect();
    match lines.iter().position(|l| l.trim_end() == "mcp_servers:") {
        Some(at) => lines.insert(at + 1, entry),
        None => {
            if !lines.is_empty() {
                lines.push(String::new());
            }
            lines.push("mcp_servers:".into());
            lines.push(entry);
        }
    }
    lines.join("\n") + "\n"
}

// ── Claude Code hooks ─────────────────────────────────────────────

fn is_anima_hook_group(group: &Value) -> bool {
    group
        .get("hooks")
        .and_then(Value::as_array)
        .is_some_and(|hooks| {
            hooks.iter().any(|h| {
                h.get("command").and_then(Value::as_str).is_some_and(|c| c.contains(HOOK_MARK))
            })
        })
}

fn set_claude_hooks(root: &mut Map<String, Value>, exe: &str, connect: bool) {
    // Forward slashes survive both cmd and POSIX shells inside double quotes.
    let exe = exe.replace('\\', "/");
    let wanted = [
        ("SessionStart", format!("\"{exe}\" recall {HOOK_MARK}")),
        ("UserPromptSubmit", format!("\"{exe}\" recall --prompt {HOOK_MARK}")),
    ];
    if connect && !root.get("hooks").is_some_and(Value::is_object) {
        root.insert("hooks".into(), json!({}));
    }
    let Some(hooks) = root.get_mut("hooks").and_then(Value::as_object_mut) else {
        return;
    };
    for (event, command) in wanted {
        let mut groups: Vec<Value> = hooks
            .get(event)
            .and_then(Value::as_array)
            .map(|g| g.iter().filter(|g| !is_anima_hook_group(g)).cloned().collect())
            .unwrap_or_default();
        if connect {
            groups.push(json!({ "hooks": [{ "type": "command", "command": command, "timeout": 10 }] }));
        }
        if groups.is_empty() {
            hooks.remove(event);
        } else {
            hooks.insert(event.into(), Value::Array(groups));
        }
    }
    if hooks.is_empty() {
        root.remove("hooks");
    }
}

// ── Harnesses ─────────────────────────────────────────────────────

fn claude_desktop_dir(home: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        home.join("AppData").join("Roaming").join("Claude")
    } else if cfg!(target_os = "macos") {
        home.join("Library").join("Application Support").join("Claude")
    } else {
        home.join(".config").join("Claude")
    }
}

fn harness_dir(home: &Path, harness: &str) -> PathBuf {
    match harness {
        "claude" => home.join(".claude"),
        "claude-desktop" => claude_desktop_dir(home),
        "codex" => home.join(".codex"),
        "cursor" => home.join(".cursor"),
        _ => home.join(".hermes"),
    }
}

fn apply(ctx: &mut Ctx, harness: &str) -> anyhow::Result<()> {
    let home = ctx.home.clone();
    let dir = harness_dir(&home, harness);
    let (connect, exe) = (ctx.connect, ctx.exe.clone());
    match harness {
        "claude" => {
            ctx.mcp_json(&home.join(".claude.json"), true)?;
            ctx.instruction_file(&dir.join("CLAUDE.md"))?;
            if ctx.hooks {
                ctx.edit_json(&dir.join("settings.json"), "recall hooks", |root| {
                    set_claude_hooks(root, &exe, connect)
                })?;
            }
        }
        "claude-desktop" => ctx.mcp_json(&dir.join("claude_desktop_config.json"), false)?,
        "cursor" => ctx.mcp_json(&dir.join("mcp.json"), false)?,
        "codex" => {
            ctx.edit_text(&dir.join("config.toml"), "MCP server", |old| {
                if connect { upsert_toml_server(old, &exe) } else { remove_toml_server(old) }
            })?;
            ctx.instruction_file(&dir.join("AGENTS.md"))?;
        }
        "hermes" => {
            ctx.edit_text(&dir.join("config.yaml"), "MCP server", |old| {
                if connect { upsert_yaml_server(old, &exe) } else { remove_yaml_server(old) }
            })?;
        }
        other => anyhow::bail!("unknown harness: {other}. Choose one of: {}, all, print", HARNESSES.join(", ")),
    }
    Ok(())
}

fn print_snippet(exe: &str) {
    let entry = json!({ "mcpServers": { SERVER_NAME: { "command": exe, "args": ["mcp"] } } });
    println!("Add this MCP server to any harness that speaks MCP over stdio:\n");
    println!("{}\n", serde_json::to_string_pretty(&entry).unwrap_or_default());
    println!("Then give the model these instructions (system prompt, AGENTS.md or similar):\n");
    println!("{INSTRUCTIONS}");
}

pub fn run(opts: &Options, connect: bool) -> anyhow::Result<()> {
    let verb = if connect { "connect" } else { "disconnect" };
    let target = opts
        .positional
        .first()
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("usage: anima-server {verb} <{}|all|print>", HARNESSES.join("|")))?;
    let home = opts.home_dir()?;
    let exe = std::env::current_exe()?.display().to_string();

    if target == "print" {
        print_snippet(&exe);
        return Ok(());
    }

    let targets: Vec<&str> = if target == "all" {
        let found: Vec<&str> = HARNESSES
            .iter()
            .copied()
            .filter(|h| harness_dir(&home, h).is_dir())
            .collect();
        anyhow::ensure!(!found.is_empty(), "no supported harness found on this machine");
        found
    } else {
        vec![target]
    };

    let dry_run = opts.has("--dry-run");
    let mut report = Vec::new();
    for harness in &targets {
        report.push(format!("{harness}:"));
        let mut ctx = Ctx {
            home: home.clone(),
            exe: exe.clone(),
            connect,
            dry_run,
            hooks: !opts.has("--no-hooks"),
            instructions: !opts.has("--no-instructions"),
            report: &mut report,
        };
        apply(&mut ctx, harness)?;
    }

    // Remember where the server is, so the harness entries need no secrets.
    let settings = resolve(opts);
    if connect && !dry_run && settings != ClientSettings::load(&home) && settings != ClientSettings::default() {
        settings.save(&home)?;
        report.push(format!("saved connection settings to {}", ClientSettings::path(&home).display()));
    }

    for line in &report {
        println!("{line}");
    }
    if connect {
        println!(
            "\nMemory server: {}. Restart the harness to load Anima.",
            settings.url.as_deref().unwrap_or(DEFAULT_URL)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXE: &str = r"C:\Program Files\Anima\anima-server.exe";

    fn opts(home: &Path, harness: &str, extra: &[&str]) -> Options {
        let mut o = Options {
            positional: vec![harness.to_string()],
            home: Some(home.to_path_buf()),
            ..Default::default()
        };
        o.flags = extra.iter().map(|s| s.to_string()).collect();
        o
    }

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn claude_connect_keeps_existing_settings_and_is_idempotent() {
        let home = tempfile::tempdir().unwrap();
        let claude_json = home.path().join(".claude.json");
        std::fs::write(&claude_json, r#"{"numStartups":3,"mcpServers":{"other":{"command":"x"}}}"#).unwrap();
        let settings = home.path().join(".claude").join("settings.json");
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        std::fs::write(&settings, r#"{"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"mine"}]}]}}"#).unwrap();

        run(&opts(home.path(), "claude", &[]), true).unwrap();
        let first = (read(&claude_json), read(&settings));
        run(&opts(home.path(), "claude", &[]), true).unwrap();
        assert_eq!(first, (read(&claude_json), read(&settings)), "second connect must change nothing");

        let root: Value = serde_json::from_str(&first.0).unwrap();
        assert_eq!(root["numStartups"], 3);
        assert_eq!(root["mcpServers"]["other"]["command"], "x");
        assert_eq!(root["mcpServers"]["anima"]["args"], json!(["mcp"]));

        let hooks: Value = serde_json::from_str(&first.1).unwrap();
        let start = hooks["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(start.len(), 2, "the user's own hook stays");
        assert_eq!(start[0]["hooks"][0]["command"], "mine");
        assert!(hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .ends_with("recall --prompt --hook=anima"));

        let md = read(&home.path().join(".claude").join("CLAUDE.md"));
        assert_eq!(md.matches(BLOCK_START).count(), 1);
        assert!(claude_json.with_file_name(".claude.json.before-anima").exists());
    }

    #[test]
    fn claude_disconnect_restores_what_was_there() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join(".claude");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("CLAUDE.md"), "# Mine\n\nKeep this.\n").unwrap();
        std::fs::write(dir.join("settings.json"), "{\n  \"model\": \"opus\"\n}\n").unwrap();

        run(&opts(home.path(), "claude", &[]), true).unwrap();
        run(&opts(home.path(), "claude", &[]), false).unwrap();

        assert_eq!(read(&dir.join("CLAUDE.md")), "# Mine\n\nKeep this.\n");
        assert_eq!(read(&dir.join("settings.json")), "{\n  \"model\": \"opus\"\n}\n");
        let root: Value = serde_json::from_str(&read(&home.path().join(".claude.json"))).unwrap();
        assert!(root["mcpServers"].get("anima").is_none());
    }

    #[test]
    fn dry_run_and_opt_outs_write_nothing_extra() {
        let home = tempfile::tempdir().unwrap();
        run(&opts(home.path(), "claude", &["--dry-run"]), true).unwrap();
        assert!(!home.path().join(".claude.json").exists());

        run(&opts(home.path(), "claude", &["--no-hooks", "--no-instructions"]), true).unwrap();
        assert!(home.path().join(".claude.json").exists());
        assert!(!home.path().join(".claude").join("settings.json").exists());
        assert!(!home.path().join(".claude").join("CLAUDE.md").exists());
    }

    #[test]
    fn invalid_json_is_left_untouched() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join(".cursor").join("mcp.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{ not json").unwrap();
        assert!(run(&opts(home.path(), "cursor", &[]), true).is_err());
        assert_eq!(read(&path), "{ not json");
    }

    #[test]
    fn codex_toml_table_is_replaced_not_duplicated() {
        let existing = "model = \"gpt-5\"\n\n[mcp_servers.anima]\ncommand = \"old\"\n\n[mcp_servers.anima.env]\nA = \"1\"\n\n[mcp_servers.other]\ncommand = \"keep\"\n";
        let updated = upsert_toml_server(existing, EXE);
        assert_eq!(updated.matches("[mcp_servers.anima]").count(), 1);
        assert!(!updated.contains("\"old\"") && !updated.contains("A = \"1\""));
        assert!(updated.contains("model = \"gpt-5\"") && updated.contains("command = \"keep\""));
        assert!(updated.contains(r#"command = "C:\\Program Files\\Anima\\anima-server.exe""#));
        let parsed: toml::Value = toml::from_str(&updated).unwrap();
        assert_eq!(parsed["mcp_servers"]["anima"]["command"].as_str(), Some(EXE));
        assert_eq!(remove_toml_server(&updated), "model = \"gpt-5\"\n\n[mcp_servers.other]\ncommand = \"keep\"\n");
    }

    #[test]
    fn hermes_yaml_entry_goes_under_existing_key() {
        let existing = "model: hermes-4\nmcp_servers:\n  files:\n    command: npx\ntools:\n  - web\n";
        let updated = upsert_yaml_server(existing, EXE);
        assert_eq!(
            updated,
            "model: hermes-4\nmcp_servers:\n  anima:\n    command: 'C:\\Program Files\\Anima\\anima-server.exe'\n    args: [\"mcp\"]\n  files:\n    command: npx\ntools:\n  - web\n"
        );
        assert_eq!(upsert_yaml_server(&updated, EXE), updated);
        assert_eq!(remove_yaml_server(&updated), existing);

        let fresh = upsert_yaml_server("", EXE);
        assert!(fresh.starts_with("mcp_servers:\n  anima:\n"));
    }

    #[test]
    fn instruction_block_round_trips() {
        let with = upsert_block("# Notes\n");
        assert!(with.starts_with("# Notes\n\n<!-- anima:start -->"));
        assert_eq!(remove_block(&with), "# Notes\n");
        assert_eq!(remove_block(&upsert_block("")), "");
    }

    #[test]
    fn all_needs_at_least_one_harness() {
        let home = tempfile::tempdir().unwrap();
        assert!(run(&opts(home.path(), "all", &[]), true).is_err());
        std::fs::create_dir_all(home.path().join(".codex")).unwrap();
        run(&opts(home.path(), "all", &[]), true).unwrap();
        assert!(read(&home.path().join(".codex").join("config.toml")).contains("[mcp_servers.anima]"));
        assert!(!home.path().join(".claude.json").exists());
    }

    #[test]
    fn unknown_harness_is_rejected() {
        let home = tempfile::tempdir().unwrap();
        assert!(run(&opts(home.path(), "emacs", &[]), true).is_err());
    }
}
