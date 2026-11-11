//! `skills`: move skills between Anima and `SKILL.md` folders.
//!
//! A skill on disk follows the Agent Skills layout: a folder holding a
//! `SKILL.md` whose YAML frontmatter gives `name` and `description`, followed
//! by the instructions.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};

use super::client::Api;
use super::Options;

const USAGE: &str = "usage: anima-server skills <list | add <path>... | export <dir> | remove <name>>";
const FILE_NAME: &str = "SKILL.md";

/// A skill read from a `SKILL.md`.
#[derive(Debug, PartialEq)]
pub struct ParsedSkill {
    pub name: String,
    pub description: String,
    pub content: String,
}

fn unquote(value: &str) -> String {
    let v = value.trim();
    let quoted = v.len() >= 2
        && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')));
    if !quoted {
        return v.to_string();
    }
    if v.starts_with('"') {
        // Double-quoted YAML scalars use JSON-style escapes.
        serde_json::from_str::<String>(v).unwrap_or_else(|_| v[1..v.len() - 1].to_string())
    } else {
        v[1..v.len() - 1].replace("''", "'")
    }
}

/// Turn any label into a valid skill name: "My Skill_v2" becomes "my-skill-v2".
pub fn slug(label: &str) -> String {
    let mut out = String::new();
    for c in label.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').chars().take(64).collect::<String>().trim_end_matches('-').to_string()
}

/// Parse a `SKILL.md`. `fallback_name` (usually the folder name) is used
/// when the frontmatter has no name.
pub fn parse_skill_md(text: &str, fallback_name: &str) -> anyhow::Result<ParsedSkill> {
    let text = text.trim_start_matches('\u{feff}');
    let mut name = String::new();
    let mut description = String::new();
    let mut body = text;

    if let Some(rest) = text.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            let front = &rest[..end];
            body = rest[end + 4..].trim_start_matches(['\r', '\n']);
            let mut current: Option<&str> = None;
            for raw in front.lines() {
                let indented = raw.starts_with(' ') || raw.starts_with('\t');
                if indented {
                    // Continuation of a folded or multi-line value.
                    if current == Some("description") {
                        if !description.is_empty() {
                            description.push(' ');
                        }
                        description.push_str(raw.trim());
                    }
                    continue;
                }
                let Some((key, value)) = raw.split_once(':') else { continue };
                let value = value.trim();
                // ">" and "|" only announce a block that follows on indented lines.
                let value = if matches!(value, ">" | "|" | ">-" | "|-") { "" } else { value };
                match key.trim() {
                    "name" => {
                        name = unquote(value);
                        current = Some("name");
                    }
                    "description" => {
                        description = unquote(value);
                        current = Some("description");
                    }
                    _ => current = None,
                }
            }
        }
    }

    let name = slug(if name.is_empty() { fallback_name } else { &name });
    anyhow::ensure!(!name.is_empty(), "no usable skill name");
    let content = body.trim().to_string();
    anyhow::ensure!(!content.is_empty(), "skill {name} has no instructions");
    if description.trim().is_empty() {
        // The first line of the body is better than nothing for deciding when to load it.
        description = content
            .lines()
            .map(|l| l.trim_start_matches('#').trim())
            .find(|l| !l.is_empty())
            .unwrap_or(&name)
            .to_string();
    }
    Ok(ParsedSkill { name, description: description.trim().to_string(), content })
}

/// Render a skill as a `SKILL.md`.
pub fn render_skill_md(name: &str, description: &str, content: &str) -> String {
    let description = description.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("---\nname: {name}\ndescription: {}\n---\n\n{}\n", json!(description), content.trim())
}

/// `SKILL.md` files at or under `path`, a few levels deep.
fn find_skill_files(path: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        out.push(path.to_path_buf());
        return;
    }
    let own = path.join(FILE_NAME);
    if own.is_file() {
        out.push(own);
        return;
    }
    if depth == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else { return };
    let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    for dir in dirs {
        find_skill_files(&dir, depth - 1, out);
    }
}

fn fallback_name(file: &Path) -> String {
    let is_skill_md = file.file_name().is_some_and(|n| n.eq_ignore_ascii_case(FILE_NAME));
    let source = if is_skill_md { file.parent() } else { Some(file) };
    source
        .and_then(|p| p.file_stem())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

async fn list(api: &Api) -> anyhow::Result<()> {
    let listed = api.get("/api/v1/skills").await?;
    let skills = listed.get("skills").and_then(Value::as_array).cloned().unwrap_or_default();
    if skills.is_empty() {
        println!("No skills stored.");
    }
    for s in skills {
        let text = |k: &str| s.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let builtin = if s.get("builtin").and_then(Value::as_bool) == Some(true) { " (built in)" } else { "" };
        println!("{}{builtin}  v{}", text("name"), s.get("version").and_then(Value::as_i64).unwrap_or(1));
        println!("    {}", text("description"));
    }
    Ok(())
}

async fn add(api: &Api, paths: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(!paths.is_empty(), "{USAGE}");
    let mut files = Vec::new();
    for p in paths {
        let path = Path::new(p);
        anyhow::ensure!(path.exists(), "{p} does not exist");
        find_skill_files(path, 3, &mut files);
    }
    anyhow::ensure!(!files.is_empty(), "no {FILE_NAME} found");
    let (mut saved, mut failed) = (0, 0);
    for file in files {
        let result = async {
            let text = std::fs::read_to_string(&file)?;
            let skill = parse_skill_md(&text, &fallback_name(&file))?;
            let body = json!({ "description": skill.description, "content": skill.content });
            let path = format!("/api/v1/skills/{}", skill.name);
            let stored = api.json(api.request(reqwest::Method::PUT, &path).json(&body)).await?;
            anyhow::Ok((skill.name, stored.get("version").and_then(Value::as_i64).unwrap_or(1)))
        }
        .await;
        match result {
            Ok((name, version)) => {
                saved += 1;
                println!("  saved   {name} (v{version}) from {}", file.display());
            }
            Err(e) => {
                failed += 1;
                println!("  skipped {}: {e}", file.display());
            }
        }
    }
    println!("{saved} skills saved, {failed} skipped.");
    anyhow::ensure!(saved > 0, "nothing was saved");
    Ok(())
}

async fn export(api: &Api, dir: &str) -> anyhow::Result<()> {
    let listed = api.get("/api/v1/skills").await?;
    let names: Vec<String> = listed
        .get("skills")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|s| s.get("name").and_then(Value::as_str).map(str::to_string)).collect())
        .unwrap_or_default();
    for name in &names {
        let skill = api.get(&format!("/api/v1/skills/{name}")).await?;
        let text = |k: &str| skill.get(k).and_then(Value::as_str).unwrap_or("");
        let folder = Path::new(dir).join(name);
        std::fs::create_dir_all(&folder)?;
        std::fs::write(folder.join(FILE_NAME), render_skill_md(name, text("description"), text("content")))?;
        println!("  wrote {}", folder.join(FILE_NAME).display());
    }
    println!("{} skills written to {dir}", names.len());
    Ok(())
}

pub async fn run(opts: &Options) -> anyhow::Result<()> {
    let api = Api::from_options(opts, Duration::from_secs(60))?;
    let args = &opts.positional;
    match args.first().map(String::as_str) {
        Some("list") | None => list(&api).await,
        Some("add") => add(&api, &args[1..]).await,
        Some("export") => export(&api, args.get(1).ok_or_else(|| anyhow::anyhow!("{USAGE}"))?).await,
        Some("remove") => {
            let name = args.get(1).ok_or_else(|| anyhow::anyhow!("{USAGE}"))?;
            api.json(api.request(reqwest::Method::DELETE, &format!("/api/v1/skills/{name}"))).await?;
            println!("Removed {name}");
            Ok(())
        }
        Some(other) => anyhow::bail!("unknown skills command: {other}\n{USAGE}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_and_body() {
        let md = "---\nname: release-checklist\ndescription: \"Ship a release: tag, build, publish.\"\nlicense: MIT\n---\n\n# Release\n\n1. Tag\n";
        let s = parse_skill_md(md, "ignored").unwrap();
        assert_eq!(s.name, "release-checklist");
        assert_eq!(s.description, "Ship a release: tag, build, publish.");
        assert_eq!(s.content, "# Release\n\n1. Tag");
    }

    #[test]
    fn folded_descriptions_and_missing_names_are_handled() {
        let md = "---\ndescription: >\n  Review a pull request\n  for correctness.\n---\nSteps here.";
        let s = parse_skill_md(md, "PR Review").unwrap();
        assert_eq!(s.name, "pr-review");
        assert_eq!(s.description, "Review a pull request for correctness.");
    }

    #[test]
    fn files_without_frontmatter_use_the_first_line() {
        let s = parse_skill_md("# Deploy the API\n\nRun the script.", "deploy_api").unwrap();
        assert_eq!((s.name.as_str(), s.description.as_str()), ("deploy-api", "Deploy the API"));
        assert!(parse_skill_md("---\nname: x\n---\n", "x").is_err(), "empty body");
        assert!(parse_skill_md("body", "???").is_err(), "no usable name");
    }

    #[test]
    fn render_round_trips_through_parse() {
        let description = "Use when: the user says \"ship it\".";
        let md = render_skill_md("ship", description, "Do the thing.\n");
        let s = parse_skill_md(&md, "").unwrap();
        assert_eq!(s, ParsedSkill { name: "ship".into(), description: description.into(), content: "Do the thing.".into() });
    }

    #[test]
    fn slug_makes_safe_names() {
        assert_eq!(slug("My Skill_v2"), "my-skill-v2");
        assert_eq!(slug("  --weird--  "), "weird");
        assert_eq!(slug("ça"), "a");
        assert!(anima_db::skills::valid_name(&slug(&"Long Name ".repeat(20))));
    }

    #[test]
    fn finds_skill_files_in_nested_folders() {
        let dir = tempfile::tempdir().unwrap();
        for sub in ["a", "group/b"] {
            let folder = dir.path().join(sub);
            std::fs::create_dir_all(&folder).unwrap();
            std::fs::write(folder.join(FILE_NAME), "body").unwrap();
        }
        std::fs::write(dir.path().join("README.md"), "not a skill").unwrap();
        let mut found = Vec::new();
        find_skill_files(dir.path(), 3, &mut found);
        assert_eq!(found.len(), 2);
        assert_eq!(fallback_name(&found[0]), "a");
        assert_eq!(fallback_name(Path::new("notes/deploy.md")), "deploy");
    }
}
