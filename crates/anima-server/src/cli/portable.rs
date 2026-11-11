//! `export` and `import`: move memories between machines in one file.
//!
//! The portable format is JSON without embeddings, so it does not depend on
//! the embedding model: the receiving server re-embeds on import. `--full`
//! takes a SQLite snapshot instead, which also carries conversations, plans
//! and history but needs the same embedding model on the other side.

use std::path::PathBuf;
use std::time::Duration;

use serde_json::{json, Value};

use super::client::Api;
use super::Options;

const SQLITE_MAGIC: &[u8] = b"SQLite format 3\0";

pub async fn export(opts: &Options) -> anyhow::Result<()> {
    let api = Api::from_options(opts, Duration::from_secs(600))?;
    let full = opts.has("--full");
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let path = opts
        .positional
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("anima-export-{stamp}.{}", if full { "db" } else { "json" })));

    let mut query = vec![("format", if full { "sqlite" } else { "json" })];
    if !full && !opts.has("--only-namespace") {
        query.push(("all", "true"));
    }
    let res = api
        .send(api.request(reqwest::Method::GET, "/api/v1/backup").query(&query))
        .await?;
    let bytes = res.bytes().await?;

    if full {
        anyhow::ensure!(bytes.starts_with(SQLITE_MAGIC), "the server did not return a SQLite database");
    }
    std::fs::write(&path, &bytes)?;

    if full {
        println!("Saved a full snapshot ({} bytes) to {}", bytes.len(), path.display());
    } else {
        let saved = serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null);
        let count = saved.get("memory_count").and_then(Value::as_u64).unwrap_or(0);
        let skills = saved.get("skills").and_then(Value::as_array).map_or(0, Vec::len);
        println!("Saved {count} memories and {skills} skills to {}", path.display());
    }
    println!("Load it elsewhere with: anima-server import {}", path.display());
    Ok(())
}

pub async fn import(opts: &Options) -> anyhow::Result<()> {
    let path = opts
        .positional
        .first()
        .ok_or_else(|| anyhow::anyhow!("usage: anima-server import <file>"))?;
    let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
    let api = Api::from_options(opts, Duration::from_secs(3600))?;

    if bytes.starts_with(SQLITE_MAGIC) {
        anyhow::ensure!(
            opts.has("--replace"),
            "{path} is a full snapshot: importing it replaces the whole database. \
             Run again with --replace to confirm."
        );
        let res = api
            .json(api.request(reqwest::Method::POST, "/api/v1/restore/sqlite").body(bytes))
            .await?;
        println!("Database replaced. Restart the Anima server to load it.");
        if let Some(backup) = res.get("backup_path").and_then(Value::as_str) {
            println!("The previous database was kept at {backup}");
        }
        return Ok(());
    }

    let backup: Value = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow::anyhow!("{path} is neither an Anima export nor a SQLite snapshot: {e}"))?;
    anyhow::ensure!(
        backup.get("memories").is_some_and(Value::is_array),
        "{path} is not an Anima export (no \"memories\" list)"
    );
    let body = json!({
        "mode": if opts.has("--replace") { "replace" } else { "merge" },
        "preserve_namespaces": !opts.has("--into-namespace"),
        "backup": backup,
    });
    let res = api.post("/api/v1/restore", &body).await?;
    let n = |k: &str| res.get(k).and_then(Value::as_u64).unwrap_or(0);
    println!(
        "Imported {} of {} memories into {} ({} skipped: already present or deleted)",
        n("imported"),
        n("total"),
        api.base(),
        n("skipped"),
    );
    println!(
        "Imported {} skills ({} skipped: same or newer copy already present)",
        n("skills_imported"),
        n("skills_skipped"),
    );
    Ok(())
}
