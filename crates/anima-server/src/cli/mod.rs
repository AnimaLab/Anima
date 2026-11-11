//! Client subcommands of the `anima-server` binary.
//!
//! These talk to a running Anima server over HTTP and never start one:
//!
//! - `connect <harness>`: register Anima with an AI harness
//! - `disconnect <harness>`: undo `connect`
//! - `export [file]` / `import <file>`: move memories between machines
//! - `mcp`: serve the memory tools over MCP stdio
//! - `recall`: print memories for a harness hook

mod client;
mod connect;
mod mcp;
mod portable;
mod recall;

use std::path::PathBuf;

#[cfg(test)]
pub use client::ClientSettings;

const USAGE: &str = "\
Anima memory server

Usage:
  anima-server [config.toml]            Start the server
  anima-server connect <harness>        Give a harness access to your memory
  anima-server disconnect <harness>     Remove Anima from a harness
  anima-server export [file]            Save every memory to one file
  anima-server import <file>            Load memories from an export
  anima-server mcp                      Serve memory tools over MCP stdio
  anima-server recall                   Print memories for a harness hook
  anima-server --install | --uninstall | --service-status

Harnesses: claude, claude-desktop, codex, cursor, hermes, all, print

Connection options (also read from ANIMA_BASE_URL, ANIMA_API_KEY,
ANIMA_NAMESPACE, then from ~/.anima/client.json):
  --url <url>          Server address (default http://127.0.0.1:3000)
  --api-key <key>      API key, when the server requires one
  --namespace <name>   Memory namespace (default: default)

connect / disconnect:
  --dry-run            Show what would change without writing
  --no-hooks           Skip automatic recall hooks
  --no-instructions    Skip the instruction block

export:
  --full               Full SQLite snapshot instead of portable JSON
  --only-namespace     Export only the selected namespace

import:
  --replace            Replace existing memories instead of merging
  --into-namespace     Put everything in the selected namespace
";

/// Parsed command-line options shared by the subcommands.
#[derive(Debug, Default)]
pub struct Options {
    pub positional: Vec<String>,
    pub url: Option<String>,
    pub api_key: Option<String>,
    pub namespace: Option<String>,
    /// Stands in for the user's home directory. Used by tests.
    pub home: Option<PathBuf>,
    pub flags: Vec<String>,
}

impl Options {
    pub fn parse(args: &[String]) -> anyhow::Result<Self> {
        let mut o = Options::default();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let mut value = |name: &str| {
                it.next()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("{name} needs a value"))
            };
            match a.as_str() {
                "--url" => o.url = Some(value("--url")?),
                "--api-key" => o.api_key = Some(value("--api-key")?),
                "--namespace" => o.namespace = Some(value("--namespace")?),
                "--home" => o.home = Some(PathBuf::from(value("--home")?)),
                f if f.starts_with("--") => o.flags.push(f.to_string()),
                _ => o.positional.push(a.clone()),
            }
        }
        Ok(o)
    }

    pub fn has(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }

    /// The user's home directory, or the `--home` override.
    pub fn home_dir(&self) -> anyhow::Result<PathBuf> {
        if let Some(h) = &self.home {
            return Ok(h.clone());
        }
        std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("cannot find the home directory"))
    }
}

/// Run a client subcommand if `args` names one. `None` means "start the server".
pub async fn dispatch(args: &[String]) -> Option<anyhow::Result<()>> {
    let command = args.first()?.as_str();
    let rest = &args[1..];
    let result = match command {
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(())
        }
        "connect" => run(rest, |o| async move { connect::run(&o, true) }).await,
        "disconnect" => run(rest, |o| async move { connect::run(&o, false) }).await,
        "export" => run(rest, |o| async move { portable::export(&o).await }).await,
        "import" => run(rest, |o| async move { portable::import(&o).await }).await,
        "mcp" => run(rest, |o| async move { mcp::serve(&o).await }).await,
        "recall" => run(rest, |o| async move { recall::run(&o).await }).await,
        _ => return None,
    };
    Some(result)
}

async fn run<F, Fut>(args: &[String], f: F) -> anyhow::Result<()>
where
    F: FnOnce(Options) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    f(Options::parse(args)?).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_values_flags_and_positionals() {
        let o = Options::parse(&args(&["claude", "--url", "http://h:1", "--dry-run", "--api-key", "k"])).unwrap();
        assert_eq!(o.positional, vec!["claude"]);
        assert_eq!(o.url.as_deref(), Some("http://h:1"));
        assert_eq!(o.api_key.as_deref(), Some("k"));
        assert!(o.has("--dry-run"));
        assert!(!o.has("--full"));
    }

    #[test]
    fn missing_value_is_an_error() {
        assert!(Options::parse(&args(&["--url"])).is_err());
    }

    #[tokio::test]
    async fn config_path_is_not_a_subcommand() {
        assert!(dispatch(&args(&["config.toml"])).await.is_none());
        assert!(dispatch(&[]).await.is_none());
    }
}
