//! Where the Anima server is and how to reach it.
//!
//! Resolution order: command-line flags, environment variables, then
//! `~/.anima/client.json` (written by `connect`). Keeping the address and key
//! in that one file means harness configs never hold the API key.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Options;

pub const DEFAULT_URL: &str = "http://127.0.0.1:3000";
pub const DEFAULT_NAMESPACE: &str = "default";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ClientSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl ClientSettings {
    pub fn path(home: &Path) -> PathBuf {
        home.join(".anima").join("client.json")
    }

    pub fn load(home: &Path) -> Self {
        std::fs::read_to_string(Self::path(home))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, home: &Path) -> anyhow::Result<()> {
        let path = Self::path(home);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(self)? + "\n")?;
        restrict_to_owner(&path);
        Ok(())
    }
}

/// The file can hold an API key: keep other users out where the OS allows it.
#[cfg(unix)]
fn restrict_to_owner(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &Path) {}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// Flags, then environment, then the saved file.
pub fn resolve(opts: &Options) -> ClientSettings {
    let saved = opts
        .home_dir()
        .map(|h| ClientSettings::load(&h))
        .unwrap_or_default();
    ClientSettings {
        url: opts
            .url
            .clone()
            .or_else(|| env("ANIMA_BASE_URL"))
            .or_else(|| env("ANIMA_URL"))
            .or(saved.url),
        api_key: opts.api_key.clone().or_else(|| env("ANIMA_API_KEY")).or(saved.api_key),
        namespace: opts.namespace.clone().or_else(|| env("ANIMA_NAMESPACE")).or(saved.namespace),
    }
}

/// HTTP client for the Anima REST API.
pub struct Api {
    base: String,
    key: Option<String>,
    namespace: String,
    http: reqwest::Client,
}

impl Api {
    pub fn new(settings: &ClientSettings, timeout: Duration) -> anyhow::Result<Self> {
        Ok(Self {
            base: settings
                .url
                .clone()
                .unwrap_or_else(|| DEFAULT_URL.into())
                .trim_end_matches('/')
                .to_string(),
            key: settings.api_key.clone(),
            namespace: settings.namespace.clone().unwrap_or_else(|| DEFAULT_NAMESPACE.into()),
            http: reqwest::Client::builder().timeout(timeout).build()?,
        })
    }

    pub fn from_options(opts: &Options, timeout: Duration) -> anyhow::Result<Self> {
        Self::new(&resolve(opts), timeout)
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .http
            .request(method, format!("{}{path}", self.base))
            .header("X-Anima-Namespace", &self.namespace);
        if let Some(key) = &self.key {
            req = req.bearer_auth(key);
        }
        req
    }

    /// Send a request and decode a JSON body, turning HTTP errors into messages
    /// a person can act on.
    pub async fn send(&self, req: reqwest::RequestBuilder) -> anyhow::Result<reqwest::Response> {
        let res = req.send().await.map_err(|e| {
            if e.is_connect() {
                anyhow::anyhow!(
                    "cannot reach Anima at {}. Start the server, or pass --url.",
                    self.base
                )
            } else {
                anyhow::anyhow!("request to {} failed: {e}", self.base)
            }
        })?;
        let status = res.status();
        if status.is_success() {
            return Ok(res);
        }
        let body = res.text().await.unwrap_or_default();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            anyhow::bail!("Anima rejected the API key. Pass --api-key or set ANIMA_API_KEY.");
        }
        anyhow::bail!("HTTP {}: {}", status.as_u16(), body.trim());
    }

    pub async fn json(&self, req: reqwest::RequestBuilder) -> anyhow::Result<Value> {
        Ok(self.send(req).await?.json().await?)
    }

    pub async fn get(&self, path: &str) -> anyhow::Result<Value> {
        self.json(self.request(reqwest::Method::GET, path)).await
    }

    pub async fn post(&self, path: &str, body: &Value) -> anyhow::Result<Value> {
        self.json(self.request(reqwest::Method::POST, path).json(body)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(ClientSettings::load(dir.path()), ClientSettings::default());
        let s = ClientSettings {
            url: Some("https://memory.example".into()),
            api_key: Some("secret".into()),
            namespace: None,
        };
        s.save(dir.path()).unwrap();
        assert_eq!(ClientSettings::load(dir.path()), s);
    }

    #[test]
    fn flags_win_over_saved_settings() {
        let dir = tempfile::tempdir().unwrap();
        ClientSettings { url: Some("http://saved".into()), api_key: None, namespace: Some("work".into()) }
            .save(dir.path())
            .unwrap();
        let opts = Options {
            url: Some("http://flag".into()),
            home: Some(dir.path().to_path_buf()),
            ..Default::default()
        };
        assert_eq!(resolve(&opts).url.as_deref(), Some("http://flag"));
    }
}
