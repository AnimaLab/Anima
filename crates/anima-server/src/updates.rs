//! Checks GitHub for a newer release of Anima, at most once a day, and keeps
//! the answer in `~/.anima/update_check.json` for the web UI and `--version`.
//! It only reads the public release list; nothing is downloaded or installed.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::app::AppState;

pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const LATEST_RELEASE_URL: &str = "https://api.github.com/repos/AnimaLab/Anima/releases/latest";
const CHECK_EVERY: Duration = Duration::from_secs(24 * 60 * 60);
const CACHE_FILE: &str = "update_check.json";

/// The last answer from GitHub, as cached on disk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LatestRelease {
    /// Version of the latest release, without the leading `v`.
    pub version: String,
    /// Release notes page.
    pub url: String,
    /// RFC 3339 time of the check.
    pub checked_at: String,
}

/// What `GET /api/v1/updates` reports.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateStatus {
    pub enabled: bool,
    pub current: String,
    pub latest: Option<String>,
    pub url: Option<String>,
    pub checked_at: Option<String>,
    pub update_available: bool,
}

impl UpdateStatus {
    pub fn new(enabled: bool, latest: Option<&LatestRelease>) -> Self {
        Self {
            enabled,
            current: CURRENT_VERSION.to_string(),
            latest: latest.map(|l| l.version.clone()),
            url: latest.map(|l| l.url.clone()),
            checked_at: latest.map(|l| l.checked_at.clone()),
            update_available: latest.is_some_and(|l| is_newer(&l.version, CURRENT_VERSION)),
        }
    }
}

/// `major.minor.patch` of a version such as `v0.4.0`; a pre-release or build
/// suffix (`-rc1`, `+abc`) makes it unparsable, so it is never offered.
fn parse(version: &str) -> Option<(u64, u64, u64)> {
    let v = version.trim().trim_start_matches('v');
    let mut parts = v.split('.');
    let triple = (
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    );
    parts.next().is_none().then_some(triple)
}

/// Whether `latest` is a newer release than `current`.
pub fn is_newer(latest: &str, current: &str) -> bool {
    match (parse(latest), parse(current)) {
        (Some(l), Some(c)) => l > c,
        _ => false,
    }
}

pub fn cache_path(anima_dir: &Path) -> PathBuf {
    anima_dir.join(CACHE_FILE)
}

/// The cached answer, if any and readable.
pub fn read_cache(anima_dir: &Path) -> Option<LatestRelease> {
    let raw = std::fs::read_to_string(cache_path(anima_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

fn write_cache(anima_dir: &Path, latest: &LatestRelease) -> std::io::Result<()> {
    std::fs::create_dir_all(anima_dir)?;
    let json = serde_json::to_string_pretty(latest).map_err(std::io::Error::other)?;
    std::fs::write(cache_path(anima_dir), json)
}

/// Whether a cached answer is older than a day (or has no usable time).
fn is_stale(latest: &LatestRelease, now: chrono::DateTime<chrono::Utc>) -> bool {
    chrono::DateTime::parse_from_rfc3339(&latest.checked_at)
        .map(|t| now.signed_duration_since(t).to_std().map_or(false, |age| age >= CHECK_EVERY))
        .unwrap_or(true)
}

#[derive(Deserialize)]
struct GithubRelease {
    tag_name: String,
    html_url: String,
}

/// Parse the GitHub `releases/latest` body (the endpoint already leaves out
/// drafts and pre-releases).
fn parse_release(body: &str, now: chrono::DateTime<chrono::Utc>) -> Option<LatestRelease> {
    let release: GithubRelease = serde_json::from_str(body).ok()?;
    let version = release.tag_name.trim().trim_start_matches('v').to_string();
    parse(&version)?;
    Some(LatestRelease { version, url: release.html_url, checked_at: now.to_rfc3339() })
}

async fn fetch_latest(client: &reqwest::Client) -> Result<LatestRelease, String> {
    let resp = client
        .get(LATEST_RELEASE_URL)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("GitHub answered {}", resp.status()));
    }
    let body = resp.text().await.map_err(|e| e.to_string())?;
    parse_release(&body, chrono::Utc::now()).ok_or_else(|| "unexpected release format".into())
}

/// Check now unless the cached answer is less than a day old, then every day.
pub fn spawn_update_check(state: Arc<AppState>, anima_dir: PathBuf) {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .user_agent(format!("anima-server/{CURRENT_VERSION}"))
            .build()
            .unwrap_or_default();
        loop {
            let cached = read_cache(&anima_dir);
            let due = cached.as_ref().map_or(true, |c| is_stale(c, chrono::Utc::now()));
            if due {
                match fetch_latest(&client).await {
                    Ok(latest) => {
                        if is_newer(&latest.version, CURRENT_VERSION) {
                            tracing::info!(
                                domain = "updates",
                                "Anima v{} is available (running v{CURRENT_VERSION}): {}",
                                latest.version,
                                latest.url
                            );
                        }
                        if let Err(e) = write_cache(&anima_dir, &latest) {
                            tracing::debug!(domain = "updates", "Could not save the update check: {e}");
                        }
                        *state.latest_release.write().await = Some(latest);
                    }
                    Err(e) => tracing::debug!(domain = "updates", "Update check failed: {e}"),
                }
            } else if let Some(c) = cached {
                *state.latest_release.write().await = Some(c);
            }
            tokio::time::sleep(CHECK_EVERY).await;
        }
    });
}

/// The line `--version` prints after the version, when the last check found
/// a newer release.
pub fn notice(anima_dir: &Path) -> Option<String> {
    let latest = read_cache(anima_dir)?;
    is_newer(&latest.version, CURRENT_VERSION)
        .then(|| format!("Anima v{} is available: {}", latest.version, latest.url))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_compares_numbers_not_strings() {
        assert!(is_newer("v0.4.0", "0.3.0"));
        assert!(is_newer("0.10.0", "0.9.9"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(!is_newer("0.3.0", "0.3.0"));
        assert!(!is_newer("0.2.9", "0.3.0"));
    }

    #[test]
    fn unparsable_or_prerelease_versions_are_never_offered() {
        assert!(!is_newer("v0.4.0-rc1", "0.3.0"));
        assert!(!is_newer("latest", "0.3.0"));
        assert!(!is_newer("0.4", "0.3.0"));
        assert!(!is_newer("0.4.0.1", "0.3.0"));
    }

    #[test]
    fn release_body_is_read_and_the_v_dropped() {
        let now = chrono::Utc::now();
        let body = r#"{"tag_name":"v0.4.0","html_url":"https://github.com/AnimaLab/Anima/releases/tag/v0.4.0","draft":false}"#;
        let latest = parse_release(body, now).unwrap();
        assert_eq!(latest.version, "0.4.0");
        assert_eq!(latest.url, "https://github.com/AnimaLab/Anima/releases/tag/v0.4.0");
        assert!(parse_release(r#"{"tag_name":"nightly","html_url":"x"}"#, now).is_none());
        assert!(parse_release("not json", now).is_none());
    }

    #[test]
    fn a_check_is_due_after_a_day_or_without_a_usable_time() {
        let now = chrono::Utc::now();
        let at = |hours: i64| LatestRelease {
            checked_at: (now - chrono::Duration::hours(hours)).to_rfc3339(),
            ..Default::default()
        };
        assert!(!is_stale(&at(1), now));
        assert!(is_stale(&at(25), now));
        assert!(is_stale(&LatestRelease { checked_at: "never".into(), ..Default::default() }, now));
    }

    #[test]
    fn cache_round_trips_and_feeds_the_notice() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_cache(dir.path()).is_none());
        assert!(notice(dir.path()).is_none());

        let newer = LatestRelease {
            version: "999.0.0".into(),
            url: "https://example.test/release".into(),
            checked_at: chrono::Utc::now().to_rfc3339(),
        };
        write_cache(dir.path(), &newer).unwrap();
        assert_eq!(read_cache(dir.path()).as_ref(), Some(&newer));
        assert_eq!(notice(dir.path()).as_deref(), Some("Anima v999.0.0 is available: https://example.test/release"));

        let same = LatestRelease { version: CURRENT_VERSION.into(), ..newer };
        write_cache(dir.path(), &same).unwrap();
        assert!(notice(dir.path()).is_none());
    }

    #[test]
    fn status_reports_an_update_only_when_newer() {
        let latest = LatestRelease { version: "999.0.0".into(), url: "u".into(), checked_at: "t".into() };
        let status = UpdateStatus::new(true, Some(&latest));
        assert!(status.update_available);
        assert_eq!(status.latest.as_deref(), Some("999.0.0"));
        assert!(!UpdateStatus::new(true, None).update_available);
    }
}
