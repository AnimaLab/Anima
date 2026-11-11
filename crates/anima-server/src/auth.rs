//! API authentication, Host validation, and CORS policy.
//!
//! Threat model: Anima usually runs on a developer machine, where the main
//! risk is a web page in the user's browser talking to `localhost`. Three
//! layers address that:
//!
//! 1. **API key** — when `server.api_key` / `ANIMA_API_KEY` is set, every
//!    `/api/*` request must carry it. Mandatory for non-loopback binds.
//! 2. **Host check** — when no key is set (loopback only), requests whose
//!    `Host` header is not a loopback name are rejected. This blocks DNS
//!    rebinding, where an attacker's domain resolves to 127.0.0.1.
//! 3. **CORS allowlist** — cross-origin browser access is denied unless the
//!    origin is listed in `server.cors_allowed_origins`.

use std::net::IpAddr;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, HOST};
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use sha2::{Digest, Sha256};
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::config::ServerConfig;
use crate::dto::ErrorResponse;

/// Alternative header for clients that cannot set `Authorization`.
const API_KEY_HEADER: &str = "x-anima-api-key";

/// Resolved authentication settings, shared with the middleware.
#[derive(Debug)]
pub struct AuthSettings {
    /// SHA-256 of the expected API key. `None` = no key configured.
    key_digest: Option<[u8; 32]>,
    /// Reject requests whose Host header is not a loopback name.
    enforce_loopback_host: bool,
}

impl AuthSettings {
    /// Build settings from server config, or fail if the combination is unsafe.
    pub fn from_config(server: &ServerConfig) -> anyhow::Result<Self> {
        let key = server.resolve_api_key();
        let loopback = is_loopback_bind(&server.host);

        if key.is_none() && !loopback && !server.allow_unauthenticated {
            anyhow::bail!(
                "refusing to listen on non-loopback address '{}' without an API key. \
                 Set ANIMA_API_KEY (or server.api_key in config.toml), or set \
                 server.allow_unauthenticated = true if another layer handles auth.",
                server.host
            );
        }
        if key.is_none() && !loopback {
            tracing::warn!(
                "API is exposed on '{}' WITHOUT authentication (allow_unauthenticated = true)",
                server.host
            );
        }

        Ok(Self {
            key_digest: key.as_deref().map(digest),
            enforce_loopback_host: key.is_none() && loopback,
        })
    }

    pub fn key_required(&self) -> bool {
        self.key_digest.is_some()
    }

    fn key_matches(&self, presented: &str) -> bool {
        match &self.key_digest {
            // Comparing fixed-length digests avoids leaking key length/prefix
            // through comparison timing.
            Some(expected) => constant_time_eq(expected, &digest(presented)),
            None => true,
        }
    }
}

fn digest(value: &str) -> [u8; 32] {
    Sha256::digest(value.as_bytes()).into()
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// True when `host` (a bind address from config) is a loopback address.
pub fn is_loopback_bind(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.trim_matches(|c| c == '[' || c == ']')
        .parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// True when a `Host` header value (`name[:port]`) names the loopback interface.
fn is_loopback_host_header(value: &str) -> bool {
    let value = value.trim();
    let name = if let Some(rest) = value.strip_prefix('[') {
        // IPv6 literal: [::1]:3000
        rest.split(']').next().unwrap_or("")
    } else {
        value.rsplit_once(':').map(|(h, _)| h).unwrap_or(value)
    };
    let name = name.trim_end_matches('.');
    name.eq_ignore_ascii_case("localhost")
        || name.to_ascii_lowercase().ends_with(".localhost")
        || name.parse::<IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false)
}

fn presented_key(req: &Request) -> Option<&str> {
    if let Some(value) = req.headers().get(AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if let Some((scheme, token)) = value.split_once(' ') {
            if scheme.eq_ignore_ascii_case("bearer") {
                return Some(token.trim());
            }
        }
    }
    req.headers().get(API_KEY_HEADER).and_then(|v| v.to_str().ok())
}

fn reject(status: StatusCode, message: &str) -> Response {
    tracing::warn!(domain = "auth", status = status.as_u16(), "{message}");
    let mut res = (status, Json(ErrorResponse { error: message.to_string() })).into_response();
    if status == StatusCode::UNAUTHORIZED {
        res.headers_mut().insert(
            axum::http::header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer"),
        );
    }
    res
}

/// Middleware guarding the whole router. Static UI assets and `/health*`
/// stay public so the UI can load and probes keep working; everything under
/// `/api/` requires the API key when one is configured.
pub async fn guard(
    State(auth): State<Arc<AuthSettings>>,
    req: Request,
    next: Next,
) -> Response {
    if auth.enforce_loopback_host {
        let host = req
            .headers()
            .get(HOST)
            .and_then(|v| v.to_str().ok())
            .or_else(|| req.uri().authority().map(|a| a.as_str()));
        if let Some(host) = host {
            if !is_loopback_host_header(host) {
                return reject(
                    StatusCode::FORBIDDEN,
                    "request Host is not a loopback address; set an API key to allow remote access",
                );
            }
        }
    }

    if auth.key_required() && req.uri().path().starts_with("/api/") {
        match presented_key(&req) {
            Some(key) if auth.key_matches(key) => {}
            Some(_) => return reject(StatusCode::UNAUTHORIZED, "invalid API key"),
            None => {
                return reject(
                    StatusCode::UNAUTHORIZED,
                    "missing API key (expected 'Authorization: Bearer <key>')",
                )
            }
        }
    }

    next.run(req).await
}

/// Build the CORS layer from the configured origin allowlist.
///
/// With an empty list no `Access-Control-Allow-Origin` header is ever sent, so
/// browsers block cross-origin pages from reading or (because every API call
/// needs a preflighted header) writing. The bundled web UI is same-origin and
/// is unaffected.
pub fn cors_layer(server: &ServerConfig) -> CorsLayer {
    let origins: Vec<HeaderValue> = server
        .cors_allowed_origins
        .iter()
        .filter_map(|origin| {
            let origin = origin.trim().trim_end_matches('/');
            if origin == "*" {
                tracing::warn!("ignoring '*' in server.cors_allowed_origins; list explicit origins");
                return None;
            }
            match HeaderValue::from_str(origin) {
                Ok(v) => Some(v),
                Err(_) => {
                    tracing::warn!("ignoring invalid CORS origin '{origin}'");
                    None
                }
            }
        })
        .collect();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS])
        .allow_headers([
            AUTHORIZATION,
            CONTENT_TYPE,
            HeaderName::from_static(API_KEY_HEADER),
            HeaderName::from_static("x-anima-namespace"),
            HeaderName::from_static("x-anima-principal"),
        ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(host: &str, api_key: &str, allow_unauthenticated: bool) -> ServerConfig {
        ServerConfig {
            host: host.into(),
            port: 3000,
            api_key: api_key.into(),
            cors_allowed_origins: Vec::new(),
            allow_unauthenticated,
        }
    }

    #[test]
    fn loopback_bind_detection() {
        assert!(is_loopback_bind("127.0.0.1"));
        assert!(is_loopback_bind("::1"));
        assert!(is_loopback_bind("localhost"));
        assert!(!is_loopback_bind("0.0.0.0"));
        assert!(!is_loopback_bind("192.168.1.10"));
    }

    #[test]
    fn loopback_host_header_detection() {
        assert!(is_loopback_host_header("localhost:3000"));
        assert!(is_loopback_host_header("127.0.0.1:3000"));
        assert!(is_loopback_host_header("[::1]:3000"));
        assert!(is_loopback_host_header("anima.localhost"));
        assert!(!is_loopback_host_header("evil.example.com:3000"));
        assert!(!is_loopback_host_header("localhost.evil.example.com"));
        assert!(!is_loopback_host_header("192.168.1.10:3000"));
    }

    #[test]
    fn key_comparison() {
        // A non-empty explicit key takes precedence over ANIMA_API_KEY.
        let auth = AuthSettings::from_config(&server("0.0.0.0", "s3cret", false)).unwrap();
        assert!(auth.key_required());
        assert!(!auth.enforce_loopback_host);
        assert!(auth.key_matches("s3cret"));
        assert!(!auth.key_matches("s3cre"));
        assert!(!auth.key_matches(""));
    }

    #[test]
    fn digest_equality_is_exact() {
        assert!(constant_time_eq(&digest("a"), &digest("a")));
        assert!(!constant_time_eq(&digest("a"), &digest("b")));
    }
}
