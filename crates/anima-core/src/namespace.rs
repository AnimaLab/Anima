use crate::error::CoreError;

/// Hierarchical namespace: "org/project/user/agent"
/// Each segment is validated (non-empty, alphanumeric + hyphens + underscores).
///
/// Queries scope a namespace to its tree: the namespace itself and its
/// descendants, matched in SQL as `(namespace = ?N OR namespace GLOB ?N || '/*')`.
/// GLOB is case-sensitive and none of its special characters (`*`, `?`, `[`)
/// can appear in a namespace, so the namespace needs no escaping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Namespace(String);

impl Namespace {
    pub fn parse(s: &str) -> Result<Self, CoreError> {
        let s = s.trim().trim_matches('/');
        if s.is_empty() {
            return Err(CoreError::InvalidNamespace("namespace cannot be empty".into()));
        }
        for segment in s.split('/') {
            if segment.is_empty() {
                return Err(CoreError::InvalidNamespace(
                    "namespace segments cannot be empty".into(),
                ));
            }
            if !segment
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
            {
                return Err(CoreError::InvalidNamespace(format!(
                    "invalid characters in segment: {segment}"
                )));
            }
        }
        Ok(Self(s.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the depth (number of segments).
    pub fn depth(&self) -> usize {
        self.0.split('/').count()
    }

    /// The namespace one level up, or `None` for a top-level namespace.
    pub fn parent(&self) -> Option<Self> {
        self.0.rsplit_once('/').map(|(p, _)| Self(p.to_string()))
    }

    /// `self/<segment>`, with `segment` cleaned by [`project_segment`].
    pub fn child(&self, segment: &str) -> Option<Self> {
        project_segment(segment).map(|s| Self(format!("{}/{s}", self.0)))
    }

    /// Whether `other` is this namespace or one of its descendants.
    pub fn contains(&self, other: &Namespace) -> bool {
        other.0 == self.0 || other.0.starts_with(&format!("{}/", self.0))
    }

    /// The deepest namespace that contains every one of `namespaces`.
    pub fn common_ancestor<'a>(namespaces: impl IntoIterator<Item = &'a Namespace>) -> Option<Self> {
        let mut iter = namespaces.into_iter();
        let first: Vec<&str> = iter.next()?.0.split('/').collect();
        let mut len = first.len();
        for ns in iter {
            len = first
                .iter()
                .zip(ns.0.split('/'))
                .take(len)
                .take_while(|(a, b)| *a == b)
                .count();
            if len == 0 {
                return None;
            }
        }
        Some(Self(first[..len].join("/")))
    }
}

/// Longest namespace segment made from a project name.
const MAX_PROJECT_SEGMENT: usize = 64;

/// A project name made into one namespace segment: lowercase, characters
/// outside `a-z 0-9 - _ .` turned into `-`, runs of `-` collapsed, at most
/// 64 characters. `None` when nothing usable is left.
pub fn project_segment(name: &str) -> Option<String> {
    let mut out = String::new();
    for c in name.trim().chars().flat_map(char::to_lowercase) {
        let c = if c.is_ascii_alphanumeric() || c == '_' || c == '.' { c } else { '-' };
        if c == '-' && (out.is_empty() || out.ends_with('-')) {
            continue;
        }
        out.push(c);
    }
    let out: String = out.trim_matches(|c| c == '-' || c == '.').chars().take(MAX_PROJECT_SEGMENT).collect();
    let out = out.trim_end_matches(|c| c == '-' || c == '.').to_string();
    (!out.is_empty()).then_some(out)
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_namespaces() {
        assert!(Namespace::parse("org").is_ok());
        assert!(Namespace::parse("org/project").is_ok());
        assert!(Namespace::parse("org/project/user-1/agent_a").is_ok());
        assert!(Namespace::parse("my.org/v2").is_ok());
    }

    #[test]
    fn test_invalid_namespaces() {
        assert!(Namespace::parse("").is_err());
        assert!(Namespace::parse("   ").is_err());
        assert!(Namespace::parse("org//project").is_err());
        assert!(Namespace::parse("org/proj ect").is_err());
    }

    #[test]
    fn parent_child_and_common_ancestor() {
        let ns = |s: &str| Namespace::parse(s).unwrap();
        assert_eq!(ns("default/anima").parent(), Some(ns("default")));
        assert_eq!(ns("default").parent(), None);
        assert_eq!(ns("default").child("My Project!"), Some(ns("default/my-project")));
        assert_eq!(ns("default").child("  "), None);
        assert!(ns("default").contains(&ns("default/anima")));
        assert!(ns("default").contains(&ns("default")));
        assert!(!ns("default").contains(&ns("defaults/anima")), "a prefix is not a parent");
        let all = [ns("default/anima"), ns("default/proxy"), ns("default")];
        assert_eq!(Namespace::common_ancestor(&all), Some(ns("default")));
        assert_eq!(Namespace::common_ancestor(&[ns("w/a/x"), ns("w/a/y")]), Some(ns("w/a")));
        assert_eq!(Namespace::common_ancestor(&[ns("default/anima")]), Some(ns("default/anima")));
        assert_eq!(Namespace::common_ancestor(&[ns("a"), ns("b")]), None);
    }

    #[test]
    fn project_names_become_one_segment() {
        assert_eq!(project_segment("llmaas-proxy").as_deref(), Some("llmaas-proxy"));
        assert_eq!(project_segment("Billing API").as_deref(), Some("billing-api"));
        assert_eq!(project_segment("a/b\\c").as_deref(), Some("a-b-c"));
        assert_eq!(project_segment("--..--"), None);
        assert_eq!(project_segment(&"x".repeat(100)).map(|s| s.len()), Some(64));
    }

    #[test]
    fn test_trim_slashes() {
        let ns = Namespace::parse("/org/project/").unwrap();
        assert_eq!(ns.as_str(), "org/project");
    }
}
