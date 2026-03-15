use crate::error::CoreError;

/// Hierarchical namespace: "org/project/user/agent"
/// Each segment is validated (non-empty, alphanumeric + hyphens + underscores).
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

    /// Returns a LIKE pattern for prefix matching: "org/project" -> "org/project/%"
    /// For exact match, returns the namespace itself.
    pub fn like_pattern(&self) -> String {
        format!("{}%", self.0)
    }

    /// Returns the depth (number of segments).
    pub fn depth(&self) -> usize {
        self.0.split('/').count()
    }
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
    fn test_like_pattern() {
        let ns = Namespace::parse("org/project").unwrap();
        assert_eq!(ns.like_pattern(), "org/project%");
    }

    #[test]
    fn test_trim_slashes() {
        let ns = Namespace::parse("/org/project/").unwrap();
        assert_eq!(ns.as_str(), "org/project");
    }
}
