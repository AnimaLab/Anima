use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Semantic category for a memory, orthogonal to the processing tier (raw/reflected/deduced/induced).
/// Controls decay rate, dedup behavior, and retrieval ranking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    /// Who the user is, how to address them, relationship facts. Near-zero decay.
    Identity,
    /// Likes, dislikes, style, communication preferences. Stable but editable.
    Preference,
    /// Ports, paths, services, infra facts, workarounds. Useful but may drift.
    Environment,
    /// Recurring tasks, schedules, habits.
    Routine,
    /// Current work, temporary context. Aggressive decay.
    Task,
    /// Things Anima concluded but user hasn't confirmed. Lower confidence.
    Inferred,
    /// Default: uncategorized or general-purpose memory.
    General,
}

impl MemoryCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Preference => "preference",
            Self::Environment => "environment",
            Self::Routine => "routine",
            Self::Task => "task",
            Self::Inferred => "inferred",
            Self::General => "general",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "identity" => Some(Self::Identity),
            "preference" => Some(Self::Preference),
            "environment" => Some(Self::Environment),
            "routine" => Some(Self::Routine),
            "task" => Some(Self::Task),
            "inferred" => Some(Self::Inferred),
            "general" => Some(Self::General),
            _ => None,
        }
    }

    /// Default temporal decay lambda for this category.
    /// Lower = slower decay. Identity barely decays; tasks decay fast.
    pub fn default_lambda(&self) -> f64 {
        match self {
            Self::Identity => 0.00001,    // ~2,888 day half-life (near-permanent)
            Self::Preference => 0.0001,   // ~289 day half-life
            Self::Environment => 0.0005,  // ~58 day half-life
            Self::Routine => 0.0003,      // ~96 day half-life
            Self::Task => 0.005,          // ~5.8 day half-life
            Self::Inferred => 0.001,      // ~29 day half-life (same as current global default)
            Self::General => 0.001,       // ~29 day half-life (same as current global default)
        }
    }
}

impl Default for MemoryCategory {
    fn default() -> Self {
        Self::General
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Superseded,
    Deleted,
}

impl MemoryStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Deleted => "deleted",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "active" => Some(Self::Active),
            "superseded" => Some(Self::Superseded),
            "deleted" => Some(Self::Deleted),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub namespace: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
    pub tags: Vec<String>,
    pub memory_type: String,
    pub status: MemoryStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub importance: i32,
    pub episode_id: Option<String>,
    pub event_date: Option<String>,
    pub hash: String,
    /// Semantic category (identity, preference, environment, routine, task, inferred, general).
    #[serde(default)]
    pub category: MemoryCategory,
}

impl Memory {
    pub fn new(
        namespace: String,
        content: String,
        metadata: Option<serde_json::Value>,
        tags: Vec<String>,
        memory_type: Option<String>,
    ) -> Self {
        let now = Utc::now();
        let id = ulid::Ulid::new().to_string();
        let hash = content_hash(&content);

        Self {
            id,
            namespace,
            content,
            metadata,
            tags,
            memory_type: memory_type.unwrap_or_else(|| "raw".to_string()),
            status: MemoryStatus::Active,
            created_at: now,
            updated_at: now,
            accessed_at: now,
            access_count: 0,
            importance: 5,
            episode_id: None,
            event_date: None,
            hash,
            category: MemoryCategory::General,
        }
    }
}

/// SHA-256 hash of lowercase, trimmed content for dedup
pub fn content_hash(content: &str) -> String {
    let normalized = content.trim().to_lowercase();
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationAction {
    Created,
    Updated,
    Superseded,
    Deduplicated,
    NoChange,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash_normalization() {
        assert_eq!(
            content_hash("  Hello World  "),
            content_hash("hello world")
        );
    }

    #[test]
    fn test_content_hash_different() {
        assert_ne!(content_hash("hello"), content_hash("world"));
    }

    #[test]
    fn test_memory_new() {
        let m = Memory::new("org/user".into(), "test content".into(), None, vec![], None);
        assert!(!m.id.is_empty());
        assert_eq!(m.status, MemoryStatus::Active);
        assert_eq!(m.access_count, 0);
        assert_eq!(m.memory_type, "raw");
        assert!(m.tags.is_empty());
    }
}
