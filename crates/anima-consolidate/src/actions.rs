use serde::{Deserialize, Serialize};

/// The LLM's decision about how to handle a new memory relative to existing ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationDecision {
    pub action: ConsolidationActionType,
    /// ID of the existing memory to update/supersede (if applicable).
    pub target_id: Option<String>,
    /// Merged content (if action is "update").
    pub merged_content: Option<String>,
    /// Only the genuinely novel claims extracted from the new memory (predict-calibrate).
    /// When set on a "create" action, this is stored instead of the raw input content.
    pub novel_content: Option<String>,
    /// Brief explanation of the decision.
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationActionType {
    Create,
    Update,
    Supersede,
    NoChange,
}

impl ConsolidationDecision {
    /// Default: create a new memory (used when LLM is unavailable or fails).
    pub fn default_create() -> Self {
        Self {
            action: ConsolidationActionType::Create,
            target_id: None,
            merged_content: None,
            novel_content: None,
            reasoning: Some("No LLM available, defaulting to create".into()),
        }
    }
}
