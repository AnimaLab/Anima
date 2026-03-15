use std::sync::Arc;

use anima_core::memory::Memory;

use crate::actions::ConsolidationDecision;
use crate::llm_client::{LlmClient, LlmError};
use crate::prompts::build_consolidation_prompt;

#[derive(Debug, thiserror::Error)]
pub enum ConsolidationError {
    #[error("llm error: {0}")]
    Llm(#[from] LlmError),

    #[error("parse error: {0}")]
    Parse(String),
}

/// Memory consolidation pipeline.
/// Checks for duplicates and uses LLM to decide how to handle near-duplicates.
pub struct Consolidator {
    llm: Arc<dyn LlmClient>,
    similarity_threshold: f64,
}

impl Consolidator {
    pub fn new(llm: Arc<dyn LlmClient>, similarity_threshold: f64) -> Self {
        Self {
            llm,
            similarity_threshold,
        }
    }

    pub fn similarity_threshold(&self) -> f64 {
        self.similarity_threshold
    }

    /// Get a reference to the underlying LLM client (for sharing with processor).
    pub fn llm_client(&self) -> Arc<dyn LlmClient> {
        self.llm.clone()
    }

    /// Ask the LLM to decide what to do with a new memory given similar existing memories.
    pub async fn decide(
        &self,
        new_content: &str,
        existing: &[(Memory, f64)],
    ) -> Result<ConsolidationDecision, ConsolidationError> {
        if existing.is_empty() {
            return Ok(ConsolidationDecision::default_create());
        }

        let prompt = build_consolidation_prompt(new_content, existing);

        let response = match self.llm.complete(&prompt).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM consolidation failed, defaulting to create: {e}");
                return Ok(ConsolidationDecision::default_create());
            }
        };

        // Parse JSON response
        let decision: ConsolidationDecision = parse_llm_response(&response)?;
        Ok(decision)
    }
}

/// Parse the LLM's JSON response, handling common formatting issues.
fn parse_llm_response(response: &str) -> Result<ConsolidationDecision, ConsolidationError> {
    // Try to extract JSON from the response (LLMs sometimes wrap in markdown)
    let json_str = extract_json(response);

    serde_json::from_str::<ConsolidationDecision>(json_str).map_err(|e| {
        ConsolidationError::Parse(format!(
            "failed to parse LLM response as JSON: {e}\nResponse was: {response}"
        ))
    })
}

/// Extract JSON object from a string that may contain markdown or extra text.
fn extract_json(s: &str) -> &str {
    let s = s.trim();

    // Try to find JSON object boundaries
    if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            return &s[start..=end];
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ConsolidationActionType;

    #[test]
    fn test_extract_json_clean() {
        let input = r#"{"action": "create", "target_id": null, "merged_content": null, "reasoning": "new"}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_with_markdown() {
        let input = "```json\n{\"action\": \"create\"}\n```";
        assert_eq!(extract_json(input), "{\"action\": \"create\"}");
    }

    #[test]
    fn test_parse_create_decision() {
        let json = r#"{"action": "create", "target_id": null, "merged_content": null, "reasoning": "genuinely new"}"#;
        let decision = parse_llm_response(json).unwrap();
        assert_eq!(decision.action, ConsolidationActionType::Create);
    }

    #[test]
    fn test_parse_update_decision() {
        let json = r#"{"action": "update", "target_id": "abc123", "merged_content": "combined text", "reasoning": "adds detail"}"#;
        let decision = parse_llm_response(json).unwrap();
        assert_eq!(decision.action, ConsolidationActionType::Update);
        assert_eq!(decision.target_id, Some("abc123".into()));
        assert_eq!(decision.merged_content, Some("combined text".into()));
    }
}
