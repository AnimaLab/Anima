use anima_core::memory::Memory;

/// Build the consolidation prompt for the LLM.
pub fn build_consolidation_prompt(new_content: &str, existing: &[(Memory, f64)]) -> String {
    let mut existing_section = String::new();
    for (memory, similarity) in existing {
        existing_section.push_str(&format!(
            "[ID: {}] (created: {}, similarity: {:.2})\n{}\n\n",
            memory.id,
            memory.created_at.format("%Y-%m-%d %H:%M"),
            similarity,
            memory.content,
        ));
    }

    format!(
        r#"You are a memory consolidation engine using predict-calibrate learning.

EXISTING MEMORIES (what is already known):
{existing_section}
NEW MEMORY TO EVALUATE:
{new_content}

Step 1 - PREDICT: Based on the existing memories above, what facts about this topic are already known?
Step 2 - CALIBRATE: Which specific claims in the new memory are NOT already covered by the existing memories? These are the "prediction gaps" — the genuinely new information.

Rules:
- "no_change": Use ONLY when the new memory adds ZERO new specific information — every named entity, exact quantity, explicit label, and concrete detail in it already appears verbatim in existing memories. When in doubt, prefer "create".
  BAD use of no_change: existing memory says "Alice exercises" and new memory says "Alice runs marathons" — "runs marathons" is a new specific detail.
  BAD use of no_change: existing memory says "Bob has cats" and new memory says "Bob's cats are named Luna and Mochi" — the names are novel.
  BAD use of no_change: existing memory says "Carol underwent a gender transition" and new memory says "Carol is a transgender woman" — the explicit identity label is novel.
- "create": Some claims are genuinely new → set novel_content to ONLY those new claims as a concise, self-contained statement. CRITICAL: preserve ALL specific named entities, exact quantities, explicit labels, and concrete activity details from the new memory; only omit facts already explicitly stated word-for-word in existing memories.
- "update": The new memory adds detail or context to an existing one → set target_id and merged_content combining both; novel_content is null.
- "supersede": The new memory contradicts an existing one (newer information wins) → set target_id; novel_content is null.

Respond with ONLY valid JSON (no markdown, no explanation outside the JSON):
{{"action": "create|update|supersede|no_change", "target_id": "<id or null>", "merged_content": "<text or null>", "novel_content": "<only novel claims or null>", "reasoning": "<brief explanation>"}}"#
    )
}
