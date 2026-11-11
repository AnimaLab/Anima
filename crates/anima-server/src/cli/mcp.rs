//! MCP server over stdio: `anima-server mcp`.
//!
//! Speaks newline-delimited JSON-RPC 2.0 on stdin/stdout and forwards each
//! tool call to the Anima REST API. Nothing but protocol messages may be
//! written to stdout.

use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::client::Api;
use super::Options;

const PROTOCOL_VERSION: &str = "2025-06-18";

const INSTRUCTIONS: &str = "Anima is the user's long-term memory, shared by every assistant and \
machine they use. At the start of a session call memory_briefing unless a briefing from Anima is \
already in context. Search memory before starting a task, and save durable facts, preferences, \
decisions and corrections as they come up. Skills listed in the briefing are loaded with skill_get. \
For the full guide, call skill_get with the name anima. Do not store secrets or credentials.";

pub async fn serve(opts: &Options) -> anyhow::Result<()> {
    let api = Api::from_options(opts, Duration::from_secs(120))?;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut out = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(line) {
            Ok(msg) => handle(&api, &msg).await,
            Err(e) => Some(error(Value::Null, -32700, &format!("parse error: {e}"))),
        };
        if let Some(reply) = reply {
            out.write_all(serde_json::to_string(&reply)?.as_bytes()).await?;
            out.write_all(b"\n").await?;
            out.flush().await?;
        }
    }
    Ok(())
}

fn ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Answer one message. Notifications (no `id`) get no reply.
async fn handle(api: &Api, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned()?;
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);

    Some(match method {
        "initialize" => {
            // Echo the client's protocol version when it sends one: the tool
            // subset used here is the same across revisions.
            let version = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(PROTOCOL_VERSION);
            ok(id, json!({
                "protocolVersion": version,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "anima", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => ok(id, json!({})),
        "tools/list" => ok(id, json!({ "tools": tools() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let (text, is_error) = match call(api, name, &args).await {
                // A plain string result is prose for the model: send it unquoted.
                Ok(Value::String(s)) => (s, false),
                Ok(v) => (serde_json::to_string_pretty(&v).unwrap_or_default(), false),
                Err(e) => (e.to_string(), true),
            };
            ok(id, json!({ "content": [{ "type": "text", "text": text }], "isError": is_error }))
        }
        other => error(id, -32601, &format!("method not found: {other}")),
    })
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": { "type": "object", "properties": properties, "required": required },
    })
}

/// Tool definitions. Names and parameters match the TypeScript bridge in `mcp/`.
pub fn tools() -> Vec<Value> {
    vec![
        tool(
            "memory_search",
            "Search the user's memories by semantic similarity, keyword match, or both. Returns \
             scored results ranked by relevance. Use it to find relevant context, past decisions, \
             preferences, or anything previously stored about the user.",
            json!({
                "query": { "type": "string", "description": "Natural language search query" },
                "limit": { "type": "number", "description": "Max results to return (default 10)" },
                "search_mode": { "type": "string", "enum": ["hybrid", "vector", "keyword"], "description": "Search mode (default: hybrid)" },
                "temporal_weight": { "type": "number", "minimum": 0, "maximum": 1, "description": "Recency weight 0-1 (default 0.2, higher prefers recent)" },
                "date_start": { "type": "string", "description": "ISO 8601 date filter, inclusive start" },
                "date_end": { "type": "string", "description": "ISO 8601 date filter, inclusive end" },
                "category": { "type": "string", "description": "Filter by category (identity, preference, environment, routine, task, ...)" },
                "max_tier": { "type": "number", "minimum": 1, "maximum": 4, "description": "Max memory tier: 1=raw, 2=reflected, 3=deduced, 4=induced" },
                "group_by": { "type": "string", "enum": ["episode_id", "category", "source"], "description": "Return the top result per group" },
                "query_rewrite": { "type": "boolean", "description": "Expand the query with extracted keywords (default: false)" },
            }),
            &["query"],
        ),
        tool(
            "memory_add",
            "Store a new memory about the user. The server deduplicates and consolidates with \
             similar memories. Use it for facts, preferences, decisions, goals, relationships, \
             habits, skills, or any personal information worth remembering. One fact per memory, \
             written so it stands on its own.",
            json!({
                "content": { "type": "string", "description": "The memory content to store" },
                "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags for categorization" },
                "consolidate": { "type": "boolean", "description": "Consolidate with similar memories (default: true)" },
                "category": { "type": "string", "description": "Category, which controls decay and ranking: identity, preference, environment, routine, task, inferred, general, or a custom one" },
            }),
            &["content"],
        ),
        tool(
            "memory_update",
            "Update an existing memory by ID: content, tags, or importance. Use it to correct or \
             enrich a memory instead of adding a contradicting one.",
            json!({
                "id": { "type": "string", "description": "Memory ID to update" },
                "content": { "type": "string", "description": "New content (re-embedded)" },
                "tags": { "type": "array", "items": { "type": "string" }, "description": "New tags (replace existing)" },
                "importance": { "type": "number", "minimum": 1, "maximum": 10, "description": "Importance score 1-10" },
            }),
            &["id"],
        ),
        tool(
            "memory_delete",
            "Soft-delete a memory by ID. It is marked deleted and can be purged later.",
            json!({ "id": { "type": "string", "description": "Memory ID to delete" } }),
            &["id"],
        ),
        tool(
            "memory_list",
            "List memories, optionally filtered by status, type or category.",
            json!({
                "limit": { "type": "number", "description": "Max results (default 50)" },
                "offset": { "type": "number", "description": "Pagination offset" },
                "status": { "type": "string", "enum": ["active", "superseded", "deleted"], "description": "Filter by status" },
                "memory_type": { "type": "string", "description": "Filter by memory type" },
                "category": { "type": "string", "description": "Filter by category" },
            }),
            &[],
        ),
        tool(
            "memory_stats",
            "Statistics for the memory namespace: counts by status and access patterns.",
            json!({}),
            &[],
        ),
        tool(
            "memory_ask",
            "Ask a natural language question about the user's memories. Searches relevant \
             memories and uses the server's configured LLM to answer. Best for questions that \
             span several memories.",
            json!({
                "question": { "type": "string", "description": "Question about the user's memories" },
                "search_limit": { "type": "number", "description": "Max search results (default 20)" },
                "max_results": { "type": "number", "description": "Max memories sent to the LLM (default 20)" },
                "skip_llm": { "type": "boolean", "description": "Return the top memories only, without LLM synthesis" },
                "memory_types": { "type": "array", "items": { "type": "string" }, "description": "Filter by memory type" },
                "max_tier": { "type": "number", "minimum": 1, "maximum": 4, "description": "Max memory tier" },
                "date_start": { "type": "string", "description": "ISO 8601 date filter, inclusive start" },
                "date_end": { "type": "string", "description": "ISO 8601 date filter, inclusive end" },
            }),
            &["question"],
        ),
        tool(
            "memory_reflect",
            "Run reflection on raw memories to extract structured facts with confidence and \
             provenance. Without memory_ids, unreflected raw memories are picked automatically.",
            json!({
                "memory_ids": { "type": "array", "items": { "type": "string" }, "description": "Memory IDs to reflect on (empty picks automatically)" },
                "limit": { "type": "number", "description": "Max raw memories to process (default 50)" },
                "async": { "type": "boolean", "description": "Queue as a background job and return immediately" },
            }),
            &[],
        ),
        tool(
            "memory_briefing",
            "A short briefing to read at the start of a session: who the user is, what is \
             relevant to the current project, and which skills are stored. Small by design; \
             use memory_search for anything more specific.",
            json!({
                "project": { "type": "string", "description": "Name of the project or working folder, to include memories relevant to it" },
            }),
            &[],
        ),
        tool(
            "skill_list",
            "List the skills stored in Anima: name, description and size. Skills are reusable \
             instructions (procedures, checklists, conventions) shared by all the user's assistants.",
            json!({}),
            &[],
        ),
        tool(
            "skill_get",
            "Load a skill's full instructions by name. Do this before a task the skill's \
             description matches, then follow it.",
            json!({ "name": { "type": "string", "description": "Skill name, as shown by skill_list or the briefing" } }),
            &["name"],
        ),
        tool(
            "skill_save",
            "Create or update a skill. Use it when the user teaches a repeatable procedure worth \
             keeping. Saving under an existing name replaces that skill and raises its version.",
            json!({
                "name": { "type": "string", "description": "Lowercase letters, digits and hyphens, e.g. 'release-checklist'" },
                "description": { "type": "string", "description": "One sentence saying what the skill does and when to use it" },
                "content": { "type": "string", "description": "The instructions, in Markdown" },
            }),
            &["name", "description", "content"],
        ),
        tool(
            "skill_delete",
            "Delete a stored skill by name.",
            json!({ "name": { "type": "string", "description": "Skill name" } }),
            &["name"],
        ),
        tool(
            "memory_discover",
            "Find memories similar to one concept while steering away from another, for \
             exploratory queries such as 'travel but not work trips'.",
            json!({
                "positive": { "type": "string", "description": "What to find" },
                "negative": { "type": "string", "description": "What to avoid" },
                "limit": { "type": "number", "description": "Max results (default 10)" },
            }),
            &["positive"],
        ),
    ]
}

fn required_str<'a>(args: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing required argument: {key}"))
}

/// Copy the listed keys that are present and non-null.
fn pick(args: &Value, keys: &[&str]) -> serde_json::Map<String, Value> {
    keys.iter()
        .filter_map(|k| args.get(*k).filter(|v| !v.is_null()).map(|v| (k.to_string(), v.clone())))
        .collect()
}

fn encode(segment: &str) -> String {
    // Form encoding writes a space as "+", which is wrong in a path segment.
    url::form_urlencoded::byte_serialize(segment.as_bytes()).collect::<String>().replace('+', "%20")
}

async fn search_ids(api: &Api, query: &str) -> anyhow::Result<Vec<Value>> {
    let found = api
        .post("/api/v1/memories/search", &json!({ "query": query, "limit": 3, "search_mode": "hybrid" }))
        .await?;
    Ok(found
        .get("results")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(|m| m.get("id").cloned()).collect())
        .unwrap_or_default())
}

async fn call(api: &Api, name: &str, args: &Value) -> anyhow::Result<Value> {
    use reqwest::Method;
    match name {
        "memory_search" => {
            required_str(args, "query")?;
            let mut body = pick(args, &[
                "query", "limit", "search_mode", "temporal_weight", "date_start", "date_end",
                "category", "max_tier", "group_by", "query_rewrite",
            ]);
            body.entry("limit").or_insert(json!(10));
            body.entry("search_mode").or_insert(json!("hybrid"));
            api.post("/api/v1/memories/search", &Value::Object(body)).await
        }
        "memory_add" => {
            required_str(args, "content")?;
            let mut body = pick(args, &["content", "tags", "consolidate", "category"]);
            body.entry("consolidate").or_insert(json!(true));
            api.post("/api/v1/memories", &Value::Object(body)).await
        }
        "memory_update" => {
            let id = required_str(args, "id")?;
            let body = pick(args, &["content", "tags", "importance"]);
            let path = format!("/api/v1/memories/{}", encode(id));
            api.json(api.request(Method::PUT, &path).json(&Value::Object(body))).await
        }
        "memory_delete" => {
            let id = required_str(args, "id")?;
            api.json(api.request(Method::DELETE, &format!("/api/v1/memories/{}", encode(id)))).await
        }
        "memory_list" => {
            let query: Vec<(String, String)> = pick(args, &["limit", "offset", "status", "memory_type", "category"])
                .into_iter()
                .map(|(k, v)| (k, v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())))
                .collect();
            api.json(api.request(Method::GET, "/api/v1/memories").query(&query)).await
        }
        "memory_stats" => api.get("/api/v1/stats").await,
        "memory_ask" => {
            required_str(args, "question")?;
            let body = pick(args, &[
                "question", "search_limit", "max_results", "skip_llm", "memory_types", "max_tier",
                "date_start", "date_end",
            ]);
            api.post("/api/v1/ask", &Value::Object(body)).await
        }
        "memory_reflect" => {
            let body = json!({
                "memory_ids": args.get("memory_ids").cloned().unwrap_or_else(|| json!([])),
                "limit": args.get("limit").cloned().unwrap_or(json!(50)),
                "async": args.get("async").cloned().unwrap_or(json!(false)),
            });
            api.post("/api/v1/reflect", &body).await
        }
        "memory_discover" => {
            let positive = required_str(args, "positive")?;
            let positive_ids = search_ids(api, positive).await?;
            if positive_ids.is_empty() {
                return Ok(json!({ "results": [], "message": "No memories matched the positive query" }));
            }
            let negative_ids = match args.get("negative").and_then(Value::as_str) {
                Some(n) if !n.trim().is_empty() => search_ids(api, n).await?,
                _ => vec![],
            };
            let body = json!({
                "positive_ids": positive_ids,
                "negative_ids": negative_ids,
                "query": positive,
                "limit": args.get("limit").cloned().unwrap_or(json!(10)),
            });
            api.post("/api/v1/memories/discover", &body).await
        }
        "memory_briefing" => {
            let mut body = pick(args, &["project"]);
            body.insert("mode".into(), json!("session"));
            let briefing = api.post("/api/v1/briefing", &Value::Object(body)).await?;
            // The text is the briefing; the structured fields would only repeat it.
            let text = briefing.get("text").and_then(Value::as_str).unwrap_or("");
            Ok(json!(if text.is_empty() { "Anima holds nothing to brief on yet." } else { text }))
        }
        "skill_list" => api.get("/api/v1/skills").await,
        "skill_get" => {
            let name = required_str(args, "name")?;
            api.get(&format!("/api/v1/skills/{}", encode(name))).await
        }
        "skill_save" => {
            let name = required_str(args, "name")?;
            required_str(args, "description")?;
            required_str(args, "content")?;
            let body = pick(args, &["description", "content"]);
            let path = format!("/api/v1/skills/{}", encode(name));
            api.json(api.request(Method::PUT, &path).json(&Value::Object(body))).await
        }
        "skill_delete" => {
            let name = required_str(args, "name")?;
            api.json(api.request(Method::DELETE, &format!("/api/v1/skills/{}", encode(name)))).await
        }
        other => anyhow::bail!("unknown tool: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::ClientSettings;

    fn api() -> Api {
        // Port 9 (discard) is never an Anima server: calls that reach the
        // network fail fast, which is all these tests need.
        let settings = ClientSettings { url: Some("http://127.0.0.1:9".into()), ..Default::default() };
        Api::new(&settings, Duration::from_millis(300)).unwrap()
    }

    #[tokio::test]
    async fn initialize_echoes_protocol_version() {
        let msg = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": { "protocolVersion": "2024-11-05" } });
        let reply = handle(&api(), &msg).await.unwrap();
        assert_eq!(reply["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(reply["result"]["serverInfo"]["name"], "anima");
        assert!(reply["result"]["capabilities"]["tools"].is_object());
    }

    #[tokio::test]
    async fn notifications_get_no_reply() {
        let msg = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(handle(&api(), &msg).await.is_none());
    }

    #[tokio::test]
    async fn lists_every_tool_with_a_schema() {
        let reply = handle(&api(), &json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" })).await.unwrap();
        let tools = reply["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec![
            "memory_search", "memory_add", "memory_update", "memory_delete", "memory_list",
            "memory_stats", "memory_ask", "memory_reflect", "memory_briefing", "skill_list",
            "skill_get", "skill_save", "skill_delete", "memory_discover",
        ]);
        for t in tools {
            assert_eq!(t["inputSchema"]["type"], "object");
            assert!(t["inputSchema"]["required"].is_array());
        }
    }

    #[tokio::test]
    async fn unknown_method_is_a_json_rpc_error() {
        let reply = handle(&api(), &json!({ "jsonrpc": "2.0", "id": 3, "method": "nope" })).await.unwrap();
        assert_eq!(reply["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn tool_failures_are_reported_in_band() {
        let missing = json!({ "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": { "name": "memory_add", "arguments": {} } });
        let reply = handle(&api(), &missing).await.unwrap();
        assert_eq!(reply["result"]["isError"], true);
        assert!(reply["result"]["content"][0]["text"].as_str().unwrap().contains("content"));

        let unreachable = json!({ "jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": { "name": "memory_stats" } });
        let reply = handle(&api(), &unreachable).await.unwrap();
        assert_eq!(reply["result"]["isError"], true);
    }

    #[test]
    fn pick_drops_nulls_and_unknown_keys() {
        let picked = pick(&json!({ "a": 1, "b": null, "c": 3 }), &["a", "b"]);
        assert_eq!(Value::Object(picked), json!({ "a": 1 }));
    }
}
