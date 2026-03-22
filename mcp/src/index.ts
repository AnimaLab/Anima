import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import * as client from "./client.js";

const server = new McpServer({
  name: "anima-memory",
  version: "0.1.0",
});

// --- Tools ---

server.tool(
  "memory_search",
  "Search the user's memories by semantic similarity, keyword match, or hybrid. Returns scored results ranked by relevance. Use this to find relevant context, past decisions, preferences, or any previously stored information about the user.",
  {
    query: z.string().describe("Natural language search query"),
    limit: z.number().optional().describe("Max results to return (default 10)"),
    search_mode: z.enum(["hybrid", "vector", "keyword"]).optional().describe("Search mode (default: hybrid)"),
    temporal_weight: z.number().min(0).max(1).optional().describe("Recency weight 0-1 (default 0.2, higher = prefer recent)"),
  },
  async ({ query, limit, search_mode, temporal_weight }) => {
    const result = await client.searchMemories(query, limit, search_mode, temporal_weight);
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

server.tool(
  "memory_add",
  "Store a new memory about the user. The server automatically deduplicates and consolidates with existing similar memories. Use for saving facts, preferences, stories, decisions, goals, relationships, emotions, habits, beliefs, skills, locations, or any personal information worth remembering.",
  {
    content: z.string().describe("The memory content to store"),
    tags: z.array(z.string()).optional().describe("Tags for categorization (e.g. ['food', 'health'])"),
    consolidate: z.boolean().optional().describe("Auto-consolidate with similar memories (default: true)"),
    category: z.string().optional().describe("Semantic category — controls decay rate and ranking. Built-in: identity, preference, environment, routine, task, inferred, general. Custom categories can be defined in config.toml"),
  },
  async ({ content, tags, consolidate, category }) => {
    const result = await client.addMemory(content, tags, consolidate ?? true, category);
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

server.tool(
  "memory_update",
  "Update an existing memory by ID. Can modify content, tags, or importance score. Use when correcting information or enriching existing memories.",
  {
    id: z.string().describe("Memory ID to update"),
    content: z.string().optional().describe("New content (triggers re-embedding)"),
    tags: z.array(z.string()).optional().describe("New tags (replaces existing)"),
    importance: z.number().min(1).max(10).optional().describe("Importance score 1-10"),
  },
  async ({ id, ...updates }) => {
    const result = await client.updateMemory(id, updates);
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

server.tool(
  "memory_delete",
  "Soft-delete a memory by ID. The memory is marked as deleted but can be purged later.",
  {
    id: z.string().describe("Memory ID to delete"),
  },
  async ({ id }) => {
    const result = await client.deleteMemory(id);
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

server.tool(
  "memory_list",
  "List memories with optional filtering by status or type. Use for browsing stored memories or checking what's been remembered.",
  {
    limit: z.number().optional().describe("Max results (default 50)"),
    offset: z.number().optional().describe("Pagination offset"),
    status: z.enum(["active", "superseded", "deleted"]).optional().describe("Filter by status"),
    memory_type: z.string().optional().describe("Filter by memory type"),
  },
  async (args) => {
    const result = await client.listMemories(args);
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

server.tool(
  "memory_stats",
  "Get statistics about the memory namespace: total count, active/superseded/deleted counts, access patterns.",
  {},
  async () => {
    const result = await client.getStats();
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

server.tool(
  "memory_ask",
  "Ask a natural language question about the user's memories. Searches relevant memories and uses the server's configured LLM to generate a direct answer. Best for complex questions that span multiple memories (e.g. 'What are the user's dietary restrictions?', 'What happened at their last meeting?').",
  {
    question: z.string().describe("Natural language question about the user's memories"),
    search_limit: z.number().optional().describe("Max search results (default 20)"),
    max_results: z.number().optional().describe("Max memories sent to LLM (default 20)"),
  },
  async ({ question, search_limit, max_results }) => {
    const result = await client.ask(question, undefined, search_limit, max_results);
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

server.tool(
  "memory_reflect",
  "Trigger reflection on raw memories to extract structured facts. Reflection analyzes raw memories and produces higher-quality 'reflected' facts with confidence scores and provenance tracking. If no memory_ids provided, auto-finds unreflected raw memories. After reflection, deduction automatically runs to infer new facts from combinations of reflected facts.",
  {
    memory_ids: z.array(z.string()).optional().describe("Specific memory IDs to reflect on (empty = auto-find unreflected)"),
    limit: z.number().optional().describe("Max raw memories to process when auto-finding (default 50)"),
    async: z.boolean().optional().describe("Queue as background job and return immediately (default: false)"),
  },
  async ({ memory_ids, limit, async: asyncMode }) => {
    const result = await client.reflect(memory_ids, limit, asyncMode);
    return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
  },
);

// --- Start ---

const transport = new StdioServerTransport();
await server.connect(transport);
