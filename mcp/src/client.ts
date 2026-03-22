const BASE_URL = process.env.ANIMA_URL || "http://127.0.0.1:3000";
const NAMESPACE = process.env.ANIMA_NAMESPACE || "default";

function headers(): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "X-Anima-Namespace": NAMESPACE,
  };
}

async function request(url: string, init?: RequestInit) {
  const res = await fetch(url, { ...init, headers: { ...headers(), ...init?.headers } });
  if (!res.ok) {
    const body = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
  return res.json();
}

export async function searchMemories(
  query: string,
  limit?: number,
  searchMode?: string,
  temporalWeight?: number,
) {
  return request(`${BASE_URL}/api/v1/memories/search`, {
    method: "POST",
    body: JSON.stringify({
      query,
      limit: limit ?? 10,
      search_mode: searchMode ?? "hybrid",
      temporal_weight: temporalWeight,
    }),
  });
}

export async function addMemory(
  content: string,
  tags?: string[],
  consolidate = true,
  category?: string,
) {
  return request(`${BASE_URL}/api/v1/memories`, {
    method: "POST",
    body: JSON.stringify({
      content,
      tags,
      consolidate,
      category,
    }),
  });
}

export async function updateMemory(
  id: string,
  updates: {
    content?: string;
    tags?: string[];
    importance?: number;
  },
) {
  return request(`${BASE_URL}/api/v1/memories/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(updates),
  });
}

export async function deleteMemory(id: string) {
  return request(`${BASE_URL}/api/v1/memories/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export async function listMemories(opts?: {
  limit?: number;
  offset?: number;
  status?: string;
  memory_type?: string;
}) {
  const params = new URLSearchParams();
  if (opts?.limit) params.set("limit", String(opts.limit));
  if (opts?.offset) params.set("offset", String(opts.offset));
  if (opts?.status) params.set("status", opts.status);
  if (opts?.memory_type) params.set("memory_type", opts.memory_type);
  return request(`${BASE_URL}/api/v1/memories?${params}`);
}

export async function getStats() {
  return request(`${BASE_URL}/api/v1/stats`);
}

export async function reflect(
  memoryIds?: string[],
  limit?: number,
  asyncMode?: boolean,
) {
  return request(`${BASE_URL}/api/v1/reflect`, {
    method: "POST",
    body: JSON.stringify({
      memory_ids: memoryIds ?? [],
      limit: limit ?? 50,
      async: asyncMode ?? false,
    }),
  });
}

export async function ask(
  question: string,
  llm?: {
    base_url: string;
    model: string;
    api_key?: string;
  },
  searchLimit?: number,
  maxResults?: number,
) {
  const body: Record<string, unknown> = {
    question,
    search_limit: searchLimit,
    max_results: maxResults,
  };
  if (llm) {
    body.llm = { ...llm, streaming: false, vision: false, tool_use: false };
  }
  return request(`${BASE_URL}/api/v1/ask`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
