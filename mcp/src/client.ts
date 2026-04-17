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
  dateStart?: string,
  dateEnd?: string,
  category?: string,
  maxTier?: number,
  groupBy?: string,
  queryRewrite?: boolean,
) {
  const body: Record<string, unknown> = {
    query,
    limit: limit ?? 10,
    search_mode: searchMode ?? "hybrid",
  };
  if (temporalWeight !== undefined) body.temporal_weight = temporalWeight;
  if (dateStart !== undefined) body.date_start = dateStart;
  if (dateEnd !== undefined) body.date_end = dateEnd;
  if (category !== undefined) body.category = category;
  if (maxTier !== undefined) body.max_tier = maxTier;
  if (groupBy !== undefined) body.group_by = groupBy;
  if (queryRewrite) body.query_rewrite = true;
  return request(`${BASE_URL}/api/v1/memories/search`, {
    method: "POST",
    body: JSON.stringify(body),
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
  skipLlm?: boolean,
  memoryTypes?: string[],
  maxTier?: number,
  dateStart?: string,
  dateEnd?: string,
) {
  const body: Record<string, unknown> = {
    question,
    search_limit: searchLimit,
    max_results: maxResults,
  };
  if (llm) {
    body.llm = { ...llm, streaming: false, vision: false, tool_use: false };
  }
  if (skipLlm) body.skip_llm = true;
  if (memoryTypes && memoryTypes.length > 0) body.memory_types = memoryTypes;
  if (maxTier !== undefined) body.max_tier = maxTier;
  if (dateStart !== undefined) body.date_start = dateStart;
  if (dateEnd !== undefined) body.date_end = dateEnd;
  return request(`${BASE_URL}/api/v1/ask`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function discover(
  positive: string,
  negative?: string,
  limit?: number,
) {
  // Resolve positive text to IDs
  const posResults = await searchMemories(positive, 3);
  const positiveIds: string[] = (posResults.results ?? []).map((r: { id: string }) => r.id);

  if (positiveIds.length === 0) {
    return { results: [], message: "No memories matched the positive query" };
  }

  // Resolve negative text to IDs if provided
  let negativeIds: string[] = [];
  if (negative) {
    const negResults = await searchMemories(negative, 3);
    negativeIds = (negResults.results ?? []).map((r: { id: string }) => r.id);
  }

  return request(`${BASE_URL}/api/v1/memories/discover`, {
    method: "POST",
    body: JSON.stringify({
      positive_ids: positiveIds,
      negative_ids: negativeIds,
      query: positive,
      limit: limit ?? 10,
    }),
  });
}
