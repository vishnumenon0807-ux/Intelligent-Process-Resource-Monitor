const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  health: () => get("/api/health"),

  latest: () => get("/api/metrics/latest"),

  metrics: (window = "1h") => get(`/api/metrics?window=${window}`),

  advisories: ({ limit = 30, category, minConfidence = 0 } = {}) => {
    const q = new URLSearchParams({ limit, min_confidence: minConfidence });
    if (category) q.set("category", category);
    return get(`/api/advisories?${q}`);
  },

  summary: () => get("/api/summary"),

  sendFeedback: async (id, feedback) => {
    const res = await fetch(`${BASE}/api/advisories/${id}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feedback }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
};