/*
 * dashboard/src/api.js
 *
 * Thin wrapper over the FastAPI backend. Every call goes through get()
 * so there is one place that handles the base URL and failed responses.
 *
 * Set VITE_API_BASE in a .env file beside package.json to point the
 * dashboard somewhere other than localhost.
 */

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  // --- meta ---------------------------------------------------------
  health() {
    return get("/api/health");
  },

  // --- metrics ------------------------------------------------------
  latest() {
    return get("/api/metrics/latest");
  },

  metrics(window = "1h") {
    return get(`/api/metrics?window=${window}`);
  },

  // --- forecasting --------------------------------------------------
  // Returns { now, metric, horizon_s, points: [{ ts, actual, predicted }] }.
  // Points past the present have a predicted value and a null actual,
  // which is what draws the line beyond the "now" marker.
  forecast(window = "1h", metric = "cpu_percent") {
    const q = new URLSearchParams({ window, metric });
    return get(`/api/forecast?${q}`);
  },

  // Rolling error on forecasts whose moment has already arrived, so
  // every row in it was predicted before the value existed.
  forecastAccuracy(hours = 24) {
    const q = new URLSearchParams({ hours: String(hours) });
    return get(`/api/forecast/accuracy?${q}`);
  },

  // --- advisories ---------------------------------------------------
  advisories({ limit = 30, category, minConfidence = 0 } = {}) {
    const q = new URLSearchParams({
      limit: String(limit),
      min_confidence: String(minConfidence),
    });
    if (category) q.set("category", category);
    return get(`/api/advisories?${q}`);
  },

  summary() {
    return get("/api/summary");
  },

  // POST rather than GET, so it does not use the get() helper.
  async sendFeedback(id, feedback) {
    const res = await fetch(`${BASE}/api/advisories/${id}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feedback }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
};