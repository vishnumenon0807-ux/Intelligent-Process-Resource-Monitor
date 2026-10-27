import { useCallback, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "./api";
import "./Forecast.css";

/*
 * Forecast panel
 * --------------
 * Plots what the machine actually did against what the LSTM said it
 * would do, on one shared time grid.
 *
 * The chart is deliberately predicted-vs-actual rather than a bare
 * forecast line extending into the future. A lone dashed line looks
 * impressive and proves nothing: a viewer cannot tell an LSTM from
 * last_value * 1.02. Overlaying the two means every point left of the
 * "now" marker is a forecast that has since been checked against
 * reality, which is evidence rather than decoration.
 *
 * Drop into App.jsx beside <History />:
 *     import Forecast from "./Forecast";
 *     ...
 *     <Forecast window={window_} />
 */

// The predictor writes every 30 s, so polling faster than the rest of
// the dashboard would just re-fetch identical rows.
const POLL_MS = 15000;

const METRICS = [
  { key: "cpu_percent", label: "CPU" },
  { key: "ram_percent", label: "Memory" },
];

// Recharts needs real colour strings, not CSS variables, for the
// gradient stops. Change these two to match your palette.
const ACTUAL = "#38bdf8";   // sky — measured
const FORECAST = "#fbbf24"; // amber — projected

const timeLabel = (iso) =>
  new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function ForecastTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="fc-tip">
      <div className="fc-tip-time">{timeLabel(label)}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="fc-tip-row">
          <span className="fc-dot" style={{ background: p.stroke }} />
          <span>{p.dataKey === "actual" ? "Actual" : "Forecast"}</span>
          <strong>{p.value == null ? "—" : `${p.value.toFixed(1)}%`}</strong>
        </div>
      ))}
    </div>
  );
}

export default function Forecast({ window: window_ = "1h" }) {
  const [metric, setMetric] = useState("cpu_percent");
  const [data, setData] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [f, a] = await Promise.all([
        api.forecast(window_, metric),
        api.forecastAccuracy(),
      ]);
      setData(f);
      setAccuracy(a);
      setError(null);
    } catch (e) {
      setError(e.message || "Could not reach the API");
    }
  }, [window_, metric]);

  useEffect(() => {
    load();
    const t = setInterval(load, POLL_MS);
    return () => clearInterval(t);
  }, [load]);

  // The x axis is keyed on the raw timestamp, not on a formatted time
  // string. Buckets can be narrower than a minute, so two neighbouring
  // points would share a label like "09:36 PM"; duplicate categories
  // break the "now" marker, which has to match exactly one of them.
  // Formatting happens in the tick and the tooltip instead.
  const points = data?.points ?? [];

  // The boundary between measured and predicted is the last point that
  // actually has a reading. Taking it from the data rather than from
  // the browser clock keeps the marker on the chart's own grid, so it
  // cannot drift if the browser and the database disagree about the
  // time.
  const lastMeasured = [...points].reverse().find((p) => p.actual != null);
  const boundary = lastMeasured?.ts;

  const forecastCount = points.filter((p) => p.predicted != null).length;
  const row = accuracy?.metrics?.find((m) => m.metric === metric);

  return (
    <section className="fc">
      <header className="fc-head">
        <h2>Forecast</h2>
        <div className="fc-tabs">
          {METRICS.map((m) => (
            <button
              key={m.key}
              type="button"
              className={m.key === metric ? "on" : ""}
              onClick={() => setMetric(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </header>

      {error && <p className="fc-msg fc-err">{error}</p>}

      {!error && forecastCount === 0 && (
        <p className="fc-msg">
          No forecasts stored yet. Start <code>predictor.py</code> and give it
          a minute.
        </p>
      )}

      {!error && forecastCount > 0 && (
        <>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart
              data={points}
              margin={{ top: 14, right: 30, bottom: 0, left: 0 }}
            >
              <defs>
                <linearGradient id="fcActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={ACTUAL} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={ACTUAL} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="fcPred" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={FORECAST} stopOpacity={0.18} />
                  <stop offset="100%" stopColor={FORECAST} stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="ts"
                tickFormatter={timeLabel}
                tickLine={false}
                minTickGap={48}
              />
              <YAxis domain={[0, 100]} tickLine={false} width={44} />
              <Tooltip content={<ForecastTooltip />} />

              {boundary && (
                <ReferenceLine
                  x={boundary}
                  stroke={FORECAST}
                  strokeDasharray="3 3"
                  label={{ value: "now", position: "top", fill: FORECAST,
                           fontSize: 11 }}
                />
              )}

              {/* Forecast sits underneath so the measured line stays
                  readable where the two overlap. */}
              <Area
                type="monotone"
                dataKey="predicted"
                stroke={FORECAST}
                strokeWidth={1.5}
                strokeDasharray="4 4"
                fill="url(#fcPred)"
                isAnimationActive={false}
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="actual"
                stroke={ACTUAL}
                strokeWidth={2}
                fill="url(#fcActual)"
                isAnimationActive={false}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>

          <div className="fc-legend">
            <span>
              <i style={{ background: ACTUAL }} /> Measured
            </span>
            <span>
              <i className="dash" style={{ borderColor: FORECAST }} /> Forecast
              {data?.horizon_s ? ` (+${data.horizon_s}s)` : ""}
            </span>
          </div>

          <div className="fc-stats">
            {row?.scored > 0 ? (
              <>
                <div className="fc-stat">
                  <strong>{row.rmse?.toFixed(2)}</strong>
                  <span>RMSE, percentage points</span>
                </div>
                <div className="fc-stat">
                  <strong>{row.scored}</strong>
                  <span>forecasts scored, last {accuracy.hours}h</span>
                </div>
                {accuracy?.version && (
                  <div className="fc-stat">
                    <strong>{accuracy.version}</strong>
                    <span>active model</span>
                  </div>
                )}
              </>
            ) : (
              <p className="fc-msg">
                Forecasts are being written. Error figures appear once the
                first predicted moment has passed.
              </p>
            )}
          </div>

          <p className="fc-note">
            Measured against forecasts whose moment has already arrived, so
            every point here was predicted before the value existed.
          </p>
        </>
      )}
    </section>
  );
}