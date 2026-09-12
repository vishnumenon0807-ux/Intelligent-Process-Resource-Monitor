import { useCallback, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "./api";

const POLL_MS = 5000;
const WINDOWS = ["15m", "1h", "6h", "24h", "7d"];
const CATEGORIES = [
  ["", "All"],
  ["cpu", "CPU"],
  ["memory", "Memory"],
  ["disk", "Disk"],
  ["startup", "Startup"],
];

function band(confidence) {
  if (confidence >= 0.75) return "var(--band-high)";
  if (confidence >= 0.5) return "var(--band-mid)";
  return "var(--band-low)";
}

function clockTime(iso) {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function since(iso) {
  const mins = Math.round((Date.now() - new Date(iso)) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  return hrs < 24 ? `${hrs} hr ago` : `${Math.round(hrs / 24)} d ago`;
}

/* --------------------------------------------------------------- rail */
function Gauge({ label, value, color }) {
  const pct = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div className="gauge">
      <div className="gauge-head">
        <span className="gauge-label">{label}</span>
        <span className="gauge-value num">
          {value == null ? "--" : `${pct.toFixed(0)}%`}
        </span>
      </div>
      <div className="gauge-track">
        <div
          className="gauge-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function Rail({ latest }) {
  const m = latest ?? {};
  const rows = [
    ["Processes", m.process_count],
    ["Threads", m.thread_count_total],
    ["RAM free", m.ram_available_mb && `${(m.ram_available_mb / 1024).toFixed(1)} GB`],
    ["CPU temp", m.cpu_temp_c && `${m.cpu_temp_c.toFixed(0)}°C`],
    ["GPU temp", m.gpu_temp_c && `${m.gpu_temp_c.toFixed(0)}°C`],
    ["Battery", m.battery_percent != null && `${m.battery_percent.toFixed(0)}%`],
  ].filter(([, v]) => v != null && v !== false && v !== undefined);

  return (
    <aside className="rail">
      <h1>Resource Monitor</h1>
      <div className="sub">
        {latest ? `Updated ${clockTime(latest.ts)}` : "Waiting for data"}
      </div>

      <Gauge label="CPU" value={m.cpu_percent} color="var(--cpu)" />
      <Gauge label="Memory" value={m.ram_percent} color="var(--ram)" />
      <Gauge label="Disk" value={m.disk_percent} color="var(--band-mid)" />
      {m.gpu_percent != null && (
        <Gauge label="GPU" value={m.gpu_percent} color="var(--band-low)" />
      )}
      {m.swap_percent > 0 && (
        <Gauge label="Swap" value={m.swap_percent} color="var(--band-high)" />
      )}

      <div className="readouts">
        {rows.map(([k, v]) => (
          <div className="readout" key={k}>
            <span>{k}</span>
            <span className="num">{v}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}

/* -------------------------------------------------------------- chart */
function History({ data, window, onWindow }) {
  const shaped = data.map((d) => ({
    ...d,
    label: new Date(d.ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>System load</h3>
        <div className="windows">
          {WINDOWS.map((w) => (
            <button
              key={w}
              aria-pressed={w === window}
              onClick={() => onWindow(w)}
            >
              {w}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={shaped} margin={{ top: 4, right: 4, bottom: 0, left: -22 }}>
          <defs>
            <linearGradient id="gCpu" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--cpu)" stopOpacity={0.28} />
              <stop offset="100%" stopColor="var(--cpu)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gRam" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--ram)" stopOpacity={0.24} />
              <stop offset="100%" stopColor="var(--ram)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#222c37" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: "#5f6c7a", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#26313d" }}
            minTickGap={44}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: "#5f6c7a", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          <Tooltip
            contentStyle={{
              background: "#1c242f",
              border: "1px solid #26313d",
              borderRadius: 3,
              fontSize: 12,
            }}
            labelStyle={{ color: "#8b99a8" }}
            formatter={(v, n) => [`${Number(v).toFixed(1)}%`, n === "cpu_percent" ? "CPU" : "Memory"]}
          />
          <Area
            type="monotone"
            dataKey="cpu_percent"
            stroke="var(--cpu)"
            strokeWidth={1.5}
            fill="url(#gCpu)"
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="ram_percent"
            stroke="var(--ram)"
            strokeWidth={1.5}
            fill="url(#gRam)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="legend">
        <span><i style={{ background: "var(--cpu)" }} />CPU</span>
        <span><i style={{ background: "var(--ram)" }} />Memory</span>
      </div>
    </section>
  );
}

/* ----------------------------------------------------------- advisory */
function Advisory({ item, onFeedback }) {
  const [open, setOpen] = useState(false);

  return (
    <article className="item">
      <button
        className="item-row"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className="edge" style={{ background: band(item.confidence) }} />
        <span className="item-title">
          {item.title}
          <span className="item-meta">
            {item.target_name ? `${item.target_name}, ` : ""}
            {since(item.created_at)}
            {item.outcome === "resolved" && ", condition cleared"}
          </span>
        </span>
        <span className="conf num">{Math.round(item.confidence * 100)}%</span>
        <span className="chevron">{open ? "\u2212" : "+"}</span>
      </button>

      {open && (
        <div className="detail">
          <dl>
            <dt>What is happening</dt>
            <dd>{item.diagnosis}</dd>
            <dt>Why</dt>
            <dd>{item.cause}</dd>
          </dl>

          <dt style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 12 }}>
            What you can do
          </dt>
          <ol>
            {item.suggestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>

          <div className="feedback">
            <span>Was this useful?</span>
            <button
              aria-pressed={item.user_feedback === "helpful"}
              onClick={() => onFeedback(item.id, "helpful")}
            >
              Yes
            </button>
            <button
              aria-pressed={item.user_feedback === "not_helpful"}
              onClick={() => onFeedback(item.id, "not_helpful")}
            >
              No
            </button>
          </div>
        </div>
      )}
    </article>
  );
}

function Feed({ items, category, onCategory, onFeedback }) {
  return (
    <section>
      <div className="feed-head">
        <h3>Advisories</h3>
        <div className="filters">
          {CATEGORIES.map(([value, label]) => (
            <button
              key={label}
              aria-pressed={value === category}
              onClick={() => onCategory(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="feed">
        {items.length === 0 ? (
          <div className="empty">
            <p>Nothing to report.</p>
            <p className="hint">
              Advisories appear when a process crosses a threshold the rules
              recognise. Put the machine under load to see it work.
            </p>
          </div>
        ) : (
          items.map((it) => (
            <Advisory key={it.id} item={it} onFeedback={onFeedback} />
          ))
        )}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- app */
export default function App() {
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [items, setItems] = useState([]);
  const [window_, setWindow] = useState("1h");
  const [category, setCategory] = useState("");
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [l, h, a] = await Promise.all([
        api.latest(),
        api.metrics(window_),
        api.advisories({ category: category || undefined, limit: 40 }),
      ]);
      setLatest(l);
      setHistory(h);
      setItems(a);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [window_, category]);

  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  const handleFeedback = async (id, feedback) => {
    setItems((prev) =>
      prev.map((it) => (it.id === id ? { ...it, user_feedback: feedback } : it))
    );
    try {
      await api.sendFeedback(id, feedback);
    } catch {
      load();
    }
  };

  return (
    <div className="shell">
      <Rail latest={latest} />

      <main className="main">
        <div className="topline">
          <h2>Live system state</h2>
          <div className="status">
            <span className={`dot${error ? " down" : ""}`} />
            {error ? "API unreachable" : "Connected"}
          </div>
        </div>

        {error && (
          <div className="error">
            Cannot reach the API. Check that uvicorn is running on port 8000.
            <br />
            <code>{error}</code>
          </div>
        )}

        <History data={history} window={window_} onWindow={setWindow} />

        <Feed
          items={items}
          category={category}
          onCategory={setCategory}
          onFeedback={handleFeedback}
        />
      </main>
    </div>
  );
}