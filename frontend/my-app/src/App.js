import { useState, useEffect, useMemo, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from "recharts";
import axios from "axios";
import "./App.css";

const API = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const SEGMENT_COLORS = {
  "Champions": "#8FD9D0",
  "Loyal Customers": "#A8E0B8",
  "At Risk": "#F5AFAF",
  "Hibernating": "#F7D2A0",
  "default": "#B7B2CC"
};

const ACTION_COLORS = {
  retain: "#F5AFAF",
  nurture: "#A8E0B8",
  let_go: "#B7B2CC",
  monitor: "#B6A6E8",
  default: "#8FD9D0"
};

function segmentColor(segment) {
  return SEGMENT_COLORS[segment] || SEGMENT_COLORS.default;
}

function actionColor(code) {
  return ACTION_COLORS[code] || ACTION_COLORS.default;
}

function riskTierColor(pct) {
  if (pct >= 66) return "#F5AFAF";
  if (pct >= 33) return "#F7D2A0";
  return "#A8E0B8";
}

function formatGBP(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return "£" + Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

/* ---------- small building blocks ---------- */

function ThemeToggle({ theme, onToggle }) {
  const isDark = theme === "dark";
  return (
    <button
      className="theme-toggle"
      onClick={onToggle}
      aria-label="Toggle color theme"
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <span className={`theme-toggle-track ${isDark ? "is-dark" : "is-light"}`}>
        <span className="theme-toggle-thumb">
          {isDark ? (
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none">
              <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none">
              <circle cx="12" cy="12" r="4.5" fill="currentColor" />
              <g stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                <path d="M12 2v2.2M12 19.8V22M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2 12h2.2M19.8 12H22M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6" />
              </g>
            </svg>
          )}
        </span>
      </span>
    </button>
  );
}

function SegmentBadge({ segment }) {
  const color = segmentColor(segment);
  return (
    <span className="pill" style={{ color, borderColor: color + "55", backgroundColor: color + "1A" }}>
      {segment || "Unclassified"}
    </span>
  );
}

function ActionBadge({ code, label }) {
  const color = actionColor(code);
  return (
    <span className="pill" style={{ color, borderColor: color + "55", backgroundColor: color + "1A" }}>
      {label}
    </span>
  );
}

function RiskGauge({ value, size = 128 }) {
  const pct = Math.max(0, Math.min(100, value || 0));
  const color = riskTierColor(pct);
  const style = {
    width: size,
    height: size,
    background: `conic-gradient(${color} ${pct * 3.6}deg, var(--surface-3) 0deg)`
  };
  const inner = Math.round(size * 0.76);
  return (
    <div className="risk-gauge" style={style}>
      <div className="risk-gauge-inner" style={{ width: inner, height: inner }}>
        <span className="risk-gauge-value" style={{ color }}>{pct.toFixed(0)}%</span>
        <span className="risk-gauge-label">churn risk</span>
      </div>
    </div>
  );
}

function KpiCard({ label, value, accent, sub, index }) {
  return (
    <div className="kpi-card" style={{ animationDelay: `${index * 60}ms` }}>
      <span className="kpi-label">{label}</span>
      <span className="kpi-value" style={accent ? { color: accent } : undefined}>{value}</span>
      {sub && <span className="kpi-sub">{sub}</span>}
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-label">{label}</p>
      <p className="chart-tooltip-value">{payload[0].value.toLocaleString()}</p>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

/* ---------- main app ---------- */

function App() {
  const [theme, setTheme] = useState(() => {
    const saved = window.localStorage.getItem("csr-theme");
    if (saved) return saved;
    const mediaQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
    return mediaQuery && mediaQuery.matches ? "light" : "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem("csr-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  const [segments, setSegments] = useState([]);
  const [actions, setActions] = useState([]);
  const [retainList, setRetainList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");

  const [customerId, setCustomerId] = useState("");
  const [customer, setCustomer] = useState(null);
  const [lookupError, setLookupError] = useState("");
  const [searching, setSearching] = useState(false);
  const lookupAbortRef = useRef(null);

  const [predictForm, setPredictForm] = useState({
    first_purchase_date: "",
    last_purchase_date: "",
    total_orders: "",
    total_spent: ""
  });
  const [predictResult, setPredictResult] = useState(null);
  const [predictError, setPredictError] = useState("");
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/segments`),
      axios.get(`${API}/actions`),
      axios.get(`${API}/retain`)
    ])
      .then(([segRes, actRes, retRes]) => {
        setSegments(segRes.data);
        setActions(actRes.data);
        setRetainList(retRes.data);
      })
      .catch(() => setLoadError("Couldn't reach the API. Is the backend running?"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!customerId) {
      setCustomer(null);
      setLookupError("");
      return;
    }

    const timeoutId = window.setTimeout(() => {
      if (lookupAbortRef.current) lookupAbortRef.current.abort();
      const controller = new AbortController();
      lookupAbortRef.current = controller;
      setLookupError("");
      setCustomer(null);
      setSearching(true);
      axios.get(`${API}/customer/${customerId}`, { signal: controller.signal })
        .then((r) => {
          if (r.data.error) setLookupError(r.data.error);
          else setCustomer(r.data);
        })
        .catch((error) => {
          if (error.name !== "CanceledError" && error.name !== "AbortError") {
            setLookupError("Customer not found");
          }
        })
        .finally(() => {
          if (lookupAbortRef.current === controller) {
            setSearching(false);
          }
        });
    }, 300);

    return () => {
      window.clearTimeout(timeoutId);
      if (lookupAbortRef.current) {
        lookupAbortRef.current.abort();
        lookupAbortRef.current = null;
      }
    };
  }, [customerId]);

  const searchCustomer = async () => {
    if (!customerId) return;

    if (lookupAbortRef.current) {
      lookupAbortRef.current.abort();
    }

    const controller = new AbortController();
    lookupAbortRef.current = controller;
    setLookupError("");
    setCustomer(null);
    setSearching(true);

    try {
      const r = await axios.get(`${API}/customer/${customerId}`, { signal: controller.signal });
      if (r.data.error) setLookupError(r.data.error);
      else setCustomer(r.data);
    } catch (error) {
      if (error.name !== "CanceledError" && error.name !== "AbortError") {
        setLookupError("Customer not found");
      }
    } finally {
      if (lookupAbortRef.current === controller) {
        setSearching(false);
      }
    }
  };

  const runPrediction = async () => {
    setPredictError("");
    setPredictResult(null);
    setPredicting(true);
    try {
      const r = await axios.post(`${API}/predict`, {
        first_purchase_date: predictForm.first_purchase_date,
        last_purchase_date: predictForm.last_purchase_date,
        total_orders: Number(predictForm.total_orders),
        total_spent: Number(predictForm.total_spent)
      });
      if (r.data.error) setPredictError(r.data.error);
      else setPredictResult(r.data);
    } catch {
      setPredictError("Prediction failed — check the inputs and try again.");
    } finally {
      setPredicting(false);
    }
  };

  const totalCustomers = useMemo(() => segments.reduce((sum, s) => sum + s.count, 0), [segments]);
  const retainCount = retainList.length;
  const revenueAtRisk = useMemo(
    () => retainList.reduce((sum, c) => sum + (c.predicted_ltv || 0), 0),
    [retainList]
  );
  const avgRetainRisk =
    retainCount > 0
      ? (retainList.reduce((sum, c) => sum + (c.churn_probability || 0), 0) / retainCount) * 100
      : 0;

  return (
    <div className="dashboard">
      <div className="bg-glow" aria-hidden="true" />

      <header className="dashboard-header">
        <div>
          <span className="eyebrow">Customer Intelligence</span>
          <h1>Segmentation &amp; Retention</h1>
          <p className="subtitle">
            Live inference over RFM, BG/NBD, Gamma-Gamma &amp; XGBoost churn scoring
          </p>
        </div>
        <div className="header-actions">
          <span className="live-badge">
            <span className="live-dot" />
            Model serving live
          </span>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      {loading ? (
        <div className="loading-state">Loading model outputs…</div>
      ) : loadError ? (
        <div className="panel error-panel">{loadError}</div>
      ) : (
        <>
          <section className="kpi-row">
            <KpiCard index={0} label="Total customers" value={totalCustomers.toLocaleString()} />
            <KpiCard
              index={1}
              label="Retain immediately"
              value={retainCount.toLocaleString()}
              accent="#F87171"
              sub="high LTV + high churn risk"
            />
            <KpiCard
              index={2}
              label="Revenue at risk"
              value={formatGBP(revenueAtRisk)}
              accent="#FBBF24"
              sub="predicted LTV, retain list"
            />
            <KpiCard
              index={3}
              label="Avg. risk in retain list"
              value={`${avgRetainRisk.toFixed(1)}%`}
              accent={riskTierColor(avgRetainRisk)}
            />
          </section>

          <section className="chart-grid">
            <div className="panel">
              <h2>Customer segments</h2>
              <p className="panel-caption">K-Means clustering on RFM features</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={segments}>
                  <CartesianGrid stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="segment" stroke="var(--text-muted)" tick={{ fontSize: 12 }} />
                  <YAxis stroke="var(--text-muted)" tick={{ fontSize: 12 }} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--surface-3)" }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {segments.map((s, i) => (
                      <Cell key={i} fill={segmentColor(s.segment)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="panel">
              <h2>Retention actions</h2>
              <p className="panel-caption">Priority matrix: LTV × churn probability</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={actions}>
                  <CartesianGrid stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="action" stroke="var(--text-muted)" tick={{ fontSize: 12 }} />
                  <YAxis stroke="var(--text-muted)" tick={{ fontSize: 12 }} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--surface-3)" }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {actions.map((a, i) => (
                      <Cell key={i} fill={actionColor(a.action_code || a.code)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel">
            <h2>Customer lookup</h2>
            <p className="panel-caption">Enter a customer ID to run live inference against the trained models</p>
            <div className="inline-controls">
              <input
                className="text-input"
                type="number"
                placeholder="Customer ID"
                value={customerId}
                onChange={(e) => setCustomerId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && searchCustomer()}
              />
              <button className="btn-primary" onClick={searchCustomer} disabled={searching}>
                {searching ? "Searching…" : "Search"}
              </button>
            </div>

            {lookupError && <p className="error-text">{lookupError}</p>}

            {customer && (
              <div className="result-card">
                <RiskGauge value={(customer.churn_probability || 0) * 100} />
                <div className="result-details">
                  <div className="result-row">
                    <span className="result-label">Customer ID</span>
                    <span className="result-value mono">{customer["Customer ID"]}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Segment</span>
                    <SegmentBadge segment={customer.Segment} />
                  </div>
                  <div className="result-row">
                    <span className="result-label">Predicted LTV</span>
                    <span className="result-value mono">{formatGBP(customer.predicted_ltv)}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Recommended action</span>
                    <ActionBadge code={customer.action_code} label={customer.action || customer.action_label} />
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="panel">
            <h2>Predict a new customer</h2>
            <p className="panel-caption">Enter basic purchase history to run live churn &amp; lifetime value inference</p>

            <div className="predict-grid">
              <Field label="First purchase date">
                <input
                  className="text-input"
                  type="date"
                  max={todayISO()}
                  value={predictForm.first_purchase_date}
                  onChange={(e) => setPredictForm((f) => ({ ...f, first_purchase_date: e.target.value }))}
                />
              </Field>
              <Field label="Last purchase date">
                <input
                  className="text-input"
                  type="date"
                  max={todayISO()}
                  value={predictForm.last_purchase_date}
                  onChange={(e) => setPredictForm((f) => ({ ...f, last_purchase_date: e.target.value }))}
                />
              </Field>
              <Field label="Total orders">
                <input
                  className="text-input"
                  type="number"
                  placeholder="e.g. 8"
                  min="1"
                  value={predictForm.total_orders}
                  onChange={(e) => setPredictForm((f) => ({ ...f, total_orders: e.target.value }))}
                />
              </Field>
              <Field label="Total spent (£)">
                <input
                  className="text-input"
                  type="number"
                  placeholder="e.g. 4200"
                  min="0"
                  value={predictForm.total_spent}
                  onChange={(e) => setPredictForm((f) => ({ ...f, total_spent: e.target.value }))}
                />
              </Field>
            </div>

            <button className="btn-primary" onClick={runPrediction} disabled={predicting}>
              {predicting ? "Running inference…" : "Predict churn risk"}
            </button>

            {predictError && <p className="error-text">{predictError}</p>}

            {predictResult && (
              <div className="result-card">
                <RiskGauge value={(predictResult.churn_probability || 0) * 100} />
                <div className="result-details">
                  <div className="result-row">
                    <span className="result-label">Frequency</span>
                    <span className="result-value mono">{predictResult.frequency}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Monetary</span>
                    <span className="result-value mono">{formatGBP(predictResult.monetary)}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Predicted LTV</span>
                    <span className="result-value mono">{formatGBP(predictResult.predicted_ltv)}</span>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Recommended action</span>
                    <ActionBadge code={predictResult.action_code} label={predictResult.action_label || predictResult.action} />
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="panel">
            <h2>Top customers to retain immediately</h2>
            <p className="panel-caption">Ranked by combined LTV and churn risk</p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Customer ID</th>
                    <th>Segment</th>
                    <th>Churn risk</th>
                    <th>Predicted LTV</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {retainList.map((c, i) => {
                    const riskPct = (c.churn_probability || 0) * 100;
                    return (
                      <tr key={i}>
                        <td className="mono">{c["Customer ID"]}</td>
                        <td><SegmentBadge segment={c.Segment} /></td>
                        <td>
                          <div className="risk-cell">
                            <div className="risk-bar-track">
                              <div
                                className="risk-bar-fill"
                                style={{ width: `${riskPct}%`, backgroundColor: riskTierColor(riskPct) }}
                              />
                            </div>
                            <span className="mono risk-cell-value">{riskPct.toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="mono">{formatGBP(c.predicted_ltv)}</td>
                        <td><ActionBadge code={c.action_code} label={c.action} /></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default App;