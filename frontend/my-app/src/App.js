import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from "recharts";
import axios from "axios";
import "./App.css";

const API = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const SEGMENT_COLORS = {
  "Champions": "#4C9EEB",
  "Loyal Customers": "#2DD4A7",
  "At Risk": "#FF6B5B",
  "Hibernating": "#F2B84B",
  "default": "#6B7684"
};

function segmentColor(segment) {
  return SEGMENT_COLORS[segment] || SEGMENT_COLORS.default;
}

function riskTierColor(pct) {
  if (pct >= 66) return "#FF6B5B";
  if (pct >= 33) return "#F2B84B";
  return "#2DD4A7";
}

function formatGBP(value) {
  if (value === undefined || value === null) return "—";
  return "£" + Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function SegmentBadge({ segment }) {
  const color = segmentColor(segment);
  return (
    <span
      className="segment-badge"
      style={{
        color,
        borderColor: color,
        backgroundColor: color + "1A"
      }}
    >
      {segment || "Unclassified"}
    </span>
  );
}

function ActionBadge({ action }) {
  return <span className="action-badge">{action}</span>;
}

function RiskGauge({ value }) {
  const pct = Math.max(0, Math.min(100, value));
  const color = riskTierColor(pct);
  const style = {
    background: `conic-gradient(${color} ${pct * 3.6}deg, var(--surface-2) 0deg)`
  };
  return (
    <div className="risk-gauge" style={style}>
      <div className="risk-gauge-inner">
        <span className="risk-gauge-value" style={{ color }}>{pct.toFixed(0)}%</span>
        <span className="risk-gauge-label">churn risk</span>
      </div>
    </div>
  );
}

function KpiCard({ label, value, accent, sub }) {
  return (
    <div className="kpi-card">
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

// ── New: prediction form for a brand-new customer ──
function PredictPanel() {
  const [form, setForm] = useState({
    first_purchase_date: "",
    last_purchase_date: "",
    total_orders: "",
    total_spent: ""
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const updateField = (field) => (e) => {
    setForm({ ...form, [field]: e.target.value });
  };

  const isValid =
    form.first_purchase_date &&
    form.last_purchase_date &&
    Number(form.total_orders) > 0 &&
    Number(form.total_spent) >= 0;

  const submitPrediction = async () => {
    if (!isValid) {
      setError("Please fill in all fields with valid values.");
      return;
    }
    setError("");
    setResult(null);
    setSubmitting(true);
    try {
      const r = await axios.post(`${API}/predict`, {
        first_purchase_date: form.first_purchase_date,
        last_purchase_date: form.last_purchase_date,
        total_orders: Number(form.total_orders),
        total_spent: Number(form.total_spent)
      });
      if (r.data.error) setError(r.data.error);
      else setResult(r.data);
    } catch {
      setError("Something went wrong running the prediction. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="panel lookup-panel">
      <h2>Predict a new customer</h2>
      <p className="panel-caption">
        Enter basic purchase history to run live churn &amp; lifetime value inference
      </p>

      <div className="predict-form">
        <label className="predict-field">
          <span className="predict-field-label">First purchase date</span>
          <input
            type="date"
            value={form.first_purchase_date}
            onChange={updateField("first_purchase_date")}
          />
        </label>

        <label className="predict-field">
          <span className="predict-field-label">Last purchase date</span>
          <input
            type="date"
            value={form.last_purchase_date}
            onChange={updateField("last_purchase_date")}
          />
        </label>

        <label className="predict-field">
          <span className="predict-field-label">Total orders</span>
          <input
            type="number"
            min="1"
            placeholder="e.g. 8"
            value={form.total_orders}
            onChange={updateField("total_orders")}
          />
        </label>

        <label className="predict-field">
          <span className="predict-field-label">Total spent (£)</span>
          <input
            type="number"
            min="0"
            step="0.01"
            placeholder="e.g. 4200"
            value={form.total_spent}
            onChange={updateField("total_spent")}
          />
        </label>
      </div>

      <button
        className="predict-submit"
        onClick={submitPrediction}
        disabled={submitting}
      >
        {submitting ? "Running inference…" : "Predict churn risk"}
      </button>

      {error && <p className="error-text">{error}</p>}

      {result && (
        <div className="customer-result">
          <RiskGauge value={(result.churn_probability || 0) * 100} />
          <div className="customer-details">
            <div className="customer-detail-row">
              <span className="detail-label">Frequency</span>
              <span className="detail-value mono">{result.frequency}</span>
            </div>
            <div className="customer-detail-row">
              <span className="detail-label">Monetary</span>
              <span className="detail-value mono">{formatGBP(result.monetary)}</span>
            </div>
            <div className="customer-detail-row">
              <span className="detail-label">Predicted LTV</span>
              <span className="detail-value mono">{formatGBP(result.predicted_ltv)}</span>
            </div>
            <div className="customer-detail-row">
              <span className="detail-label">Recommended action</span>
              <ActionBadge action={result.action} />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function App() {
  const [segments, setSegments] = useState([]);
  const [actions, setActions] = useState([]);
  const [retainList, setRetainList] = useState([]);
  const [customerId, setCustomerId] = useState("");
  const [customer, setCustomer] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);

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
      .finally(() => setLoading(false));
  }, []);

  const searchCustomer = async () => {
    if (!customerId) return;
    setError("");
    setCustomer(null);
    setSearching(true);
    try {
      const r = await axios.get(`${API}/customer/${customerId}`);
      if (r.data.error) setError(r.data.error);
      else setCustomer(r.data);
    } catch {
      setError("Customer not found");
    } finally {
      setSearching(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") searchCustomer();
  };

  const totalCustomers = segments.reduce((sum, s) => sum + s.count, 0);
  const retainCount = retainList.length;
  const revenueAtRisk = retainList.reduce((sum, c) => sum + (c.predicted_ltv || 0), 0);
  const avgRetainRisk =
    retainCount > 0
      ? (retainList.reduce((sum, c) => sum + (c.churn_probability || 0), 0) / retainCount) * 100
      : 0;

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div>
          <h1>Customer Segmentation &amp; Retention</h1>
          <p className="subtitle">Live model inference over RFM, BG/NBD, Gamma-Gamma &amp; XGBoost churn scoring</p>
        </div>
        <span className="live-badge">
          <span className="live-dot" />
          Model serving live
        </span>
      </header>

      {loading ? (
        <div className="loading-state">Loading model outputs…</div>
      ) : (
        <>
          <section className="kpi-row">
            <KpiCard label="Total customers" value={totalCustomers.toLocaleString()} />
            <KpiCard
              label="Retain immediately"
              value={retainCount.toLocaleString()}
              accent="#FF6B5B"
              sub="high LTV + high churn risk"
            />
            <KpiCard
              label="Revenue at risk"
              value={formatGBP(revenueAtRisk)}
              accent="#F2B84B"
              sub="predicted LTV, retain list"
            />
            <KpiCard
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
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--surface-2)" }} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
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
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--surface-2)" }} />
                  <Bar dataKey="count" fill="#4C9EEB" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel lookup-panel">
            <h2>Customer lookup</h2>
            <p className="panel-caption">Enter a customer ID to run live inference against the trained models</p>
            <div className="lookup-controls">
              <input
                type="number"
                placeholder="Customer ID"
                value={customerId}
                onChange={(e) => setCustomerId(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button onClick={searchCustomer} disabled={searching}>
                {searching ? "Searching…" : "Search"}
              </button>
            </div>

            {error && <p className="error-text">{error}</p>}

            {customer && (
              <div className="customer-result">
                <RiskGauge value={(customer.churn_probability || 0) * 100} />
                <div className="customer-details">
                  <div className="customer-detail-row">
                    <span className="detail-label">Customer ID</span>
                    <span className="detail-value mono">{customer["Customer ID"]}</span>
                  </div>
                  <div className="customer-detail-row">
                    <span className="detail-label">Segment</span>
                    <SegmentBadge segment={customer.Segment} />
                  </div>
                  <div className="customer-detail-row">
                    <span className="detail-label">Predicted LTV</span>
                    <span className="detail-value mono">{formatGBP(customer.predicted_ltv)}</span>
                  </div>
                  <div className="customer-detail-row">
                    <span className="detail-label">Recommended action</span>
                    <ActionBadge action={customer.action} />
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* ── New section ── */}
          <PredictPanel />

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
                        <td><ActionBadge action={c.action} /></td>
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