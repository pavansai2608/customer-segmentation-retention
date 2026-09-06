import { useState, useEffect, useMemo, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList
} from "recharts";
import axios from "axios";
import "./App.css";

const API = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

/* ---------- design tokens, resolved at render ---------- */

const SEGMENT_TONE = {
  "Champions": "accent",
  "Loyal Customers": "plain",
  "At Risk": "warn",
  "Hibernating": "neutral"
};

const SEGMENT_FILL = {
  "Champions": "var(--chart-accent)",
  "Loyal Customers": "var(--chart-dim)",
  "At Risk": "var(--chart-warn)",
  "Hibernating": "var(--chart-neutral)"
};

const ACTION_TONE = {
  retain: "warn",
  nurture: "accent",
  monitor: "plain",
  let_go: "neutral"
};

const ACTION_FILL = {
  retain: "var(--chart-warn)",
  nurture: "var(--chart-accent)",
  monitor: "var(--chart-dim)",
  let_go: "var(--chart-neutral)"
};

const ACTION_CODE_BY_LABEL = {
  "retain immediately": "retain",
  "let go": "let_go",
  "nurture": "nurture",
  "monitor": "monitor"
};

// Top of the retain list in models/final_decision_matrix.csv — prefilled so the
// lookup demonstrates itself on first load.
const DEFAULT_CUSTOMER_ID = "13093";

const SECTIONS = [
  { id: "overview", label: "Overview", icon: "Overview" },
  { id: "analysis", label: "Distribution", icon: "Distribution" },
  { id: "lookup", label: "Lookup", icon: "Lookup" },
  { id: "predict", label: "Predict", icon: "Predict" },
  { id: "retain", label: "Retain list", icon: "Retain" }
];

/* Server labels ship with emoji prefixes; icons here are drawn, so strip them. */
function plainLabel(label) {
  if (!label) return "";
  return String(label)
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE0F}]/gu, "")
    .replace(/\s+/g, " ")
    .trim();
}

function actionCode(row) {
  if (!row) return "let_go";
  const direct = row.action_code || row.code;
  if (direct) return direct;
  const key = plainLabel(row.action || row.action_label).toLowerCase();
  return ACTION_CODE_BY_LABEL[key] || "let_go";
}

function segmentTone(segment) {
  return SEGMENT_TONE[segment] || "neutral";
}

function segmentFill(segment) {
  return SEGMENT_FILL[segment] || "var(--neutral-chart)";
}

function riskTier(pct) {
  if (pct >= 66) return { label: "High", tone: "warn", fill: "var(--chart-warn)", text: "var(--warn)" };
  if (pct >= 33) return { label: "Medium", tone: "plain", fill: "var(--chart-mid)", text: "var(--text)" };
  return { label: "Low", tone: "accent", fill: "var(--chart-accent)", text: "var(--accent-text)" };
}

function formatGBP(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return "£" + Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatCompactGBP(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1000) {
    return "£" + (n / 1000).toLocaleString(undefined, { maximumFractionDigits: 1 }) + "K";
  }
  return formatGBP(n);
}

// A missing `response` means the request never landed — a stopped or
// unreachable backend, not a customer that does not exist.
function lookupErrorFor(error) {
  if (!error || !error.response) return "Couldn't reach the API";
  if (error.response.status === 404) return "Customer not found";
  return "Lookup failed";
}

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function oneYearAgoISO() {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 1);
  return d.toISOString().slice(0, 10);
}

/* ---------- icons: one stroke weight, one grid ---------- */

const iconProps = {
  width: 18, height: 18, viewBox: "0 0 24 24", fill: "none",
  stroke: "currentColor", strokeWidth: 1.35, strokeLinecap: "round", strokeLinejoin: "round",
  "aria-hidden": "true", focusable: "false"
};

const Icon = {
  Overview: (p) => (
    <svg {...iconProps} {...p}><path d="M3 13h5v8H3zM9.5 3h5v18h-5zM16 9h5v12h-5z" /></svg>
  ),
  Distribution: (p) => (
    <svg {...iconProps} {...p}><path d="M4 6h13M4 12h9M4 18h16" /></svg>
  ),
  Lookup: (p) => (
    <svg {...iconProps} {...p}><circle cx="11" cy="11" r="6.5" /><path d="m16 16 4.5 4.5" /></svg>
  ),
  Predict: (p) => (
    <svg {...iconProps} {...p}><path d="M3 15.5 8.5 10l4 4L21 5.5" /><path d="M15.5 5.5H21v5.5" /></svg>
  ),
  Retain: (p) => (
    <svg {...iconProps} {...p}><path d="M12 3 4.5 6v6c0 4.4 3.1 8.2 7.5 9 4.4-.8 7.5-4.6 7.5-9V6z" /><path d="M12 8.5v4" /><path d="M12 15.6h.01" /></svg>
  ),
  Sun: (p) => (
    <svg {...iconProps} {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" /></svg>
  ),
  Moon: (p) => (
    <svg {...iconProps} {...p}><path d="M20.5 13.4A8.5 8.5 0 1 1 10.6 3.5a6.7 6.7 0 0 0 9.9 9.9" /></svg>
  ),
  Alert: (p) => (
    <svg {...iconProps} {...p}><path d="M12 4.5 2.8 20h18.4z" /><path d="M12 10v4" /><path d="M12 17.2h.01" /></svg>
  ),
  Inbox: (p) => (
    <svg {...iconProps} {...p} width={28} height={28}><path d="M3.5 13.5h4l1.5 3h6l1.5-3h4" /><path d="M5.6 5.5h12.8l2.1 8v5H3.5v-5z" /></svg>
  )
};

/* ---------- primitives ---------- */

function Tag({ tone, children }) {
  return (
    <span className={`tag tag-${tone}`}>
      <span className="tag-dot" aria-hidden="true" />
      {children}
    </span>
  );
}

function SegmentBadge({ segment }) {
  return <Tag tone={segmentTone(segment)}>{segment || "Unclassified"}</Tag>;
}

function ActionBadge({ code, label }) {
  const resolved = code || actionCode({ action: label });
  return <Tag tone={ACTION_TONE[resolved] || "neutral"}>{plainLabel(label) || "—"}</Tag>;
}

function Stat({ label, value, note, title }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-figure" title={title}>{value}</span>
      {note && <span className="stat-note">{note}</span>}
    </div>
  );
}

function Panel({ id, title, caption, children, flush }) {
  return (
    <section className="section" id={id} aria-labelledby={id ? `${id}-title` : undefined}>
      <div className="section-head">
        <h2 id={id ? `${id}-title` : undefined}>{title}</h2>
        {caption && <p>{caption}</p>}
      </div>
      <div className="panel">
        {flush ? children : <div className="panel-body">{children}</div>}
      </div>
    </section>
  );
}

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

function ErrorNote({ title, children }) {
  return (
    <div className="note note-error" role="alert">
      <Icon.Alert width={16} height={16} />
      <span>
        <span className="note-title">{title}</span>
        {children && <span className="note-body"> {children}</span>}
      </span>
    </div>
  );
}

function EmptyState({ title, body }) {
  return (
    <div className="empty">
      <Icon.Inbox />
      <p className="empty-title">{title}</p>
      <p className="empty-body">{body}</p>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-label">{plainLabel(label) || label}</p>
      <p className="chart-tooltip-value">{payload[0].value.toLocaleString()} customers</p>
    </div>
  );
}

/* A horizontal band with the reading rendered as a figure, not a dial. */
function RiskReadout({ value }) {
  const pct = Math.max(0, Math.min(100, value || 0));
  const tier = riskTier(pct);
  return (
    <div className="risk">
      <span className="risk-label">Probability of churn</span>
      <div className="risk-top">
        <span className="risk-value" style={{ color: tier.text }}>{pct.toFixed(1)}%</span>
        <Tag tone={tier.tone}>{tier.label}</Tag>
      </div>
      <div
        className="meter"
        role="img"
        aria-label={`Churn probability ${pct.toFixed(1)} percent, ${tier.label} risk`}
      >
        <div
          className="meter-fill"
          style={{ transform: `scaleX(${pct / 100})`, backgroundColor: tier.fill }}
        />
        <span className="meter-tick" style={{ left: "33%" }} aria-hidden="true" />
        <span className="meter-tick" style={{ left: "66%" }} aria-hidden="true" />
      </div>
      <div className="meter-scale" aria-hidden="true">
        <span>Low</span><span>Medium</span><span>High</span>
      </div>
    </div>
  );
}

function Readout({ rows }) {
  return (
    <div className="readout-list">
      {rows.map((row) => (
        <div className="readout-row" key={row.key}>
          <span className="readout-key">{row.key}</span>
          <span className={`readout-val${row.mono ? " num" : ""}`}>{row.value}</span>
        </div>
      ))}
    </div>
  );
}

function CategoryChart({ data, dataKey, categoryKey, fillFor, formatCategory }) {
  const height = Math.max(170, data.length * 58);
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ top: 2, right: 52, bottom: 2, left: 0 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category"
            dataKey={categoryKey}
            width={146}
            axisLine={false}
            tickLine={false}
            tickFormatter={formatCategory}
            tick={{ fill: "var(--text-2)", fontSize: 13.5, fontWeight: 500, letterSpacing: "-0.01em" }}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--surface-3)" }} />
          <Bar dataKey={dataKey} radius={7} barSize={14} isAnimationActive={false}>
            {data.map((entry, i) => (
              <Cell key={i} fill={fillFor(entry)} />
            ))}
            <LabelList
              dataKey={dataKey}
              position="right"
              offset={14}
              fill="var(--text-3)"
              fontSize={12.5}
              fontWeight={500}
              formatter={(v) => Number(v).toLocaleString()}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- loading scaffolding ---------- */

function Skeletons() {
  return (
    <>
      <div className="hero" aria-hidden="true">
        <div>
          <div className="skeleton sk-line" style={{ width: "34%" }} />
          <div className="skeleton sk-hero" style={{ width: "70%", marginTop: 20 }} />
        </div>
        <div className="hero-side">
          {[0, 1, 2].map((i) => (
            <div className="stat" key={i}>
              <div className="skeleton sk-line" style={{ width: "52%" }} />
              <div className="skeleton sk-stat" style={{ marginTop: 10 }} />
            </div>
          ))}
        </div>
      </div>
      <div className="grid-2" aria-hidden="true">
        {[0, 1].map((i) => (
          <div className="panel" key={i}>
            <div className="panel-body">
              <div className="skeleton sk-line" style={{ width: "36%" }} />
              <div className="skeleton sk-chart" style={{ marginTop: 24 }} />
            </div>
          </div>
        ))}
      </div>
      <div className="panel" aria-hidden="true">
        <div className="panel-body">
          {[0, 1, 2, 3, 4, 5].map((i) => <div className="skeleton sk-row" key={i} />)}
        </div>
      </div>
    </>
  );
}

/* ---------- main app ---------- */

function App() {
  const [theme, setTheme] = useState(() => {
    const saved = window.localStorage.getItem("csr-theme");
    return saved || "dark";
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

  const [customerId, setCustomerId] = useState(DEFAULT_CUSTOMER_ID);
  const [customer, setCustomer] = useState(null);
  const [lookupError, setLookupError] = useState("");
  const [searching, setSearching] = useState(false);
  const lookupAbortRef = useRef(null);

  const [predictForm, setPredictForm] = useState({
    first_purchase_date: oneYearAgoISO(),
    last_purchase_date: todayISO(),
    total_orders: "1",
    total_spent: "0"
  });
  const [predictResult, setPredictResult] = useState(null);
  const [predictError, setPredictError] = useState("");
  const [predicting, setPredicting] = useState(false);

  const [activeSection, setActiveSection] = useState("overview");

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
            setLookupError(lookupErrorFor(error));
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
        setLookupError(lookupErrorFor(error));
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

  /* Highlight the section currently in view. Purely navigational. */
  useEffect(() => {
    if (loading || loadError) return undefined;
    if (typeof IntersectionObserver === "undefined") return undefined;

    const elements = SECTIONS
      .map((s) => document.getElementById(s.id))
      .filter(Boolean);
    if (!elements.length) return undefined;

    // The callback only reports sections whose visibility just changed, so the
    // running set is kept here and the decision is made from all of it.
    const visible = new Set();

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) visible.add(e.target.id);
          else visible.delete(e.target.id);
        });

        const candidates = SECTIONS
          .filter((s) => visible.has(s.id))
          .map((s) => {
            const node = document.getElementById(s.id);
            return { id: s.id, top: node ? node.getBoundingClientRect().top : 0 };
          })
          .sort((a, b) => b.top - a.top);
        if (!candidates.length) return;

        // Active is the last section whose top has crossed the line, not
        // whichever started highest up the page. The line sits a third of the
        // way down so the final section can still win at the page bottom,
        // where there is no scroll left to give it.
        const line = window.innerHeight * 0.35;
        const crossed = candidates.find((c) => c.top <= line);
        setActiveSection((crossed || candidates[candidates.length - 1]).id);
      },
      { rootMargin: "-72px 0px -55% 0px", threshold: 0 }
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [loading, loadError]);

  const goToSection = (id) => {
    setActiveSection(id);
    const el = document.getElementById(id);
    if (el && typeof el.scrollIntoView === "function") {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const isDark = theme === "dark";
  const avgTier = riskTier(avgRetainRisk);
  const navCounts = { retain: retainCount || null };

  return (
    <div className="app">
      <a className="skip-link" href="#overview">Skip to content</a>

      <aside className="rail">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="8" height="8" rx="2" fill="currentColor" opacity="0.55" />
              <rect x="13" y="3" width="8" height="8" rx="2" fill="currentColor" />
              <rect x="3" y="13" width="8" height="8" rx="2" fill="currentColor" />
              <rect x="13" y="13" width="8" height="8" rx="2" fill="currentColor" opacity="0.35" />
            </svg>
          </span>
          <span className="brand-text">
            <span className="brand-name">Customer Intelligence</span>
            <span className="brand-sub">Retention analytics</span>
          </span>
        </div>

        <nav className="nav" aria-label="Dashboard sections">
          <span className="nav-label">Workspace</span>
          {SECTIONS.map((s) => {
            const Glyph = Icon[s.icon];
            const count = navCounts[s.id];
            return (
              <button
                key={s.id}
                type="button"
                className={`nav-item${activeSection === s.id ? " is-active" : ""}`}
                aria-current={activeSection === s.id ? "true" : undefined}
                aria-label={s.label}
                title={s.label}
                onClick={() => goToSection(s.id)}
              >
                {Glyph ? <Glyph /> : null}
                <span>{s.label}</span>
                {count ? <span className="nav-count">{count}</span> : null}
              </button>
            );
          })}
        </nav>

        <div className="rail-foot">
          <div className={`status${loadError ? " is-down" : ""}`}>
            <span className="status-dot" aria-hidden="true" />
            {loadError ? "API unreachable" : "Model serving live"}
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-titles">
            <h1>Segmentation &amp; Retention</h1>
            <p className="topbar-sub">RFM · BG/NBD · Gamma-Gamma · XGBoost churn scoring</p>
          </div>
          <div className="topbar-actions">
            <button
              type="button"
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
              title={isDark ? "Switch to light theme" : "Switch to dark theme"}
            >
              {isDark ? <Icon.Sun /> : <Icon.Moon />}
            </button>
          </div>
        </header>

        <main className="content" id="content">
          {loading ? (
            <Skeletons />
          ) : loadError ? (
            <ErrorNote title="Couldn't load model outputs.">
              {loadError} Start it with <code>uvicorn main:app --reload</code> from the
              {" "}<code>backend/</code> directory, then reload this page.
            </ErrorNote>
          ) : (
            <div className="fade-in" style={{ display: "contents" }}>
              <section className="section hero" id="overview" aria-label="Portfolio overview">
                <div className="hero-lead">
                  <p className="hero-label">Retain immediately</p>
                  <p className="hero-figure">{retainCount.toLocaleString()}</p>
                  <p className="hero-note">
                    customers holding high predicted lifetime value while trending toward churn.
                    Everything below ranks them.
                  </p>
                </div>
                <div className="hero-side">
                  <Stat
                    label="Revenue exposure"
                    value={formatCompactGBP(revenueAtRisk)}
                    title={formatGBP(revenueAtRisk)}
                    note="predicted LTV of that list"
                  />
                  <Stat
                    label="Customers scored"
                    value={totalCustomers.toLocaleString()}
                    note="across four RFM segments"
                  />
                  <Stat
                    label="Avg. churn probability"
                    value={`${avgRetainRisk.toFixed(1)}%`}
                    note={`${avgTier.label} band`}
                  />
                </div>
              </section>

              <section className="section" id="analysis" aria-labelledby="analysis-title">
                <div className="section-head">
                  <h2 id="analysis-title">Distribution</h2>
                  <p>Where the scored book sits today, and what the model recommends doing about it</p>
                </div>
                <div className="grid-2">
                  <div className="panel">
                    <div className="panel-body">
                      <p className="panel-title">Customer segments</p>
                      <p className="panel-caption">K-Means clustering over recency, frequency and monetary value</p>
                      {segments.length ? (
                        <CategoryChart
                          data={segments}
                          dataKey="count"
                          categoryKey="segment"
                          fillFor={(s) => segmentFill(s.segment)}
                        />
                      ) : (
                        <EmptyState
                          title="No segments returned"
                          body="The decision matrix loaded without any segment rows."
                        />
                      )}
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-body">
                      <p className="panel-title">Recommended actions</p>
                      <p className="panel-caption">Priority matrix: predicted LTV against churn probability</p>
                      {actions.length ? (
                        <CategoryChart
                          data={actions}
                          dataKey="count"
                          categoryKey="action"
                          formatCategory={plainLabel}
                          fillFor={(a) => ACTION_FILL[actionCode(a)] || "var(--neutral-chart)"}
                        />
                      ) : (
                        <EmptyState
                          title="No actions returned"
                          body="The decision matrix loaded without any action rows."
                        />
                      )}
                    </div>
                  </div>
                </div>
              </section>

              <Panel
                id="lookup"
                title="Customer lookup"
                caption="Enter an existing customer ID to re-score them against the trained models"
              >
                <div className="search-row">
                  <div className="search-field">
                    <span className="search-icon"><Icon.Lookup width={16} height={16} /></span>
                    <input
                      className="input"
                      type="number"
                      placeholder="Customer ID"
                      aria-label="Customer ID"
                      value={customerId}
                      onChange={(e) => setCustomerId(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && searchCustomer()}
                    />
                  </div>
                  <button className="btn" onClick={searchCustomer} disabled={searching}>
                    {searching && <span className="spinner" aria-hidden="true" />}
                    {searching ? "Searching" : "Search"}
                  </button>
                </div>
                <p className="search-hint">
                  Prefilled with {DEFAULT_CUSTOMER_ID}, the highest predicted value on the
                  retain list. Replace it with any scored customer ID.
                </p>

                <div aria-live="polite">
                  {lookupError && (
                    <div className="form-error">
                      <ErrorNote title={lookupError}>
                        {lookupError === "Customer not found"
                          ? "No scored record matches that ID. Try one from the list further down the page."
                          : "The scoring API did not respond. Check that the backend is running, then search again."}
                      </ErrorNote>
                    </div>
                  )}

                  {customer && (
                    <div className="readout">
                      <RiskReadout value={(customer.churn_probability || 0) * 100} />
                      <Readout
                        rows={[
                          { key: "Customer ID", value: customer["Customer ID"], mono: true },
                          { key: "Segment", value: <SegmentBadge segment={customer.Segment} /> },
                          { key: "Predicted LTV", value: formatGBP(customer.predicted_ltv), mono: true },
                          {
                            key: "Recommended action",
                            value: (
                              <ActionBadge
                                code={customer.action_code}
                                label={customer.action || customer.action_label}
                              />
                            )
                          }
                        ]}
                      />
                    </div>
                  )}
                </div>
              </Panel>

              <Panel
                id="predict"
                title="Score a new customer"
                caption="Enter basic purchase history to run live churn and lifetime value inference"
              >
                <div className="predict-grid">
                  <Field label="First purchase date">
                    <input
                      className="input"
                      type="date"
                      max={todayISO()}
                      value={predictForm.first_purchase_date}
                      onChange={(e) => setPredictForm((f) => ({ ...f, first_purchase_date: e.target.value }))}
                    />
                  </Field>
                  <Field label="Last purchase date">
                    <input
                      className="input"
                      type="date"
                      max={todayISO()}
                      value={predictForm.last_purchase_date}
                      onChange={(e) => setPredictForm((f) => ({ ...f, last_purchase_date: e.target.value }))}
                    />
                  </Field>
                  <Field label="Total orders" hint="Whole orders placed to date">
                    <input
                      className="input"
                      type="number"
                      placeholder="8"
                      min="1"
                      value={predictForm.total_orders}
                      onChange={(e) => setPredictForm((f) => ({ ...f, total_orders: e.target.value }))}
                    />
                  </Field>
                  <Field label="Total spent (£)" hint="Lifetime revenue, excluding returns">
                    <input
                      className="input"
                      type="number"
                      placeholder="4200"
                      min="0"
                      value={predictForm.total_spent}
                      onChange={(e) => setPredictForm((f) => ({ ...f, total_spent: e.target.value }))}
                    />
                  </Field>
                </div>

                <button className="btn" onClick={runPrediction} disabled={predicting}>
                  {predicting && <span className="spinner" aria-hidden="true" />}
                  {predicting ? "Running inference" : "Predict churn risk"}
                </button>

                <div aria-live="polite">
                  {predictError && (
                    <div className="form-error">
                      <ErrorNote title="Prediction failed.">{predictError}</ErrorNote>
                    </div>
                  )}

                  {predictResult && (
                    <div className="readout">
                      <RiskReadout value={(predictResult.churn_probability || 0) * 100} />
                      <Readout
                        rows={[
                          { key: "Frequency", value: predictResult.frequency, mono: true },
                          { key: "Monetary", value: formatGBP(predictResult.monetary), mono: true },
                          { key: "Predicted LTV", value: formatGBP(predictResult.predicted_ltv), mono: true },
                          {
                            key: "Recommended action",
                            value: (
                              <ActionBadge
                                code={predictResult.action_code}
                                label={predictResult.action_label || predictResult.action}
                              />
                            )
                          }
                        ]}
                      />
                    </div>
                  )}
                </div>
              </Panel>

              <Panel
                id="retain"
                title="Retain immediately"
                caption="Highest predicted lifetime value among customers likely to churn"
                flush
              >
                {retainList.length ? (
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th scope="col" className="cell-rank">#</th>
                          <th scope="col">Customer</th>
                          <th scope="col">Segment</th>
                          <th scope="col" className="col-num">Churn risk</th>
                          <th scope="col" className="col-num">Predicted LTV</th>
                          <th scope="col">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {retainList.map((c, i) => {
                          const riskPct = (c.churn_probability || 0) * 100;
                          const tier = riskTier(riskPct);
                          return (
                            <tr key={i}>
                              <td className="cell-rank num">{i + 1}</td>
                              <td className="cell-id num">{c["Customer ID"]}</td>
                              <td><SegmentBadge segment={c.Segment} /></td>
                              <td>
                                <div className="risk-cell">
                                  <span className="risk-track">
                                    <span
                                      className="risk-track-fill"
                                      style={{ width: `${riskPct}%`, backgroundColor: tier.fill }}
                                    />
                                  </span>
                                  <span className="risk-num num">{riskPct.toFixed(1)}%</span>
                                </div>
                              </td>
                              <td className="col-num num">{formatGBP(c.predicted_ltv)}</td>
                              <td><ActionBadge code={c.action_code} label={c.action} /></td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <EmptyState
                    title="Nothing needs rescuing right now"
                    body="No customer currently sits above both the churn threshold and the lifetime value median. Re-run the pipeline after the next data refresh to update this list."
                  />
                )}
              </Panel>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
