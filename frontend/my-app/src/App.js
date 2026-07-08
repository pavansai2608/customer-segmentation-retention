import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";

const COLORS = {
  "Champions": "#2ecc71",
  "Loyal Customers": "#27ae60",
  "At Risk": "#e74c3c",
  "Hibernating": "#f39c12"
};

function App() {
  const [segments, setSegments] = useState([]);
  const [actions, setActions] = useState([]);
  const [retainList, setRetainList] = useState([]);
  const [customerId, setCustomerId] = useState("");
  const [customer, setCustomer] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios.get(`${API}/segments`).then(r => setSegments(r.data));
    axios.get(`${API}/actions`).then(r => setActions(r.data));
    axios.get(`${API}/retain`).then(r => setRetainList(r.data));
  }, []);

  const searchCustomer = async () => {
    setError("");
    setCustomer(null);
    try {
      const r = await axios.get(`${API}/customer/${customerId}`);
      if (r.data.error) setError(r.data.error);
      else setCustomer(r.data);
    } catch {
      setError("Customer not found");
    }
  };

  return (
    <div style={{ fontFamily: "Arial", padding: "20px", background: "#f5f5f5", minHeight: "100vh" }}>
      <h1 style={{ color: "#2c3e50" }}>Customer Segmentation & Retention Dashboard</h1>

      {/* Segment Chart */}
      <div style={{ background: "white", padding: "20px", borderRadius: "8px", marginBottom: "20px" }}>
        <h2>Customer Segments (K-Means)</h2>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={segments}>
            <XAxis dataKey="segment" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count">
              {segments.map((s, i) => (
                <Cell key={i} fill={COLORS[s.segment] || "#3498db"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Action Summary */}
      <div style={{ background: "white", padding: "20px", borderRadius: "8px", marginBottom: "20px" }}>
        <h2>Retention Action Summary</h2>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={actions}>
            <XAxis dataKey="action" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3498db" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Customer Search */}
      <div style={{ background: "white", padding: "20px", borderRadius: "8px", marginBottom: "20px" }}>
        <h2>Customer Lookup</h2>
        <input
          type="number"
          placeholder="Enter Customer ID"
          value={customerId}
          onChange={e => setCustomerId(e.target.value)}
          style={{ padding: "8px", marginRight: "10px", width: "200px" }}
        />
        <button onClick={searchCustomer} style={{ padding: "8px 16px", background: "#3498db", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}>
          Search
        </button>
        {error && <p style={{ color: "red" }}>{error}</p>}
        {customer && (
          <div style={{ marginTop: "15px", padding: "15px", background: "#f8f9fa", borderRadius: "4px" }}>
            <p><b>Customer ID:</b> {customer["Customer ID"]}</p>
            <p><b>Segment:</b> {customer.Segment}</p>
            <p><b>Churn Probability:</b> {(customer.churn_probability * 100).toFixed(1)}%</p>
            <p><b>Predicted LTV:</b> £{customer.predicted_ltv?.toFixed(2)}</p>
            <p><b>Recommended Action:</b> {customer.action}</p>
          </div>
        )}
      </div>

      {/* Top Retain Customers */}
      <div style={{ background: "white", padding: "20px", borderRadius: "8px" }}>
        <h2>🔴 Top Customers to Retain Immediately</h2>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "#e74c3c", color: "white" }}>
              <th style={{ padding: "10px" }}>Customer ID</th>
              <th style={{ padding: "10px" }}>Segment</th>
              <th style={{ padding: "10px" }}>Churn Risk</th>
              <th style={{ padding: "10px" }}>Predicted LTV</th>
              <th style={{ padding: "10px" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {retainList.map((c, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #eee", textAlign: "center" }}>
                <td style={{ padding: "10px" }}>{c["Customer ID"]}</td>
                <td style={{ padding: "10px" }}>{c.Segment}</td>
                <td style={{ padding: "10px" }}>{(c.churn_probability * 100).toFixed(1)}%</td>
                <td style={{ padding: "10px" }}>£{c.predicted_ltv?.toFixed(2)}</td>
                <td style={{ padding: "10px" }}>{c.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;