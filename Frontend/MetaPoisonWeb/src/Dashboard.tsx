import React, { useEffect, useMemo, useState } from "react";
import { announceToScreenReader, getStatusAriaLabel } from "./a11y";
// import Playground from "./Playground.tsx";

type Row = {
  id?: string;
  dataset?: string;
  score: number;
  y_true?: number;
  flagged?: number;
};

type Metrics = {
  TP: number; FP: number; TN: number; FN: number;
  precision: number; recall: number; f1: number; tpr: number; fpr: number; acc: number;
};

type HealthResponse = {
  status: string;
  mode: string;
  tau_local: number;
  global_threshold: number;
  n_per_dataset_thresholds: number;
  trained_datasets: string[];
};

type ThresholdsResponse = {
  per_dataset: Record<string, number>;
  global: number;
};

const API_BASE = "http://localhost:5000";
const PRESET_TOPK = [50, 100, 150, 300, 500, 1000, 2000];

export function clamp01(n: number) { return Math.max(0, Math.min(1, n)); }

export function computeMetrics(rows: Row[], thr: number): Metrics | null {
  const labeled = rows.some((r) => r.y_true === 0 || r.y_true === 1);
  if (!labeled) return null;

  let TP = 0, FP = 0, TN = 0, FN = 0;
  for (const r of rows) {
    if (r.y_true !== 0 && r.y_true !== 1) continue;
    const pred = r.score >= thr ? 1 : 0;
    if (pred === 1 && r.y_true === 1) TP++;
    if (pred === 1 && r.y_true === 0) FP++;
    if (pred === 0 && r.y_true === 0) TN++;
    if (pred === 0 && r.y_true === 1) FN++;
  }

  const precision = TP / (TP + FP + 1e-9);
  const recall = TP / (TP + FN + 1e-9);
  const f1 = (2 * precision * recall) / (precision + recall + 1e-9);
  const tpr = recall;
  const fpr = FP / (FP + TN + 1e-9);
  const acc = (TP + TN) / (TP + TN + FP + FN + 1e-9);
  return { TP, FP, TN, FN, precision, recall, f1, tpr, fpr, acc };
}

export function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * clamp01(q);
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  return sorted[base];
}

function formatPct(x: number) { return `${(x * 100).toFixed(2)}%`; }
function formatNum(x: number, d = 3) { return Number.isFinite(x) ? x.toFixed(d) : "-"; }

async function analyzeCSV(file: File, tau: number | null, datasetHint: string) {
  const form = new FormData();
  form.append("file", file);
  if (tau !== null) form.append("tau", String(tau));
  if (datasetHint) form.append("dataset_hint", datasetHint);

  const res = await fetch(`${API_BASE}/analyze_csv`, { method: "POST", body: form });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload?.error || "Analyze CSV failed");
  return payload;
}

async function analyzeUCI(uci_id: number, tau: number | null) {
  const body: any = { uci_id };
  if (tau !== null) body.tau = tau;

  const res = await fetch(`${API_BASE}/analyze_uci`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload?.error || "Analyze UCI failed");
  return payload;
}

async function analyzeURL(url: string, tau: number | null, datasetHint: string) {
  const body: any = { url };
  if (tau !== null) body.tau = tau;
  if (datasetHint) body.dataset_hint = datasetHint;

  const res = await fetch(`${API_BASE}/analyze_url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload?.error || "Analyze URL failed");
  return payload;
}

const themes = {
  dark: {
    bg: "#1a1a1a",
    card: "#2a2a2a",
    text: "#ffffff",
    subtext: "#cfcfcf",
    accent: "#1f7ae0",
    border: "rgba(255,255,255,0.15)",
    inputBg: "#2a2a2a",
    inputText: "#ffffff",
    inputBorder: "rgba(255,255,255,0.15)"
  },
  light: {
    bg: "#f5f6f8",
    card: "#ffffff",
    text: "#111111",
    subtext: "#444444",
    accent: "#1f7ae0",
    border: "rgba(0,0,0,0.12)",
    inputBg: "#ffffff",
    inputText: "#111111",
    inputBorder: "rgba(0,0,0,0.2)"
  }
};


export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const t = themes[theme];
  const [sourceType, setSourceType] = useState<"csv" | "uci" | "url">("csv");
  const [csvFile, setCsvFile] = useState<File | null>(null);

  const [rows, setRows] = useState<Row[]>([]);
  const [status, setStatus] = useState<string>("Upload a CSV or provide a UCI id / URL.");
  const [err, setErr] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const [tau, setTau] = useState<number>(3.5);
  const [tauOverride] = useState<string>("");
  const [datasetHint] = useState<string>("");
  
  const [uciId, setUciId] = useState<string>("73");
  const [csvUrl, setCsvUrl] = useState<string>("");

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [thresholds, setThresholds] = useState<ThresholdsResponse | null>(null);
  const [tauLocal, setTauLocal] = useState<number>(0);
  const [inferredDataset, setInferredDataset] = useState<string>("");
  const [thresholdSource, setThresholdSource] = useState<string>("");
  const [usedTau, setUsedTau] = useState<number>(3.5);
  const [interactiveTau, setInteractiveTau] = useState<number>(3.5);
  const [topK, setTopK] = useState<number>(150);
  const [datasetId, setDatasetId] = useState<string | null>(null);

  const scores = useMemo(() => rows.map((r) => r.score), [rows]);
  const distribution = useMemo(() => {
    if (scores.length === 0) return null;
    return {
      min: Math.min(...scores),
      p50: quantile(scores, 0.5),
      p95: quantile(scores, 0.95),
      max: Math.max(...scores),
    };
  }, [scores]);

  const flagged = useMemo(() => rows.filter((r) => r.score >= interactiveTau), [rows, interactiveTau]);
  const metrics = useMemo(() => computeMetrics(rows, interactiveTau), [rows, interactiveTau]);
  const labeledMode = rows.some((r) => r.y_true === 0 || r.y_true === 1);

  useEffect(() => {
    (async () => {
      try {
        const resHealth = await fetch(`${API_BASE}/health`);
        const h = await resHealth.json();
        setHealth(h);
        setTau(h.global_threshold || 3.5);
        setUsedTau(h.global_threshold || 3.5);
        setInteractiveTau(h.global_threshold || 3.5);

        const resThresholds = await fetch(`${API_BASE}/thresholds`);
        const t = await resThresholds.json();
        setThresholds(t);
      } catch (e) {
        console.error("Failed to fetch health/thresholds:", e);
      }
    })();
  }, []);

  function loadFromBackend(res: any) {
    const newRows: Row[] = res.scores.map((s: number, i: number) => ({
      id: String(i),
      score: s,
      flagged: res.flags[i],
    }));
    setRows(newRows);
    setDatasetId(res.dataset_id || null);
    setInferredDataset(res.uci_id ? `uci_${res.uci_id}` : res.mode || "");
    setThresholdSource(res.threshold_source || "unknown");
    setUsedTau(res.tau);
    setInteractiveTau(res.tau);
    setTauLocal(res.tau_local ?? null);
    setTopK(prev => Math.min(prev, newRows.length));
  }

async function exportCleanDataset() {
  if (!datasetId) {
    alert("No dataset available for export");
    return;
  }

  // collect clean row indices
  const cleanIds = rows
    .filter(r => r.score < interactiveTau && r.flagged !== 1)
    .map(r => Number(r.id));

  if (cleanIds.length === 0) {
    alert("No clean rows to export");
    return;
  }

  const res = await fetch(`${API_BASE}/export_clean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: datasetId,
      clean_ids: cleanIds,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(err?.error || "Export failed");
    return;
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "clean_dataset.csv";
  a.click();

  URL.revokeObjectURL(url);
}

  const getTauForRequest = () => {
    if (tauOverride.trim()) {
      const val = parseFloat(tauOverride);
      if (!isNaN(val)) return val;
    }
    return null;
  };

  async function handleAnalyze() {
    try {
      setErr("");
      setLoading(true);
      setStatus("Analyzing dataset...");
      announceToScreenReader("Starting dataset analysis");

      let res;

      if (sourceType === "csv") {
        if (!csvFile) throw new Error("Please upload a CSV file.");
        res = await analyzeCSV(csvFile, getTauForRequest(), datasetHint);
      }

      if (sourceType === "uci") {
        if (!uciId.trim()) throw new Error("Please enter a UCI dataset ID.");
        res = await analyzeUCI(Number(uciId), getTauForRequest());
      }

      if (sourceType === "url") {
        if (!csvUrl.trim()) throw new Error("Please provide a CSV URL.");
        res = await analyzeURL(csvUrl.trim(), getTauForRequest(), datasetHint);
      }

      loadFromBackend(res);

      const statusMsg = `Analyzed ${res.n_rows} rows. τ=${formatNum(res.tau, 4)} (${res.threshold_source})`;
      setStatus(statusMsg);
      announceToScreenReader(`Analysis complete: ${statusMsg}`);

    } catch (ex: any) {
      const errorMsg = ex?.message ?? "Failed";
      setErr(errorMsg);
      setStatus("Failed.");
      announceToScreenReader(`Error: ${errorMsg}`, "assertive");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      id="main-content"
      role="main"
      style={{
        width: "96vw",
        minWidth: "90%",
        margin: 0,
        padding: "20px 24px",
        background: t.bg,
        color: t.text,
        minHeight: "100vh",
        overflowX: "hidden",
      }}
    >
      {/* Accessibility: ARIA live region for announcements */}
      <div
        id="aria-live-region"
        aria-live="polite"
        aria-atomic="true"
        style={{ position: "absolute", left: "-9999px" }}
      />

      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, background: t.card, padding: "16px", borderRadius: "8px", marginBottom: "20px" }}>
        <div>
          <h1 style={{ margin: 0, color: t.text, fontSize: "2em" }}>Poison Detection Dashboard</h1>
          <p style={{ margin: "6px 0 0", opacity: 0.8, color: t.subtext }}>
            Upload a dataset or provide a UCI dataset ID / URL to analyze for data poisoning.
          </p>
        </div>

        <button
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode (currently in ${theme} mode)`}
          onClick={() => {
            setTheme(theme === "dark" ? "light" : "dark");
            announceToScreenReader(`Switched to ${theme === "dark" ? "light" : "dark"} mode`);
          }}
          style={{
            background: t.card,
            color: t.text,
            border: `1px solid ${t.border}`,
            padding: "8px 16px",
            borderRadius: 6,
            cursor: "pointer",
            whiteSpace: "nowrap",
            minHeight: "44px",
          }}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? "☀ Light Mode" : "🌙 Dark Mode"}
        </button>
      </header>

      <div style={{ width: "100%" }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "1rem",
          }}>

          <div className="card">
            <section style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 14 }}>
              <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
                <h3 style={{ marginTop: 0 }}>Run Detection</h3>
                <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                  {["csv", "uci", "url"].map((type) => (
                    <button
                      key={type}
                      onClick={() => {
                        setSourceType(type as any);

                        // Clear other inputs when switching
                        setCsvFile(null);
                        setCsvUrl("");
                        setUciId("");
                      }}
                      style={{
                        padding: "6px 12px",
                        borderRadius: 6,
                        border: t.border,
                        background: sourceType === type ? "#1f7ae0" : t.bg,
                        color: t.text,
                        cursor: "pointer",
                      }}
                    >
                      {type.toUpperCase()}
                    </button>
                  ))}
                </div>

                <div style={{ display: "grid", gap: 10 }}>
                  {sourceType === "csv" && (
                  <label style={{ display: "grid", gap: 6, fontWeight: 600 }}>
                    <span>Upload CSV file (AUTO)</span>
                    <span style={{ fontSize: 12, opacity: 0.7, fontWeight: 400 }}>
                      Select a CSV file to automatically analyze for data poisoning detection
                    </span>
                    <input
                      style={{
                        background: t.inputBg,
                        color: t.inputText,
                        border: `1px solid ${t.inputBorder}`,
                        borderRadius: 6,
                        padding: "8px 10px",
                        fontSize: 14,
                      }}
                      type="file"
                      accept=".csv,text/csv"
                      aria-describedby="csv-upload-help"
                      onChange={async (e) => {
                        const f = e.target.files?.[0];
                        if (!f) return;
                        try {
                          setErr("");
                          setLoading(true);
                          setStatus("Uploading CSV → analyzing...");
                          announceToScreenReader("Uploading CSV file for analysis");
                          const res = await analyzeCSV(f, getTauForRequest(), datasetHint);
                          loadFromBackend(res);
                          const statusMsg = `Analyzed ${res.n_rows} rows. τ=${formatNum(res.tau, 4)} (${res.threshold_source})`;
                          setStatus(statusMsg);
                          announceToScreenReader(`Analysis complete: ${statusMsg}`);
                        } catch (ex: any) {
                          const errorMsg = ex?.message ?? "Failed";
                          setErr(errorMsg);
                          setStatus("Failed.");
                          announceToScreenReader(`Error: ${errorMsg}`, "assertive");
                        } finally {
                          setLoading(false);
                        }
                      }}
                    />
                    <span id="csv-upload-help" style={{ fontSize: 11, opacity: 0.6 }}>
                      Supported: CSV files. Maximum size: 100MB
                    </span>
                  </label>
                   )}

                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 10, alignItems: "end" }}>
                    {sourceType === "uci" && (
                    <label style={{ display: "grid", gap: 6, fontWeight: 600 }}>
                      <span>UCI Dataset ID</span>
                      <span style={{ fontSize: 12, opacity: 0.7, fontWeight: 400 }}>
                        Enter UCI Machine Learning Repository dataset ID (e.g., 73)
                      </span>
                      <input 
                        style={{
                          background: t.inputBg,
                          color: t.inputText,
                          border: `1px solid ${t.inputBorder}`,
                          borderRadius: 6,
                          padding: "8px 10px",
                          fontSize: 14,
                        }}
                        value={uciId} 
                        onChange={(e) => setUciId(e.target.value)}
                        placeholder="Enter dataset ID"
                        aria-describedby="uci-help"
                        type="number"
                        min="1"
                      />
                      <span id="uci-help" style={{ fontSize: 11, opacity: 0.6 }}>
                        Available at: openml.org (search for OpenML IDs)
                      </span>
                    </label>
                    )}
                    {/* <button
                      onClick={async () => {
                        try {
                          setErr("");
                          setLoading(true);
                          setStatus("Fetching UCI → analyzing...");
                          announceToScreenReader(`Loading UCI dataset ${uciId}`);
                          const res = await analyzeUCI(Number(uciId), getTauForRequest());
                          loadFromBackend(res);
                          const statusMsg = `Analyzed ${res.n_rows} rows. τ=${formatNum(res.tau, 4)} (${res.threshold_source})`;
                          setStatus(statusMsg);
                          announceToScreenReader(`Analysis complete: ${statusMsg}`);
                        } catch (ex: any) {
                          const errorMsg = ex?.message ?? "Failed";
                          setErr(errorMsg);
                          setStatus("Failed.");
                          announceToScreenReader(`Error loading UCI dataset: ${errorMsg}`, "assertive");
                        } finally {
                          setLoading(false);
                        }
                      }}
                      disabled={loading}
                      aria-label={`Analyze UCI dataset ${uciId} (currently ${loading ? "loading" : "ready"})`}
                      style={{padding: "8px 16px", cursor: loading ? "not-allowed" : "pointer", opacity: loading ? 0.6 : 1, minHeight: "44px"}}
                    >
                      {loading ? "Analyzing..." : "Analyze UCI ID"}
                    </button> */}
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 10, alignItems: "end" }}>
                    {sourceType === "url" && (
                    <label style={{ display: "grid", gap: 6, fontWeight: 600 }}>
                      <span>CSV From URL</span>
                      <span style={{ fontSize: 12, opacity: 0.7, fontWeight: 400 }}>
                        Provide a direct link to a CSV file
                      </span>
                      <input 
                        style={{
                          background: t.inputBg,
                          color: t.inputText,
                          border: `1px solid ${t.inputBorder}`,
                          borderRadius: 6,
                          padding: "8px 10px",
                          fontSize: 14,
                        }}
                        placeholder="https://example.com/data.csv" 
                        value={csvUrl} 
                        onChange={(e) => setCsvUrl(e.target.value)}
                        aria-describedby="url-help"
                        type="url"
                      />
                      <span id="url-help" style={{ fontSize: 11, opacity: 0.6 }}>
                        Must be a publicly accessible HTTP/HTTPS URL
                      </span>
                    </label>
                    )}
                    {/* <button
                      disabled={!csvUrl.trim() || loading}
                      onClick={async () => {
                        try {
                          setErr("");
                          setLoading(true);
                          setStatus("Fetching URL -> analyzing...");
                          announceToScreenReader("Loading CSV from URL");
                          const res = await analyzeURL(csvUrl.trim(), getTauForRequest(), datasetHint);
                          loadFromBackend(res);
                          const statusMsg = `Analyzed ${res.n_rows} rows. τ=${formatNum(res.tau, 4)} (${res.threshold_source})`;
                          setStatus(statusMsg);
                          announceToScreenReader(`Analysis complete: ${statusMsg}`);
                        } catch (ex: any) {
                          const errorMsg = ex?.message ?? "Failed";
                          setErr(errorMsg);
                          setStatus("Failed.");
                          announceToScreenReader(`Error loading from URL: ${errorMsg}`, "assertive");
                        } finally {
                          setLoading(false);
                        }
                      }}
                      aria-label={`Analyze CSV from URL${!csvUrl.trim() ? " (URL required)" : ""}`}
                      style={{padding: "8px 16px", cursor: (!csvUrl.trim() || loading) ? "not-allowed" : "pointer", opacity: (!csvUrl.trim() || loading) ? 0.6 : 1, minHeight: "44px"}}
                    >
                      {loading ? "Analyzing..." : "Analyze URL"}
                    </button> */}
                  </div>
                  <button
                    onClick={handleAnalyze}
                    disabled={loading}
                    style={{
                      marginTop: 16,
                      background: "#1f7ae0",
                      color: "white",
                      border: "none",
                      borderRadius: 6,
                      padding: "12px 20px",
                      fontSize: 16,
                      fontWeight: 600,
                      cursor: loading ? "not-allowed" : "pointer",
                      opacity: loading ? 0.7 : 1,
                      minHeight: "48px",
                      width: "100%",
                    }}
                  >
                    {loading ? "Analyzing..." : "Analyze Dataset"}
                  </button>
                  

                  <div 
                    style={{ fontSize: 16, opacity: 0.85 }}
                    role="status"
                    aria-live="polite"
                    aria-label={getStatusAriaLabel(status)}
                  >
                    {status}
                  </div>
                  {err ? (
                    <div 
                      style={{ color: "#ffb4b4" }}
                      role="alert"
                      aria-live="assertive"
                    >
                      Error: {err}
                    </div>
                  ) : null}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
                <div style={{ fontSize: 14, opacity: 0.7 }}>Rows</div>
                <div style={{ fontSize: 26, fontWeight: 700 }}>{rows.length.toLocaleString()}</div>
              </div>

              <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
                <div style={{ fontSize: 14, opacity: 0.7 }}>Flagged (score ≥ τ)</div>
                <div style={{ fontSize: 26, fontWeight: 700 }}>{flagged.length.toLocaleString()}</div>
                <div style={{ fontSize: 14, opacity: 0.7 }}>τ = {formatNum(interactiveTau, 2)}</div>
              </div>
              
              <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
                <div style={{ fontSize: 14, opacity: 0.7 }}>Score spread</div>
                {distribution ? (
                  <div style={{ fontSize: 14, lineHeight: 1.6 }}>
                    <div>min: {formatNum(distribution.min, 4)}</div>
                    <div>p50: {formatNum(distribution.p50, 4)}</div>
                    <div>p95: {formatNum(distribution.p95, 4)}</div>
                    <div>max: {formatNum(distribution.max, 4)}</div>
                  </div>
                ) : (
                  <div style={{ opacity: 0.7 }}>—</div>
                )}
              </div>
            
            <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
              <h3 style={{ marginTop: 0, color: t.text }}>Detection Quality</h3>
              {!metrics ? (
                <div style={{ opacity: 0.8, color: t.subtext }}>No labels - only flags and score ranking.</div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
                  <Metric label="Precision" value={metrics.precision} />
                  <Metric label="Recall (TPR)" value={metrics.recall} />
                  <Metric label="F1" value={metrics.f1} />
                  <Metric label="FPR" value={metrics.fpr} />
                  <Metric label="Accuracy" value={metrics.acc} />
                </div>
              )}
            </div>
            </div>
            </section>

            <section style={{marginTop: 14, background: t.card, padding: 14, borderRadius: 8}}>
              <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
                <h2 style={{ marginTop: 0, color: t.text, fontSize: "1.4em" }}>Score Distribution</h2>
                <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 8 }}>
                  Histogram showing the distribution of anomaly scores across all rows.
                  Blue bars represent scores below the threshold (clean), red bars represent flagged scores.
                </div>
                <ScoreHistogram rows={rows} tau={interactiveTau} height={320}  theme={t} />
              </div>
            </section>
          </div>

          


          {/*Right side*/}
          <div className="card">
            <div style={{ background: t.card, padding: 14, borderRadius: 8, marginTop: 12 }}>
                <h3 style={{ marginTop: 0 }}>Thresholds</h3>

                <div
                  style={{display: "flex", gap: 16, flexWrap: "wrap", fontSize: 16, lineHeight: 1.6}}>
                  <div><strong>Local τ:</strong> {formatNum(tauLocal, 4)}</div>
                  <div><strong>Global τ:</strong> {formatNum(health?.global_threshold ?? usedTau, 4)}</div>
                  <div><strong>Final τ:</strong> {formatNum(usedTau, 4)}</div>
                  <div><strong>Source:</strong> {thresholdSource}</div>
                </div>
              </div>
            {rows.length > 0 && (
              <section style={{marginTop: 14, background: t.card, padding: 14, borderRadius: 8}}>
                <h2 style={{ marginTop: 0, color: t.text, fontSize: "1.4em" }}>Interactive Threshold Tuning</h2>
                <p style={{ fontSize: 13, opacity: 0.75, margin: "0 0 16px 0" }}>
                  Adjust the detection threshold to see real-time impact on flagged rows and metrics. 
                  A higher threshold results in fewer flagged rows but higher precision.
                </p>
                <div style={{display: "grid", gridTemplateColumns: "1fr auto", gap: 20, alignItems: "center"}}>
                  <div>
                    <div style={{display: "flex", alignItems: "center", gap: 12, marginBottom: 8, flexWrap: "wrap"}}>
                      <label 
                        htmlFor="threshold-slider"
                        style={{fontSize: 16, opacity: 0.85, minWidth: 120, fontWeight: 600}}
                      >
                        Threshold: <strong>{formatNum(interactiveTau, 4)}</strong>
                      </label>
                      <input 
                        id="threshold-slider"
                        type="range" 
                        min={Math.max(0, distribution?.min || 0)} 
                        max={Math.min(20, distribution?.max || 10)}
                        step={0.1}
                        value={interactiveTau}
                        onChange={(e) => {
                          const newValue = parseFloat(e.target.value);
                          setInteractiveTau(newValue);
                          announceToScreenReader(`Threshold adjusted to ${formatNum(newValue, 4)}`);
                        }}
                        aria-valuemin={0}
                        aria-valuemax={20}
                        aria-valuenow={interactiveTau}
                        aria-valuetext={`${formatNum(interactiveTau, 4)}`}
                        style={{flex: 1, minWidth: 150}}
                        title="Adjust detection threshold"
                      />
                      <button 
                        onClick={() => {
                          setInteractiveTau(usedTau);
                          announceToScreenReader(`Threshold reset to ${formatNum(usedTau, 2)}`);
                        }}
                        style={{
                          background: t.card,
                          color: t.text,
                          border: `1px solid ${t.border}`,
                          borderRadius: 6,
                          padding: "6px 12px",
                          fontSize: 14,
                          cursor: "pointer",
                          minHeight: "40px",
                          whiteSpace: "nowrap",
                          transition: "all 0.2s ease"
                        }}
                        title="Reset to original threshold"
                      >
                        Reset to {formatNum(usedTau, 2)}
                      </button>
                    </div>
                    <div style={{fontSize: 14, opacity: 0.75, lineHeight: 1.5}}>
                      Slide to adjust threshold and see real-time impact on flagged rows and metrics.
                      Original threshold from backend: <strong>{formatNum(usedTau, 4)}</strong> ({thresholdSource})
                    </div>
                  </div>
                  <div style={{textAlign: "center", padding: "0 20px"}}>
                    <div 
                      style={{fontSize: 32, fontWeight: 700, color: flagged.length > rows.length * 0.5 ? "#ff6b6b" : "#4ecdc4"}}
                      aria-label={`${(flagged.length / rows.length * 100).toFixed(2)} percent of rows flagged`}
                    >
                      {formatPct(flagged.length / rows.length)}
                    </div>
                    <div style={{fontSize: 14, opacity: 0.75, color: t.subtext}}>Flagged</div>
                  </div>
                </div>
              </section>
            )}
            <section style={{marginTop: 14, background: t.card, padding: 14, borderRadius: 8}}>
              <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
                <h2 style={{ marginTop: 0, color: t.text, fontSize: "1.4em" }}>Flagged Rows Visualization</h2>
                <div style={{ fontSize: 14, opacity: 0.8, marginBottom: 8 }}>
                  Each point represents a row. The horizontal dashed line shows the current threshold. 
                  Larger, brighter points indicate flagged (suspicious) rows above the threshold.
                </div>
                <ScoreScatter rows={rows} tau={interactiveTau} height={480} theme={t} />
              </div>
            </section>
            
          </div>
        </div>
      </div>
      <section style={{marginTop: 18, display: "grid", gridTemplateColumns: "1fr", gap: 14}}>
        <div style={{background: t.card, padding: 14, borderRadius: 8, textAlign: "left"}}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 12,
              flexWrap: "wrap",
            }}>
            <h2 style={{ marginTop: 0, fontSize: "1.4em" }}>Most Suspicious Rows</h2>

            <button
              onClick={() => exportCleanDataset()}
              disabled={!datasetId || rows.length === 0}
              aria-label="Export clean dataset (rows below threshold)"
              style={{
                background: "#1f7ae0",
                color: "white",
                border: "none",
                borderRadius: 6,
                padding: "8px 16px",
                fontSize: 14,
                cursor: "pointer",
                whiteSpace: "nowrap",
                minHeight: "40px",
              }}>
              Export Clean Dataset
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
            <div style={{ fontSize: 14, opacity: 0.8, fontWeight: 600 }}>
              Show top rows by score:
            </div>

            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {PRESET_TOPK
                .filter(v => v <= rows.length)
                .map(v => (
                  <button
                    key={v}
                    onClick={() => {
                      setTopK(v);
                      announceToScreenReader(`Showing top ${v} rows`);
                    }}
                    aria-pressed={topK === v}
                    aria-label={`Show top ${v} suspicious rows${topK === v ? " (currently selected)" : ""}`}
                    style={{
                      fontSize: 13,
                      padding: "6px 10px",
                      borderRadius: 999,
                      background: topK === v ? "#1f7ae0" : t.bg ,
                      color: t.text,
                      border: t.border,
                      
                      cursor: "pointer",
                      opacity: topK === v ? 1 : 0.85,
                      minHeight: "36px",
                      minWidth: "36px",
                    }}
                  >
                    {v}
                  </button>
                ))}
            </div>
          </div>

          {rows.length === 0 ? (
            <div style={{ opacity: 0.7 }}>No data loaded. Upload a dataset to get started.</div>
          ) : (
            <div className={theme === "dark" ? "dark-scroll" : "light-scroll"}
              style={{ overflow: "auto", maxHeight: 360 }}>
              <table 
                style={{ width: "100%", borderCollapse: "collapse", fontSize: 16 }}
                role="table"
                aria-label="Most suspicious rows by anomaly score"
              >
                <thead>
                  <tr style={{ textAlign: "left", borderBottom: "2px solid rgba(255,255,255,0.12)" }}>
                    <th scope="col" style={{ padding: "8px 6px", fontWeight: 700 }}>Row ID</th>
                    <th scope="col" style={{ padding: "8px 6px", fontWeight: 700 }}>Anomaly Score</th>
                    <th scope="col" style={{ padding: "8px 6px", fontWeight: 700 }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[...rows]
                    .sort((a, b) => b.score - a.score)
                    .slice(0, topK)
                    .map((r, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                        <td style={{ padding: "8px 6px" }}>{r.id ?? `row_${i}`}</td>
                        <td style={{ padding: "8px 6px" }}>{formatNum(r.score, 4)}</td>
                        <td style={{ padding: "8px 6px" }}>
                          <span
                            style={{
                              padding: "4px 8px",
                              borderRadius: 999,
                              border: "1px solid rgba(255,255,255,0.15)",
                              opacity: r.score >= interactiveTau ? 1 : 0.6,
                              backgroundColor: r.score >= interactiveTau ? "rgba(255,107,107,0.2)" : "rgba(78,205,196,0.2)",
                            }}
                            aria-label={r.score >= interactiveTau ? "Flagged as suspicious" : "Clean, below threshold"}
                          >
                            {r.score >= interactiveTau ? "🚩 FLAGGED" : "✓ Clean"}
                          </span>
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ padding: 12, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10}}>
      <div style={{ fontSize: 14, opacity: 0.75 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700 }}>{formatPct(value)}</div>
    </div>
  );
}

function ScoreScatter({ rows, tau, height = 260, theme }: { rows: Row[]; tau: number; height?: number, theme: any }) {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const MAX_POINTS = 8000;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    const widthCss = parent ? Math.max(320, parent.clientWidth) : 700;

    canvas.style.width = `${widthCss}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.floor(widthCss * dpr);
    canvas.height = Math.floor(height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, widthCss, height);

    if (!rows || rows.length === 0) {
      ctx.globalAlpha = 0.75;
      ctx.font = "13px system-ui";
      ctx.fillStyle = theme.text;
      ctx.fillText("No data yet.", 10, 20);
      return;
    }

    const N = rows.length;
    const step = Math.max(1, Math.floor(N / MAX_POINTS));

    const sampled: { x: number; y: number; flagged: boolean }[] = [];
    for (let i = 0; i < N; i += step) {
      const r = rows[i];
      sampled.push({ x: i, y: r.score, flagged: r.score >= tau });
    }

    const ys = sampled.map((p) => p.y).sort((a, b) => a - b);
    // const q = (arr: number[], qq: number) => {
    //   const pos = (arr.length - 1) * Math.max(0, Math.min(1, qq));
    //   const base = Math.floor(pos);
    //   const rest = pos - base;
    //   return arr[base + 1] !== undefined ? arr[base] + rest * (arr[base + 1] - arr[base]) : arr[base];
    // };

    const yLo = Math.min(...ys);
    const yHi = Math.max(...ys);
    const pad = 0.08 * (yHi - yLo + 1e-9);
    const yMin = yLo - pad;
    const yMax = yHi + pad;

    const margin = { l: 34, r: 10, t: 10, b: 22 };
    const W = widthCss - margin.l - margin.r;
    const H = height - margin.t - margin.b;

    const xToPx = (x: number) => margin.l + (x / Math.max(1, N - 1)) * W;
    const yToPx = (y: number) => margin.t + (1 - (y - yMin) / (yMax - yMin + 1e-9)) * H;

    ctx.globalAlpha = 0.25;
    ctx.strokeStyle = theme.text;
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(margin.l, margin.t + H);
    ctx.lineTo(margin.l + W, margin.t + H);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(margin.l, margin.t);
    ctx.lineTo(margin.l, margin.t + H);
    ctx.stroke();

    const yTau = yToPx(tau);
    if (yTau >= margin.t && yTau <= margin.t + H) {
      ctx.globalAlpha = 0.45;
      ctx.strokeStyle = theme.text;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(margin.l, yTau);
      ctx.lineTo(margin.l + W, yTau);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.globalAlpha = 0.75;
      ctx.font = "12px system-ui";
      ctx.fillStyle = theme.text;
      ctx.fillText(`τ=${tau.toFixed(2)}`, margin.l + 6, Math.max(margin.t + 12, yTau - 6));
    }

    for (const p of sampled) {
      const x = xToPx(p.x);
      const y = yToPx(p.y);
      if (y < margin.t - 10 || y > margin.t + H + 10) continue;

      ctx.beginPath();
      ctx.globalAlpha = p.flagged ? 0.9 : 0.18;
      ctx.fillStyle = theme.text;
      const r = p.flagged ? 2.2 : 1.3;
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 0.7;
    ctx.font = "12px system-ui";
    ctx.fillStyle = theme.text;
    ctx.fillText("row index", margin.l + W / 2 - 22, margin.t + H + 18);
    ctx.save();
    ctx.translate(12, margin.t + H / 2 + 18);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("score", 0, 0);
    ctx.restore();
  }, [rows, tau, height]);

  return (
    <div style={{ width: "100%" }} role="figure" aria-label="Scatter plot of anomaly scores">
      <canvas ref={canvasRef} role="img" aria-label="Scatter plot showing anomaly scores by row index. Larger brighter points are flagged as suspicious." />
      <div style={{ fontSize: 14, opacity: 0.75, marginTop: 6, padding: "8px 12px", backgroundColor: theme === themes.dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", borderRadius: "4px" }}>
        <strong>How to read this chart:</strong> X-axis shows row index, Y-axis shows anomaly score. 
        The horizontal dashed line represents the current threshold (τ). Brighter, larger points above the line are flagged as suspicious.
      </div>
    </div>
  );
}

function ScoreHistogram({ rows, tau, height = 320, theme }: { rows: Row[]; tau: number; height?: number, theme: any  }) {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    const widthCss = parent ? Math.max(320, parent.clientWidth) : 700;

    canvas.style.width = `${widthCss}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.floor(widthCss * dpr);
    canvas.height = Math.floor(height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, widthCss, height);

    if (!rows || rows.length === 0) {
      ctx.globalAlpha = 0.75;
      ctx.font = "13px system-ui";
      ctx.fillStyle = theme.text;
      ctx.fillText("No data yet.", 10, 20);
      return;
    }

    const scores = rows.map(r => r.score);
    const minS = Math.min(...scores);
    const maxS = Math.max(...scores);
    
    const nBins = 50;
    const binWidth = (maxS - minS) / nBins;
    const bins = new Array(nBins).fill(0);
    
    for (const s of scores) {
      const idx = Math.min(nBins - 1, Math.floor((s - minS) / (binWidth + 1e-9)));
      bins[idx]++;
    }

    const maxCount = Math.max(...bins);
    
    const margin = { l: 40, r: 10, t: 10, b: 30 };
    const W = widthCss - margin.l - margin.r;
    const H = height - margin.t - margin.b;

    const barWidth = W / nBins;

    for (let i = 0; i < nBins; i++) {
      const binStart = minS + i * binWidth;
      const binEnd = binStart + binWidth;
      const isBelowThreshold = binEnd < tau;
      
      const x = margin.l + i * barWidth;
      const barHeight = (bins[i] / maxCount) * H;
      const y = margin.t + H - barHeight;

      ctx.globalAlpha = isBelowThreshold ? 0.3 : 0.8;
      ctx.fillStyle = isBelowThreshold ? "#4a9eff" : "#ff6b6b";
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    }

    const tauX = margin.l + ((tau - minS) / (maxS - minS)) * W;
    if (tauX >= margin.l && tauX <= margin.l + W) {
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = "#ffd700";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(tauX, margin.t);
      ctx.lineTo(tauX, margin.t + H);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.globalAlpha = 1;
      ctx.font = "12px system-ui";
      ctx.fillStyle = "#ffd700";
      ctx.fillText(`τ=${tau.toFixed(2)}`, tauX + 4, margin.t + 14);
    }

    ctx.globalAlpha = 0.3;
    ctx.strokeStyle = theme.text;
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(margin.l, margin.t + H);
    ctx.lineTo(margin.l + W, margin.t + H);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(margin.l, margin.t);
    ctx.lineTo(margin.l, margin.t + H);
    ctx.stroke();

    ctx.globalAlpha = 0.7;
    ctx.font = "11px system-ui";
    ctx.fillStyle = theme.text;
    ctx.fillText(minS.toFixed(2), margin.l, margin.t + H + 18);
    ctx.fillText(maxS.toFixed(2), margin.l + W - 30, margin.t + H + 18);
    ctx.fillText("score", margin.l + W / 2 - 15, margin.t + H + 18);
    
    ctx.save();
    ctx.translate(12, margin.t + H / 2 + 18);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("count", 0, 0);
    ctx.restore();
  }, [rows, tau, height]);

  return (
    <div style={{ width: "100%" }} role="figure" aria-label="Histogram of anomaly score distribution">
      <canvas ref={canvasRef} role="img" aria-label="Histogram showing distribution of anomaly scores. Blue bars are below threshold (clean), red bars are above threshold (flagged)." />
      <div style={{ fontSize: 14, opacity: 0.75, marginTop: 6, padding: "8px 12px", backgroundColor: theme === themes.dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", borderRadius: "4px" }}>
        <strong>How to read this chart:</strong> Blue bars represent scores below the threshold (clean rows), 
        red bars represent scores at or above the threshold (flagged rows). The dashed golden line shows the current threshold.
      </div>
    </div>
  );
}