import { useEffect, useMemo, useState } from "react";
import {
  analyzeImage,
  getHistory,
  getHistorySummary,
  getModelInfo,
  healthCheck,
  submitFeedback
} from "./api";

const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB || 8);
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const DECISION_PRESETS = {
  screening: 0.35,
  balanced: 0.45
};
const BRAND_ASSETS = {
  light: "/branding/logo_light.png"
};

function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [historyRecords, setHistoryRecords] = useState([]);
  const [historySummary, setHistorySummary] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [gradcam, setGradcam] = useState(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [decisionMode, setDecisionMode] = useState("custom");

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const onFileChange = (event) => {
    const nextFile = event.target.files?.[0] || null;
    setPrediction(null);
    setGradcam(null);
    setError("");

    if (!nextFile) {
      setFile(null);
      return;
    }
    if (!nextFile.type?.startsWith("image/")) {
      setFile(null);
      setError("Please upload an image file.");
      return;
    }
    if (nextFile.size > MAX_UPLOAD_BYTES) {
      setFile(null);
      setError(`Please upload an image smaller than ${MAX_UPLOAD_MB} MB.`);
      return;
    }
    setFile(nextFile);
  };

  const refreshStatus = async () => {
    setError("");
    try {
      const [healthPayload, modelPayload] = await Promise.all([healthCheck(), getModelInfo()]);
      setHealth(healthPayload);
      setModelInfo(modelPayload);
      if (modelPayload?.default_threshold !== undefined) {
        setThreshold(modelPayload.default_threshold);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const refreshHistory = async () => {
    setError("");
    try {
      const [records, summary] = await Promise.all([getHistory(100), getHistorySummary()]);
      setHistoryRecords(records);
      setHistorySummary(summary);
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    refreshStatus();
    refreshHistory();
  }, []);

  const onAnalyze = async () => {
    if (!file) {
      setError("Choose a chest X-ray image first.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const result = await analyzeImage(file, threshold.toFixed(2));
      setPrediction({
        predicted_label: result.predicted_label,
        confidence: result.confidence,
        probabilities: result.probabilities,
        threshold: result.threshold,
        inference_mode: result.inference_mode,
        model_arch: result.model_arch,
        checkpoint_loaded: result.checkpoint_loaded,
        analysis_id: result.analysis_id,
      });
      setGradcam({
        predicted_label: result.predicted_label,
        confidence: result.confidence,
        heatmap_base64: result.heatmap_base64,
        inference_mode: result.inference_mode,
        model_arch: result.model_arch,
        checkpoint_loaded: result.checkpoint_loaded,
        gradcam_mode: result.gradcam_mode,
        lung_focus_score: result.lung_focus_score,
        off_lung_attention_ratio: result.off_lung_attention_ratio,
        explainability_warning: result.explainability_warning,
      });
      await refreshHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const onSelectDecisionMode = (mode) => {
    setError("");
    if (mode === "custom") {
      setDecisionMode("custom");
      return;
    }
    const preset = DECISION_PRESETS[mode];
    if (preset === undefined) {
      return;
    }
    setDecisionMode(mode);
    setThreshold(preset);
  };

  const onClear = () => {
    setFile(null);
    setPrediction(null);
    setGradcam(null);
    setError("");
  };

  const onFeedback = async (recordId, feedback) => {
    try {
      await submitFeedback(recordId, feedback);
      await refreshHistory();
    } catch (err) {
      setError(err.message);
    }
  };

  const onDownloadReport = () => {
    if (!prediction) {
      setError("Run an analysis before downloading a report.");
      return;
    }
    const report = {
      timestamp_utc: new Date().toISOString(),
      file_name: file?.name || null,
      threshold: prediction.threshold,
      predicted_label: prediction.predicted_label,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      inference_mode: prediction.inference_mode,
      model_arch: prediction.model_arch,
      checkpoint_loaded: prediction.checkpoint_loaded,
      model_info: modelInfo || null,
      explainability: gradcam
        ? {
            lung_focus_score: gradcam.lung_focus_score,
            off_lung_attention_ratio: gradcam.off_lung_attention_ratio,
            explainability_warning: gradcam.explainability_warning || null
          }
        : null
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `medscanassist-report-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const onDownloadCsvReport = () => {
    if (!prediction) {
      setError("Run an analysis before downloading a report.");
      return;
    }

    const probabilities = prediction.probabilities || {};
    const probabilityColumns = Object.keys(probabilities).sort();
    const headers = [
      "timestamp_utc",
      "file_name",
      "threshold",
      "predicted_label",
      "confidence",
      "inference_mode",
      "model_arch",
      "checkpoint_loaded",
      "best_epoch",
      "best_val_acc",
      "best_val_loss",
      ...probabilityColumns.map((label) => `prob_${label}`),
      "lung_focus_score",
      "off_lung_attention_ratio",
      "explainability_warning"
    ];

    const values = [
      new Date().toISOString(),
      file?.name || "",
      String(prediction.threshold ?? ""),
      String(prediction.predicted_label ?? ""),
      String(prediction.confidence ?? ""),
      String(prediction.inference_mode ?? ""),
      String(prediction.model_arch ?? ""),
      String(prediction.checkpoint_loaded ?? ""),
      String(modelInfo?.best_epoch ?? ""),
      String(modelInfo?.best_val_acc ?? ""),
      String(modelInfo?.best_val_loss ?? ""),
      ...probabilityColumns.map((label) => String(probabilities[label] ?? "")),
      String(gradcam?.lung_focus_score ?? ""),
      String(gradcam?.off_lung_attention_ratio ?? ""),
      String(gradcam?.explainability_warning ?? "")
    ];

    const toCsvCell = (value) => `"${String(value).replaceAll("\"", "\"\"")}"`;
    const csv = `${headers.map(toCsvCell).join(",")}\n${values.map(toCsvCell).join(",")}\n`;

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `medscanassist-report-${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const statusOk = health?.status === "ok";
  const hasPreview = Boolean(previewUrl);
  const hasPrediction = Boolean(prediction);
  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];
  const sliderProgress = ((threshold - 0.05) / (0.95 - 0.05)) * 100;
  const overlayPercent = Math.round(overlayOpacity * 100);

  return (
    <main className="page">
      <header className="hero">
        <div className="brand-strip" aria-label="MedScanAssist branding">
          <img
            className="brand-wordmark"
            src={BRAND_ASSETS.light}
            alt="MedScanAssist logo"
            loading="eager"
            onError={(event) => {
              event.currentTarget.style.display = "none";
            }}
          />
        </div>
        <h1 className="sr-only">MedScanAssist</h1>
        <p>
          Upload a chest X-ray, run inference, and inspect Grad-CAM in a transparent workflow.
        </p>
      </header>

      <section className="tabs" aria-label="Main views">
        <button
          type="button"
          className={`tab-button ${activeTab === "analyze" ? "active" : ""}`}
          onClick={() => setActiveTab("analyze")}
        >
          Analyze
        </button>
        <button
          type="button"
          className={`tab-button ${activeTab === "history" ? "active" : ""}`}
          onClick={() => setActiveTab("history")}
        >
          Review History
        </button>
      </section>

      <section className="status-bar" aria-label="System status">
        <span className={`pill ${statusOk ? "ok" : "warn"}`}>
          API: {statusOk ? "Healthy" : "Unavailable"}
        </span>
        <span className={`pill ${modelInfo?.checkpoint_loaded ? "ok" : "warn"}`}>
          Mode: {modelInfo?.inference_mode || "unknown"}
        </span>
        <span className="pill neutral">Arch: {modelInfo?.model_arch || "n/a"}</span>
        <button type="button" className="ghost" onClick={refreshStatus}>
          Refresh status
        </button>
      </section>

      <details className="card disclosure">
        <summary>Model transparency details</summary>
        <section className="disclosure-body">
          <h3>Model Transparency</h3>
          <div className="meta-grid">
            <p>
              <strong>Inference mode:</strong> {modelInfo?.inference_mode || "unknown"}
            </p>
            <p>
              <strong>Architecture:</strong> {modelInfo?.model_arch || "n/a"}
            </p>
            <p>
              <strong>Checkpoint loaded:</strong> {String(modelInfo?.checkpoint_loaded ?? false)}
            </p>
            <p>
              <strong>Best epoch:</strong> {modelInfo?.best_epoch ?? "n/a"}
            </p>
            <p>
              <strong>Best val acc:</strong>{" "}
              {modelInfo?.best_val_acc !== null && modelInfo?.best_val_acc !== undefined
                ? modelInfo.best_val_acc.toFixed(4)
                : "n/a"}
            </p>
            <p>
              <strong>Default threshold:</strong>{" "}
              {modelInfo?.default_threshold !== undefined ? modelInfo.default_threshold.toFixed(2) : "n/a"}
            </p>
          </div>
          <p className="muted">
            Threshold controls classification tradeoff. Higher values reduce false positives but may increase false
            negatives.
          </p>
        </section>
      </details>

      {activeTab === "analyze" ? (
        <>
          <section className="panel">
            <div className="panel-heading">
              <h2>Run Analysis</h2>
              <p>Only image upload and threshold are required.</p>
            </div>

            <div className="controls">
              <label className="field">
                <span>Chest X-ray image</span>
                <div className={`file-upload ${file ? "selected" : ""}`}>
                  <input
                    id="xray-upload"
                    className="file-input-hidden"
                    type="file"
                    accept="image/*"
                    onChange={onFileChange}
                  />
                  <div className="file-upload-copy">
                    <strong>{file ? "X-ray ready" : "Upload chest X-ray"}</strong>
                    <span>{file?.name || "PNG/JPG/JPEG supported"}</span>
                  </div>
                  <label htmlFor="xray-upload" className="file-upload-button">
                    {file ? "Change file" : "Choose file"}
                  </label>
                </div>
              </label>

              <div className="field">
                <span>Decision mode</span>
                <div className="mode-toggle" role="group" aria-label="Decision threshold mode">
                  <button
                    type="button"
                    className={`mode-pill ${decisionMode === "screening" ? "active" : ""}`}
                    onClick={() => onSelectDecisionMode("screening")}
                  >
                    Screening
                  </button>
                  <button
                    type="button"
                    className={`mode-pill ${decisionMode === "balanced" ? "active" : ""}`}
                    onClick={() => onSelectDecisionMode("balanced")}
                  >
                    Balanced
                  </button>
                  <button
                    type="button"
                    className={`mode-pill ${decisionMode === "custom" ? "active" : ""}`}
                    onClick={() => onSelectDecisionMode("custom")}
                  >
                    Custom
                  </button>
                </div>
                <small>
                  Screening ({DECISION_PRESETS.screening.toFixed(2)}) maximizes F1/sensitivity.
                  Balanced ({DECISION_PRESETS.balanced.toFixed(2)}) optimizes Youden-J index.
                  Both derived from threshold analysis on test data.
                </small>
              </div>

              <label className="field">
                <span>Decision threshold: {threshold.toFixed(2)}</span>
                <input
                  className="threshold-slider"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.01"
                  value={threshold}
                  style={{ "--slider-progress": `${sliderProgress}%` }}
                  onChange={(event) => {
                    setDecisionMode("custom");
                    setThreshold(Number(event.target.value));
                  }}
                />
                <small>Higher threshold reduces false positives but can increase false negatives.</small>
              </label>

              <div className="row">
                <button onClick={onAnalyze} type="button" disabled={loading}>
                  {loading ? "Analyzing..." : "Analyze X-ray"}
                </button>
                <button onClick={onClear} type="button" className="ghost">
                  Clear
                </button>
              </div>
            </div>
          </section>

          {error && <p className="error">{error}</p>}

          {!hasPreview && !hasPrediction && (
            <p className="muted inline-note">Results and Grad-CAM appear after you run analysis.</p>
          )}

          {(hasPreview || hasPrediction) && (
            <section className="grid">
              {hasPreview && (
                <article className="card">
                  <h3>Input Preview</h3>
                  <img className="preview" src={previewUrl} alt="Uploaded chest x-ray preview" />
                </article>
              )}

              {hasPrediction && (
                <article className="card">
                  <h3>Prediction</h3>
                  <>
                    <p className="prediction-line">
                      <strong>{prediction.predicted_label.toUpperCase()}</strong>{" "}
                      ({(prediction.confidence * 100).toFixed(1)}% confidence)
                    </p>
                    <p className="muted">
                      Threshold used: {prediction.threshold.toFixed(2)} | Mode: {prediction.inference_mode}
                    </p>
                    <div className="prob-list">
                      {sortedProbabilities.map(([label, value]) => (
                        <div key={label} className="prob-item">
                          <div className="prob-meta">
                            <span>{label}</span>
                            <span>{(value * 100).toFixed(1)}%</span>
                          </div>
                          <div className="meter">
                            <div className="meter-fill" style={{ width: `${Math.max(2, value * 100)}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                </article>
              )}
            </section>
          )}

          {hasPreview && (
            <details className="card disclosure">
              <summary>Grad-CAM view</summary>
              <section className="disclosure-body">
                <h3>Grad-CAM</h3>
                {gradcam?.gradcam_mode === "synthetic" && (
                  <p className="attention-warning">
                    Synthetic heatmap: no trained checkpoint is loaded. This overlay is a
                    center-weighted placeholder and does not reflect real model activations.
                  </p>
                )}
                {gradcam?.explainability_warning && gradcam?.gradcam_mode === "real" && (
                  <p className="attention-warning">{gradcam.explainability_warning}</p>
                )}
                {gradcam && (
                  <p className="muted">
                    Mode: {gradcam.gradcam_mode === "real" ? "Real Grad-CAM" : "Synthetic placeholder"}
                    {gradcam.gradcam_mode === "real" && (
                      <> | Lung focus: {(gradcam.lung_focus_score * 100).toFixed(1)}% | Off-lung:{" "}
                      {(gradcam.off_lung_attention_ratio * 100).toFixed(1)}%</>
                    )}
                  </p>
                )}
                <div className="cam-wrapper">
                  <div className="cam-controls">
                    <label className="field">
                      <span>Overlay opacity: {overlayPercent}%</span>
                      <input
                        className="threshold-slider"
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={overlayOpacity}
                        style={{ "--slider-progress": `${overlayPercent}%` }}
                        onChange={(event) => setOverlayOpacity(Number(event.target.value))}
                      />
                    </label>
                  </div>
                  <div className="cam-grid">
                    <div>
                      <p className="cam-label">Original</p>
                      <img className="preview" src={previewUrl} alt="Original chest x-ray" />
                    </div>
                    <div>
                      <p className="cam-label">Overlay</p>
                      <div className="overlay-frame">
                        <img className="preview overlay-base" src={previewUrl} alt="Original chest x-ray background" />
                        {gradcam?.heatmap_base64 && (
                          <img
                            className="preview overlay-top"
                            style={{ opacity: overlayOpacity }}
                            src={`data:image/png;base64,${gradcam.heatmap_base64}`}
                            alt="Grad-CAM overlay"
                          />
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            </details>
          )}

          {hasPrediction && (
            <details className="card disclosure">
              <summary>Export report</summary>
              <section className="disclosure-body">
                <h3>Export</h3>
                <p className="muted">
                  Export JSON for complete machine-readable audit records, or CSV for spreadsheet-style review.
                </p>
                <div className="row">
                  <button type="button" onClick={onDownloadReport} disabled={!prediction}>
                    Download report (.json)
                  </button>
                  <button type="button" className="ghost" onClick={onDownloadCsvReport} disabled={!prediction}>
                    Download report (.csv)
                  </button>
                </div>
              </section>
            </details>
          )}
        </>
      ) : (
        <>
          {error && <p className="error">{error}</p>}

          <section className="grid metrics-grid">
            <article className="card metric-card">
              <h3>Total Analyzed</h3>
              <p className="metric-value">{historySummary?.total_reviews ?? 0}</p>
            </article>
            <article className="card metric-card">
              <h3>Pneumonia</h3>
              <p className="metric-value">{historySummary?.pneumonia_count ?? 0}</p>
            </article>
            <article className="card metric-card">
              <h3>Normal</h3>
              <p className="metric-value">{historySummary?.normal_count ?? 0}</p>
            </article>
            <article className="card metric-card">
              <h3>Avg Confidence</h3>
              <p className="metric-value">
                {historySummary ? `${(historySummary.avg_confidence * 100).toFixed(1)}%` : "0.0%"}
              </p>
            </article>
            <article className="card metric-card">
              <h3>Confirmed Correct</h3>
              <p className="metric-value metric-correct">{historySummary?.feedback_correct ?? 0}</p>
            </article>
            <article className="card metric-card">
              <h3>Flagged Incorrect</h3>
              <p className="metric-value metric-incorrect">{historySummary?.feedback_incorrect ?? 0}</p>
            </article>
            <article className="card metric-card span-2">
              <h3>Clinician-Reviewed Accuracy</h3>
              <p className="metric-value">
                {historySummary?.feedback_accuracy != null
                  ? `${(historySummary.feedback_accuracy * 100).toFixed(1)}%`
                  : "No reviews yet"}
              </p>
              <p className="muted metric-sub">Based on clinician feedback only</p>
            </article>
          </section>

          <section className="card">
            <div className="history-header">
              <h3>Recent Reviews</h3>
              <button type="button" className="ghost" onClick={refreshHistory}>
                Refresh history
              </button>
            </div>
            {historyRecords.length === 0 ? (
              <p className="muted">No analyses saved yet. Run an analysis to populate this table.</p>
            ) : (
              <div className="table-wrap">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Time (UTC)</th>
                      <th>File</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                      <th>Threshold</th>
                      <th>Mode</th>
                      <th>Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historyRecords.map((row) => (
                      <tr key={row.id}>
                        <td>{new Date(row.created_at_utc).toLocaleString()}</td>
                        <td>{row.file_name || "-"}</td>
                        <td>{row.predicted_label}</td>
                        <td>{(row.confidence * 100).toFixed(1)}%</td>
                        <td>{row.threshold.toFixed(2)}</td>
                        <td>{row.inference_mode}</td>
                        <td className="feedback-cell">
                          <button
                            type="button"
                            className={`feedback-btn ${row.feedback === "correct" ? "active-correct" : ""}`}
                            title="Mark as correct"
                            onClick={() => onFeedback(row.id, row.feedback === "correct" ? "clear" : "correct")}
                            aria-label="Mark correct"
                          >
                            &#x2713;
                          </button>
                          <button
                            type="button"
                            className={`feedback-btn ${row.feedback === "incorrect" ? "active-incorrect" : ""}`}
                            title="Mark as incorrect"
                            onClick={() => onFeedback(row.id, row.feedback === "incorrect" ? "clear" : "incorrect")}
                            aria-label="Mark incorrect"
                          >
                            &#x2717;
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </main>
  );
}

export default App;
