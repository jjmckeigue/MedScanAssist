import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { analyzeImage, getModelInfo, getPatients } from "../api";

const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB || 8);
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;
const DECISION_PRESETS = { screening: 0.35, balanced: 0.45 };

export default function AnalyzePage() {
  const [searchParams] = useSearchParams();
  const preselectedPatientId = searchParams.get("patient_id");

  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [modelInfo, setModelInfo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [gradcam, setGradcam] = useState(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [decisionMode, setDecisionMode] = useState("custom");

  const [patientId, setPatientId] = useState(preselectedPatientId || "");
  const [patientSearch, setPatientSearch] = useState("");
  const [patientOptions, setPatientOptions] = useState([]);
  const [showPatientDropdown, setShowPatientDropdown] = useState(false);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  useEffect(() => {
    getModelInfo()
      .then((info) => {
        setModelInfo(info);
        if (info?.default_threshold !== undefined) setThreshold(info.default_threshold);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!patientSearch && !preselectedPatientId) { setPatientOptions([]); return; }
    const timer = setTimeout(() => {
      getPatients(patientSearch, 10)
        .then((data) => setPatientOptions(data.patients || []))
        .catch(() => setPatientOptions([]));
    }, 200);
    return () => clearTimeout(timer);
  }, [patientSearch, preselectedPatientId]);

  useEffect(() => {
    if (preselectedPatientId) {
      getPatients("", 200).then((data) => {
        setPatientOptions(data.patients || []);
        setPatientId(preselectedPatientId);
      }).catch(() => {});
    }
  }, [preselectedPatientId]);

  const selectedPatient = patientOptions.find((p) => String(p.id) === String(patientId));

  const onFileChange = (event) => {
    const nextFile = event.target.files?.[0] || null;
    setPrediction(null);
    setGradcam(null);
    setError("");
    if (!nextFile) { setFile(null); return; }
    if (!nextFile.type?.startsWith("image/")) { setFile(null); setError("Please upload an image file."); return; }
    if (nextFile.size > MAX_UPLOAD_BYTES) { setFile(null); setError(`Please upload an image smaller than ${MAX_UPLOAD_MB} MB.`); return; }
    setFile(nextFile);
  };

  const onSelectDecisionMode = (mode) => {
    setError("");
    if (mode === "custom") { setDecisionMode("custom"); return; }
    const preset = DECISION_PRESETS[mode];
    if (preset === undefined) return;
    setDecisionMode(mode);
    setThreshold(preset);
  };

  const onAnalyze = async () => {
    if (!file) { setError("Choose a chest X-ray image first."); return; }
    setError("");
    setLoading(true);
    try {
      const pid = patientId || undefined;
      const result = await analyzeImage(file, threshold.toFixed(2), pid);
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
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const onClear = () => { setFile(null); setPrediction(null); setGradcam(null); setError(""); };

  const onDownloadReport = () => {
    if (!prediction) return;
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
      patient_id: patientId || null,
      model_info: modelInfo || null,
      explainability: gradcam
        ? { lung_focus_score: gradcam.lung_focus_score, off_lung_attention_ratio: gradcam.off_lung_attention_ratio, explainability_warning: gradcam.explainability_warning || null }
        : null,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `medscanassist-report-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const hasPreview = Boolean(previewUrl);
  const hasPrediction = Boolean(prediction);
  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];
  const sliderProgress = ((threshold - 0.05) / (0.95 - 0.05)) * 100;
  const overlayPercent = Math.round(overlayOpacity * 100);

  return (
    <>
      <section className="panel">
        <div className="panel-heading">
          <h2>Run Analysis</h2>
          <p>Upload a chest X-ray, optionally link to a patient profile.</p>
        </div>

        <div className="controls">
          <div className="field">
            <span>Patient (optional)</span>
            <div className="patient-select-wrap">
              <input
                className="patient-search-input"
                type="text"
                placeholder="Search patients by name or MRN..."
                value={selectedPatient ? `${selectedPatient.last_name}, ${selectedPatient.first_name}${selectedPatient.medical_record_number ? ` (${selectedPatient.medical_record_number})` : ""}` : patientSearch}
                onChange={(e) => {
                  setPatientSearch(e.target.value);
                  setPatientId("");
                  setShowPatientDropdown(true);
                }}
                onFocus={() => setShowPatientDropdown(true)}
                onBlur={() => setTimeout(() => setShowPatientDropdown(false), 200)}
              />
              {patientId && (
                <button type="button" className="patient-clear-btn" onClick={() => { setPatientId(""); setPatientSearch(""); }}>
                  &times;
                </button>
              )}
              {showPatientDropdown && patientOptions.length > 0 && !patientId && (
                <ul className="patient-dropdown">
                  {patientOptions.map((p) => (
                    <li key={p.id}>
                      <button
                        type="button"
                        className="patient-option"
                        onMouseDown={() => {
                          setPatientId(String(p.id));
                          setPatientSearch("");
                          setShowPatientDropdown(false);
                        }}
                      >
                        <strong>{p.last_name}, {p.first_name}</strong>
                        {p.medical_record_number && <span className="mrn-tag">{p.medical_record_number}</span>}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <small>Link this analysis to a patient to track their progression over time.</small>
          </div>

          <label className="field">
            <span>Chest X-ray image</span>
            <div className={`file-upload ${file ? "selected" : ""}`}>
              <input id="xray-upload" className="file-input-hidden" type="file" accept="image/*" onChange={onFileChange} />
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
              {["screening", "balanced", "custom"].map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={`mode-pill ${decisionMode === mode ? "active" : ""}`}
                  onClick={() => onSelectDecisionMode(mode)}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
            <small>
              Screening ({DECISION_PRESETS.screening.toFixed(2)}) maximizes F1/sensitivity.
              Balanced ({DECISION_PRESETS.balanced.toFixed(2)}) optimizes Youden-J index.
            </small>
          </div>

          <label className="field">
            <span>Decision threshold: {threshold.toFixed(2)}</span>
            <input
              className="threshold-slider"
              type="range" min="0.05" max="0.95" step="0.01"
              value={threshold}
              style={{ "--slider-progress": `${sliderProgress}%` }}
              onChange={(e) => { setDecisionMode("custom"); setThreshold(Number(e.target.value)); }}
            />
            <small>Higher threshold reduces false positives but can increase false negatives.</small>
          </label>

          <div className="row">
            <button onClick={onAnalyze} type="button" disabled={loading}>
              {loading ? "Analyzing..." : "Analyze X-ray"}
            </button>
            <button onClick={onClear} type="button" className="ghost">Clear</button>
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
                Synthetic heatmap: no trained checkpoint is loaded. This overlay is a center-weighted placeholder and does not reflect real model activations.
              </p>
            )}
            {gradcam?.explainability_warning && gradcam?.gradcam_mode === "real" && (
              <p className="attention-warning">{gradcam.explainability_warning}</p>
            )}
            {gradcam && (
              <p className="muted">
                Mode: {gradcam.gradcam_mode === "real" ? "Real Grad-CAM" : "Synthetic placeholder"}
                {gradcam.gradcam_mode === "real" && (
                  <> | Lung focus: {(gradcam.lung_focus_score * 100).toFixed(1)}% | Off-lung: {(gradcam.off_lung_attention_ratio * 100).toFixed(1)}%</>
                )}
              </p>
            )}
            <div className="cam-wrapper">
              <div className="cam-controls">
                <label className="field">
                  <span>Overlay opacity: {overlayPercent}%</span>
                  <input
                    className="threshold-slider" type="range" min="0" max="1" step="0.01"
                    value={overlayOpacity}
                    style={{ "--slider-progress": `${overlayPercent}%` }}
                    onChange={(e) => setOverlayOpacity(Number(e.target.value))}
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
            <p className="muted">Export JSON for complete machine-readable audit records.</p>
            <div className="row">
              <button type="button" onClick={onDownloadReport} disabled={!prediction}>
                Download report (.json)
              </button>
            </div>
          </section>
        </details>
      )}
    </>
  );
}
