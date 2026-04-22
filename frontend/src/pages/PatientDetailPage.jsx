import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  deletePatient,
  getPatient,
  getPatientProgression,
  imageUrl,
  submitFeedback,
  updatePatient,
} from "../api";

function ProgressionChart({ points }) {
  if (!points || points.length < 2) {
    return <p className="muted">Need at least 2 analyses to show progression.</p>;
  }

  const width = 600;
  const height = 220;
  const pad = { top: 20, right: 20, bottom: 40, left: 50 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xScale = (i) => pad.left + (i / (points.length - 1)) * plotW;
  const yScale = (v) => pad.top + plotH - v * plotH;

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(i).toFixed(1)} ${yScale(p.confidence).toFixed(1)}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="progression-svg" aria-label="Confidence progression chart">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={pad.left} x2={width - pad.right} y1={yScale(v)} y2={yScale(v)} stroke="#e2e8f0" strokeWidth="1" />
          <text x={pad.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize="11" fill="#64748b">
            {(v * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      <path d={pathD} fill="none" stroke="#2f6fdb" strokeWidth="2.5" strokeLinejoin="round" />

      {points.map((p, i) => (
        <g key={p.id}>
          <circle
            cx={xScale(i)} cy={yScale(p.confidence)} r="5"
            fill={p.predicted_label.toLowerCase() === "pneumonia" ? "#e04545" : "#16a34a"}
            stroke="#fff" strokeWidth="2"
          />
          <title>
            {new Date(p.created_at_utc).toLocaleDateString()} - {p.predicted_label} ({(p.confidence * 100).toFixed(1)}%)
          </title>
        </g>
      ))}

      {points.map((p, i) => {
        if (points.length <= 8 || i === 0 || i === points.length - 1 || i % Math.ceil(points.length / 6) === 0) {
          return (
            <text
              key={`label-${p.id}`}
              x={xScale(i)} y={height - 8}
              textAnchor="middle" fontSize="10" fill="#64748b"
            >
              {new Date(p.created_at_utc).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
            </text>
          );
        }
        return null;
      })}

      <text x={6} y={pad.top + plotH / 2} textAnchor="middle" fontSize="11" fill="#64748b"
        transform={`rotate(-90, 6, ${pad.top + plotH / 2})`}>
        Confidence
      </text>
    </svg>
  );
}

export default function PatientDetailPage() {
  const { patientId } = useParams();
  const navigate = useNavigate();

  const [patient, setPatient] = useState(null);
  const [progression, setProgression] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(false);
  const [editData, setEditData] = useState({});
  const [saving, setSaving] = useState(false);

  const refresh = async () => {
    setError("");
    try {
      const [detail, prog] = await Promise.all([
        getPatient(patientId),
        getPatientProgression(patientId),
      ]);
      setPatient(detail);
      setProgression(prog.points || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, [patientId]);

  const onSaveEdit = async () => {
    setSaving(true);
    try {
      await updatePatient(patientId, editData);
      setEditing(false);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const onDelete = async () => {
    if (!window.confirm(`Delete ${patient.first_name} ${patient.last_name}? Their analyses will be unlinked but preserved.`)) return;
    try {
      await deletePatient(patientId);
      navigate("/patients");
    } catch (err) {
      setError(err.message);
    }
  };

  const onFeedback = async (recordId, feedback) => {
    try {
      await submitFeedback(recordId, feedback);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <p className="muted">Loading patient...</p>;
  if (!patient) return <p className="error">Patient not found.</p>;

  return (
    <>
      <div className="patient-detail-nav">
        <Link to="/patients" className="ghost back-link">&larr; All Patients</Link>
      </div>

      {error && <p className="error">{error}</p>}

      <section className="card patient-header-card">
        {!editing ? (
          <>
            <div className="patient-info-row">
              <div>
                <h2 className="patient-name">{patient.last_name}, {patient.first_name}</h2>
                <div className="patient-meta">
                  {patient.medical_record_number && (
                    <span className="pill neutral">MRN: {patient.medical_record_number}</span>
                  )}
                  {patient.date_of_birth && (
                    <span className="pill neutral">DOB: {patient.date_of_birth}</span>
                  )}
                  <span className="pill neutral">{patient.analysis_count} analysis{patient.analysis_count !== 1 ? "es" : ""}</span>
                </div>
                {patient.notes && <p className="patient-notes">{patient.notes}</p>}
              </div>
              <div className="patient-actions">
                <Link to={`/analyze?patient_id=${patient.id}`} className="analyze-link-btn">
                  New Analysis
                </Link>
                <button type="button" className="ghost" onClick={() => {
                  setEditing(true);
                  setEditData({
                    first_name: patient.first_name,
                    last_name: patient.last_name,
                    date_of_birth: patient.date_of_birth || "",
                    medical_record_number: patient.medical_record_number || "",
                    notes: patient.notes || "",
                  });
                }}>
                  Edit
                </button>
                <button type="button" className="ghost danger-ghost" onClick={onDelete}>Delete</button>
              </div>
            </div>
          </>
        ) : (
          <div className="patient-form compact">
            <div className="form-row">
              <label className="field">
                <span>First Name</span>
                <input type="text" value={editData.first_name} onChange={(e) => setEditData({ ...editData, first_name: e.target.value })} />
              </label>
              <label className="field">
                <span>Last Name</span>
                <input type="text" value={editData.last_name} onChange={(e) => setEditData({ ...editData, last_name: e.target.value })} />
              </label>
            </div>
            <div className="form-row">
              <label className="field">
                <span>Date of Birth</span>
                <input type="date" value={editData.date_of_birth} onChange={(e) => setEditData({ ...editData, date_of_birth: e.target.value })} />
              </label>
              <label className="field">
                <span>MRN</span>
                <input type="text" value={editData.medical_record_number} onChange={(e) => setEditData({ ...editData, medical_record_number: e.target.value })} />
              </label>
            </div>
            <label className="field">
              <span>Notes</span>
              <textarea rows={2} value={editData.notes} onChange={(e) => setEditData({ ...editData, notes: e.target.value })} />
            </label>
            <div className="row">
              <button type="button" onClick={onSaveEdit} disabled={saving}>{saving ? "Saving..." : "Save"}</button>
              <button type="button" className="ghost" onClick={() => setEditing(false)}>Cancel</button>
            </div>
          </div>
        )}
      </section>

      {progression.length > 0 && (
        <section className="card">
          <h3>Confidence Progression</h3>
          <p className="muted">Tracking pneumonia detection confidence across visits. Red dots = pneumonia, green dots = normal.</p>
          <ProgressionChart points={progression} />
        </section>
      )}

      <section className="card">
        <div className="history-header">
          <h3>Analysis History</h3>
          <Link to={`/analyze?patient_id=${patient.id}`} className="ghost view-btn">+ New Analysis</Link>
        </div>

        {(!patient.analyses || patient.analyses.length === 0) ? (
          <p className="muted">No analyses for this patient yet. Run an analysis linked to this profile.</p>
        ) : (
          <div className="analysis-timeline">
            {patient.analyses.map((a) => (
              <article key={a.id} className="timeline-card">
                <div className="timeline-card-header">
                  <span className="timeline-date">
                    {new Date(a.created_at_utc).toLocaleDateString(undefined, {
                      year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
                    })}
                  </span>
                  <span className={`pill ${a.predicted_label.toLowerCase() === "pneumonia" ? "warn" : "ok"}`}>
                    {a.predicted_label.toUpperCase()}
                  </span>
                </div>
                <div className="timeline-card-body">
                  {a.image_path && (
                    <img className="timeline-thumb" src={imageUrl(a.image_path)} alt={`X-ray from ${a.created_at_utc}`} loading="lazy" />
                  )}
                  <div className="timeline-details">
                    <p><strong>Confidence:</strong> {(a.confidence * 100).toFixed(1)}%</p>
                    <p><strong>Threshold:</strong> {a.threshold.toFixed(2)}</p>
                    <p><strong>File:</strong> {a.file_name || "—"}</p>
                    <div className="feedback-cell">
                      <button
                        type="button"
                        className={`feedback-btn ${a.feedback === "correct" ? "active-correct" : ""}`}
                        title="Mark correct"
                        onClick={() => onFeedback(a.id, a.feedback === "correct" ? "clear" : "correct")}
                      >&#x2713;</button>
                      <button
                        type="button"
                        className={`feedback-btn ${a.feedback === "incorrect" ? "active-incorrect" : ""}`}
                        title="Mark incorrect"
                        onClick={() => onFeedback(a.id, a.feedback === "incorrect" ? "clear" : "incorrect")}
                      >&#x2717;</button>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
