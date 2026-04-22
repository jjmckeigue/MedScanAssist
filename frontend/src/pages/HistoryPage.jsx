import { useEffect, useState } from "react";
import { getHistory, getHistorySummary, submitFeedback } from "../api";

export default function HistoryPage() {
  const [historyRecords, setHistoryRecords] = useState([]);
  const [historySummary, setHistorySummary] = useState(null);
  const [error, setError] = useState("");

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

  useEffect(() => { refreshHistory(); }, []);

  const onFeedback = async (recordId, feedback) => {
    try {
      await submitFeedback(recordId, feedback);
      await refreshHistory();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
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
          <button type="button" className="ghost" onClick={refreshHistory}>Refresh history</button>
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
                      >&#x2713;</button>
                      <button
                        type="button"
                        className={`feedback-btn ${row.feedback === "incorrect" ? "active-incorrect" : ""}`}
                        title="Mark as incorrect"
                        onClick={() => onFeedback(row.id, row.feedback === "incorrect" ? "clear" : "incorrect")}
                        aria-label="Mark incorrect"
                      >&#x2717;</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
