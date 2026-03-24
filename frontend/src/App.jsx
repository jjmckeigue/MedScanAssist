import { useMemo, useState } from "react";
import { generateGradCam, healthCheck, predictImage } from "./api";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [gradcam, setGradcam] = useState(null);
  const [error, setError] = useState("");

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  const onCheckHealth = async () => {
    setError("");
    try {
      const result = await healthCheck();
      setHealth(result);
    } catch (err) {
      setError(err.message);
    }
  };

  const onRunPrediction = async () => {
    if (!file) {
      setError("Select an image first.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const [pred, cam] = await Promise.all([predictImage(file), generateGradCam(file)]);
      setPrediction(pred);
      setGradcam(cam);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1>MedScanAssist (v1)</h1>
      <p className="subtitle">API-first chest X-ray classification + Grad-CAM scaffold.</p>

      <section className="card">
        <h2>1) Upload Chest X-ray</h2>
        <input
          type="file"
          accept="image/*"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
        <div className="row">
          <button onClick={onCheckHealth} type="button">
            Check API Health
          </button>
          <button onClick={onRunPrediction} type="button" disabled={loading}>
            {loading ? "Running..." : "Run Predict + Grad-CAM"}
          </button>
        </div>
        {health && <pre>{JSON.stringify(health, null, 2)}</pre>}
      </section>

      {previewUrl && (
        <section className="card">
          <h2>2) Image Preview</h2>
          <img className="preview" src={previewUrl} alt="Chest x-ray preview" />
        </section>
      )}

      {prediction && (
        <section className="card">
          <h2>3) Prediction Result</h2>
          <pre>{JSON.stringify(prediction, null, 2)}</pre>
        </section>
      )}

      {gradcam?.heatmap_base64 && (
        <section className="card">
          <h2>4) Grad-CAM Overlay</h2>
          <img
            className="preview"
            src={`data:image/png;base64,${gradcam.heatmap_base64}`}
            alt="Grad-CAM overlay"
          />
        </section>
      )}

      {error && <p className="error">{error}</p>}
    </main>
  );
}

export default App;
