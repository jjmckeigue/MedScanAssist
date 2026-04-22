import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { getModelInfo, healthCheck } from "./api";
import AnalyzePage from "./pages/AnalyzePage";
import HistoryPage from "./pages/HistoryPage";
import PatientDetailPage from "./pages/PatientDetailPage";
import PatientsPage from "./pages/PatientsPage";

const BRAND_ASSETS = { light: "/branding/logo_light.png" };

function App() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  const refreshStatus = async () => {
    try {
      const [h, m] = await Promise.all([healthCheck(), getModelInfo()]);
      setHealth(h);
      setModelInfo(m);
    } catch {
      /* status bar will show unavailable */
    }
  };

  useEffect(() => { refreshStatus(); }, []);

  const statusOk = health?.status === "ok";

  return (
    <main className="page">
      <header className="hero">
        <div className="brand-strip" aria-label="MedScanAssist branding">
          <img
            className="brand-wordmark"
            src={BRAND_ASSETS.light}
            alt="MedScanAssist logo"
            loading="eager"
            onError={(e) => { e.currentTarget.style.display = "none"; }}
          />
        </div>
        <h1 className="sr-only">MedScanAssist</h1>
        <p>Upload a chest X-ray, run inference, and inspect Grad-CAM in a transparent workflow.</p>
      </header>

      <nav className="tabs" aria-label="Main navigation">
        <NavLink to="/" end className={({ isActive }) => `tab-button ${isActive ? "active" : ""}`}>
          Analyze
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => `tab-button ${isActive ? "active" : ""}`}>
          Review History
        </NavLink>
        <NavLink to="/patients" className={({ isActive }) => `tab-button ${isActive ? "active" : ""}`}>
          Patients
        </NavLink>
      </nav>

      <section className="status-bar" aria-label="System status">
        <span className={`pill ${statusOk ? "ok" : "warn"}`}>
          API: {statusOk ? "Healthy" : "Unavailable"}
        </span>
        <span className={`pill ${modelInfo?.checkpoint_loaded ? "ok" : "warn"}`}>
          Mode: {modelInfo?.inference_mode || "unknown"}
        </span>
        <span className="pill neutral">Arch: {modelInfo?.model_arch || "n/a"}</span>
        <button type="button" className="ghost" onClick={refreshStatus}>Refresh status</button>
      </section>

      <Routes>
        <Route path="/" element={<AnalyzePage />} />
        <Route path="/analyze" element={<AnalyzePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/patients" element={<PatientsPage />} />
        <Route path="/patients/:patientId" element={<PatientDetailPage />} />
      </Routes>
    </main>
  );
}

export default App;
