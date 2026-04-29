import { useCallback, useEffect, useRef, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { clearTokens, getCurrentUser, getModelInfo, healthCheck, isAuthenticated } from "./api";
import AnalyzePage from "./pages/AnalyzePage";
import HistoryPage from "./pages/HistoryPage";
import LoginPage from "./pages/LoginPage";
import PatientDetailPage from "./pages/PatientDetailPage";
import PatientsPage from "./pages/PatientsPage";
import VerifyPage from "./pages/VerifyPage";

const BRAND_ASSETS = { light: "/branding/logo_light.png" };
const RETRY_INTERVAL_MS = 5000;
const MAX_RETRIES = 12;

function App() {
  const [authed, setAuthed] = useState(isAuthenticated());
  const [currentUser, setCurrentUser] = useState(null);

  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [connectionState, setConnectionState] = useState("connecting");
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const retryCount = useRef(0);
  const retryTimer = useRef(null);

  const handleLogin = useCallback(() => {
    setAuthed(true);
    getCurrentUser()
      .then(setCurrentUser)
      .catch(() => setCurrentUser(null));
  }, []);

  const handleLogout = useCallback(() => {
    clearTokens();
    setAuthed(false);
    setCurrentUser(null);
  }, []);

  useEffect(() => {
    if (authed) {
      getCurrentUser()
        .then(setCurrentUser)
        .catch(() => {
          clearTokens();
          setAuthed(false);
        });
    }
  }, [authed]);

  const attemptConnection = useCallback(async () => {
    try {
      const [h, m] = await Promise.all([healthCheck(), getModelInfo()]);
      setHealth(h);
      setModelInfo(m);
      setConnectionState("connected");
      setLastRefreshed(new Date());
      retryCount.current = 0;
      if (retryTimer.current) clearTimeout(retryTimer.current);
      return true;
    } catch {
      retryCount.current += 1;
      if (retryCount.current >= MAX_RETRIES) {
        setConnectionState("unavailable");
      }
      return false;
    }
  }, []);

  useEffect(() => {
    if (!authed) return;
    let cancelled = false;

    const connect = async () => {
      const ok = await attemptConnection();
      if (ok || cancelled) return;

      const scheduleRetry = () => {
        if (cancelled || retryCount.current >= MAX_RETRIES) return;
        retryTimer.current = setTimeout(async () => {
          if (cancelled) return;
          const success = await attemptConnection();
          if (!success && !cancelled) scheduleRetry();
        }, RETRY_INTERVAL_MS);
      };
      scheduleRetry();
    };

    connect();
    return () => {
      cancelled = true;
      if (retryTimer.current) clearTimeout(retryTimer.current);
    };
  }, [attemptConnection, authed]);

  const onRefreshStatus = async () => {
    setRefreshing(true);
    try {
      const [h, m] = await Promise.all([healthCheck(), getModelInfo()]);
      setHealth(h);
      setModelInfo(m);
      setConnectionState("connected");
      setLastRefreshed(new Date());
      retryCount.current = 0;
    } catch {
      setConnectionState("unavailable");
    } finally {
      setRefreshing(false);
    }
  };

  const onRetryConnection = () => {
    setConnectionState("connecting");
    retryCount.current = 0;
    attemptConnection().then((ok) => {
      if (!ok) {
        const scheduleRetry = () => {
          if (retryCount.current >= MAX_RETRIES) return;
          retryTimer.current = setTimeout(async () => {
            const success = await attemptConnection();
            if (!success) scheduleRetry();
          }, RETRY_INTERVAL_MS);
        };
        scheduleRetry();
      }
    });
  };

  const location = useLocation();

  // Verify page is always accessible (user clicks link from email)
  if (location.pathname === "/verify") {
    return <VerifyPage />;
  }

  if (!authed) {
    return <LoginPage onLogin={handleLogin} />;
  }

  const statusOk = health?.status === "ok";
  const isConnecting = connectionState === "connecting";
  const isUnavailable = connectionState === "unavailable";

  const timeSinceRefresh = lastRefreshed
    ? `${Math.round((Date.now() - lastRefreshed.getTime()) / 1000)}s ago`
    : null;

  return (
    <>
    <a href="#main-content" className="skip-link">Skip to main content</a>
    <main id="main-content" className="page">
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
        <p>Upload a chest X-ray, run inference, and inspect Eigen-CAM in a transparent workflow.</p>
      </header>

      {currentUser && (
        <div className="user-bar">
          <span className="user-info">
            Signed in as <strong>{currentUser.full_name}</strong> ({currentUser.email})
          </span>
          <button type="button" className="ghost logout-btn" onClick={handleLogout}>
            Sign Out
          </button>
        </div>
      )}

      <aside className="clinical-disclaimer" role="alert">
        <strong>Clinical Use Notice</strong>
        <p>
          MedScanAssist is an AI-assisted screening aid, <em>not</em> a diagnostic tool.
          All outputs must be independently reviewed by a qualified healthcare professional
          before any clinical decision is made.
        </p>
        <details className="disclaimer-details">
          <summary>Algorithm &amp; validation details</summary>
          <div className="disclaimer-body">
            <p>
              <strong>Intended use:</strong> Binary pneumonia screening on frontal (PA/AP)
              adult chest X-ray images (PNG / JPEG). Not validated for pediatric populations.
            </p>
            <p>
              <strong>Algorithm:</strong> DenseNet-121 convolutional neural network fine-tuned
              on the NIH ChestX-ray14 dataset (112,120 images, 30,805 patients). Patient-level
              splits prevent data leakage between train/val/test sets.
            </p>
            <p>
              <strong>Validation performance (test set, n=984):</strong> Accuracy 75.4%,
              Sensitivity 70.0%, Specificity 72.2%, F1 (pneumonia) 0.555.
              Shortcut reliance index 0.20 (low risk).
            </p>
            <p>
              <strong>Known limitations:</strong> Labels are NLP-extracted from radiology reports
              (not expert-annotated), binary classification only (pneumonia vs. normal), no
              multi-label pathology detection, not validated on DICOM inputs or across
              diverse clinical sites.
            </p>
          </div>
        </details>
      </aside>

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

      {isConnecting && (
        <div className="connection-banner connecting" role="status" aria-live="polite">
          <div className="connection-spinner" aria-hidden="true" />
          <div>
            <strong>Connecting to API service...</strong>
            <p>The backend may take up to 60 seconds to wake up on first load. Please wait.</p>
          </div>
        </div>
      )}

      {isUnavailable && (
        <div className="connection-banner unavailable" role="alert" aria-live="assertive">
          <div>
            <strong>Unable to reach the API service.</strong>
            <p>The backend may be temporarily down. Check your connection or try again.</p>
          </div>
          <button type="button" onClick={onRetryConnection}>Retry Connection</button>
        </div>
      )}

      <section className="status-bar" aria-label="System status">
        <span className={`pill ${isConnecting ? "connecting-pill" : statusOk ? "ok" : "warn"}`}>
          {isConnecting ? "API: Connecting..." : statusOk ? "API: Healthy" : "API: Unavailable"}
        </span>
        <span className={`pill ${modelInfo?.checkpoint_loaded ? "ok" : "warn"}`}>
          Mode: {modelInfo?.inference_mode || "unknown"}
        </span>
        <span className="pill neutral">Arch: {modelInfo?.model_arch || "n/a"}</span>
        <button
          type="button"
          className={`ghost refresh-btn ${refreshing ? "refreshing" : ""}`}
          onClick={onRefreshStatus}
          disabled={refreshing || isConnecting}
        >
          {refreshing ? "Refreshing..." : "Refresh status"}
        </button>
        {lastRefreshed && !refreshing && (
          <span className="last-refreshed">Updated {timeSinceRefresh}</span>
        )}
      </section>

      <details className="card disclosure">
        <summary>Model transparency details</summary>
        <section className="disclosure-body">
          <h3>Model Transparency</h3>
          {!modelInfo && isConnecting ? (
            <p className="muted">Waiting for API connection...</p>
          ) : !modelInfo ? (
            <p className="muted">Model information unavailable. Refresh status or check API connectivity.</p>
          ) : (
            <>
              <div className="meta-grid">
                <p><strong>Inference mode:</strong> {modelInfo.inference_mode}</p>
                <p><strong>Architecture:</strong> {modelInfo.model_arch}</p>
                <p><strong>Checkpoint loaded:</strong> {String(modelInfo.checkpoint_loaded)}</p>
                <p><strong>Best epoch:</strong> {modelInfo.best_epoch ?? "n/a"}</p>
                <p>
                  <strong>Best val accuracy:</strong>{" "}
                  {modelInfo.best_val_acc != null ? modelInfo.best_val_acc.toFixed(4) : "n/a"}
                </p>
                <p>
                  <strong>Default threshold:</strong>{" "}
                  {modelInfo.default_threshold != null ? modelInfo.default_threshold.toFixed(2) : "n/a"}
                </p>
                <p>
                  <strong>Temperature scaling:</strong>{" "}
                  {modelInfo.temperature != null ? modelInfo.temperature.toFixed(3) : "n/a"}
                </p>
                <p><strong>Class names:</strong> {modelInfo.class_names?.join(", ") || "n/a"}</p>
              </div>
              <p className="muted">
                The decision threshold controls the sensitivity/specificity tradeoff. Higher values reduce false
                positives but may increase false negatives. Temperature scaling calibrates probability outputs.
              </p>
            </>
          )}
        </section>
      </details>

      <Routes>
        <Route path="/" element={<AnalyzePage />} />
        <Route path="/analyze" element={<AnalyzePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/patients" element={<PatientsPage />} />
        <Route path="/patients/:patientId" element={<PatientDetailPage />} />
      </Routes>
    </main>
    </>
  );
}

export default App;
