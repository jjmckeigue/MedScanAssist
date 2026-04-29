import { useState } from "react";
import { resendVerification } from "../api";

export default function LoginPage({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingVerification, setPendingVerification] = useState(false);
  const [resendStatus, setResendStatus] = useState("");

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResendStatus("");
    setLoading(true);

    const endpoint = mode === "register" ? "/auth/register" : "/auth/login";
    const body =
      mode === "register"
        ? { email, password, full_name: fullName }
        : { email, password };

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        if (res.status === 403 && data.detail?.includes("verify")) {
          setPendingVerification(true);
          setError("");
        } else {
          setError(data.detail || "Authentication failed.");
        }
        return;
      }

      if (data.requires_verification) {
        setPendingVerification(true);
        return;
      }

      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("refresh_token", data.refresh_token);
      onLogin();
    } catch {
      setError("Network error. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    if (!email) {
      setResendStatus("Please enter your email address first.");
      return;
    }
    setResendStatus("Sending...");
    try {
      const data = await resendVerification(email);
      setResendStatus(data.message || "Verification email sent! Check your inbox.");
    } catch {
      setResendStatus("Could not resend. Please try again later.");
    }
  };

  if (pendingVerification) {
    return (
      <div className="login-page">
        <div className="login-card">
          <div className="login-header">
            <img
              className="login-logo"
              src="/branding/logo_light.png"
              alt="MedScanAssist"
              onError={(e) => { e.currentTarget.style.display = "none"; }}
            />
            <h1>Check Your Email</h1>
            <p className="muted">
              We sent a verification link to <strong>{email}</strong>.
              Click the link in the email to activate your account.
            </p>
          </div>

          <div className="verify-info">
            <div className="verify-icon" aria-hidden="true">&#9993;</div>
            <p>The link expires in 24 hours. Check your spam folder if you don't see it.</p>
          </div>

          <div className="verify-actions">
            <button
              type="button"
              onClick={handleResend}
              className="ghost"
            >
              Resend verification email
            </button>
            {resendStatus && <p className="muted resend-status">{resendStatus}</p>}
          </div>

          <p className="login-footer muted">
            Already verified?{" "}
            <button
              type="button"
              className="link-btn"
              onClick={() => {
                setPendingVerification(false);
                setMode("login");
                setError("");
                setResendStatus("");
              }}
            >
              Sign in
            </button>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-header">
          <img
            className="login-logo"
            src="/branding/logo_light.png"
            alt="MedScanAssist"
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />
          <h1>MedScanAssist</h1>
          <p className="muted">
            {mode === "login"
              ? "Sign in to access the clinical analysis platform."
              : "Create an account to get started."}
          </p>
        </div>

        <div className="login-toggle" role="group" aria-label="Authentication mode">
          <button
            type="button"
            className={`mode-pill ${mode === "login" ? "active" : ""}`}
            onClick={() => {
              setMode("login");
              setError("");
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`mode-pill ${mode === "register" ? "active" : ""}`}
            onClick={() => {
              setMode("register");
              setError("");
            }}
          >
            Register
          </button>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {mode === "register" && (
            <label className="field">
              <span>Full Name</span>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Dr. Jane Smith"
                required
                minLength={1}
                maxLength={200}
                autoComplete="name"
              />
            </label>
          )}

          <label className="field">
            <span>Email</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="clinician@hospital.org"
              required
              autoComplete="email"
            />
          </label>

          <label className="field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={
                mode === "register"
                  ? "Min 8 chars, upper + lower + digit"
                  : "Enter your password"
              }
              required
              minLength={mode === "register" ? 8 : 1}
              autoComplete={mode === "register" ? "new-password" : "current-password"}
            />
          </label>

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}

          <button type="submit" disabled={loading}>
            {loading
              ? "Please wait..."
              : mode === "login"
              ? "Sign In"
              : "Create Account"}
          </button>
        </form>

        <p className="login-footer muted">
          {mode === "login" ? (
            <>
              Don&apos;t have an account?{" "}
              <button
                type="button"
                className="link-btn"
                onClick={() => {
                  setMode("register");
                  setError("");
                }}
              >
                Register here
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                type="button"
                className="link-btn"
                onClick={() => {
                  setMode("login");
                  setError("");
                }}
              >
                Sign in
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
