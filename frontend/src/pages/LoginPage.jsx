import { useState } from "react";
import { requestPasswordReset, resendVerification } from "../api";

export default function LoginPage({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingVerification, setPendingVerification] = useState(false);
  const [verificationEmailSent, setVerificationEmailSent] = useState(true);
  const [devVerificationUrl, setDevVerificationUrl] = useState("");
  const [resendStatus, setResendStatus] = useState("");

  const [forgotFlowSuccess, setForgotFlowSuccess] = useState(false);
  const [forgotApiMessage, setForgotApiMessage] = useState("");
  const [forgotResetEmailSent, setForgotResetEmailSent] = useState(false);
  const [forgotDevResetUrl, setForgotDevResetUrl] = useState("");

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

  const goToLogin = () => {
    setMode("login");
    setError("");
    setForgotFlowSuccess(false);
    setForgotApiMessage("");
    setForgotDevResetUrl("");
    setForgotResetEmailSent(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResendStatus("");

    if (mode === "forgot") {
      setLoading(true);
      try {
        const data = await requestPasswordReset(email);
        setForgotApiMessage(data.message || "");
        setForgotResetEmailSent(data.reset_email_sent === true);
        setForgotDevResetUrl(typeof data.dev_reset_url === "string" ? data.dev_reset_url : "");
        setForgotFlowSuccess(true);
      } catch {
        setError("Network error. Is the API running?");
      } finally {
        setLoading(false);
      }
      return;
    }

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
        setVerificationEmailSent(data.verification_email_sent !== false);
        setDevVerificationUrl(
          typeof data.dev_verification_url === "string" ? data.dev_verification_url : ""
        );
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
      const sent = data.verification_email_sent !== false;
      const devUrl =
        typeof data.dev_verification_url === "string" ? data.dev_verification_url : "";
      setVerificationEmailSent(sent);
      setDevVerificationUrl(devUrl);
      if (!sent && devUrl) {
        setResendStatus("Email could not be sent. Use the development verification link below.");
      } else if (!sent) {
        setResendStatus(
          "Email could not be sent. Ask your administrator to set SMTP on the API (check server logs), then try again."
        );
      } else {
        setResendStatus(data.message || "Verification email sent! Check your inbox.");
      }
    } catch {
      setResendStatus("Could not resend. Please try again later.");
    }
  };

  if (forgotFlowSuccess) {
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
            <h1>Check your email</h1>
            <p className="muted">{forgotApiMessage}</p>
          </div>

          <div className="verify-info">
            <div className="verify-icon" aria-hidden="true">&#9993;</div>
            {forgotResetEmailSent ? (
              <p>
                If <strong>{email}</strong> matches an account, you should receive a reset link
                shortly. It expires in about an hour — check spam if needed.
              </p>
            ) : forgotDevResetUrl ? (
              <p>
                Email could not be sent from this environment. Use the link below to reset locally,
                or configure SMTP on the API.
              </p>
            ) : (
              <p>
                If <strong>{email}</strong> is registered, email could not be delivered. Ask your
                administrator to configure SMTP, then try &quot;Forgot password?&quot; again.
              </p>
            )}
          </div>

          {forgotDevResetUrl && (
            <p className="verify-dev-link">
              <a href={forgotDevResetUrl} className="link-btn">
                Open password reset link (local development)
              </a>
            </p>
          )}

          <p className="login-footer muted">
            <button type="button" className="link-btn" onClick={goToLogin}>
              Back to sign in
            </button>
          </p>
        </div>
      </div>
    );
  }

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
              {verificationEmailSent ? (
                <>
                  We sent a verification link to <strong>{email}</strong>. Click the link in the
                  email to activate your account.
                </>
              ) : (
                <>
                  Your account was created, but the server could not send email to{" "}
                  <strong>{email}</strong>. Configure <code className="inline-code">SMTP_PASSWORD</code>{" "}
                  and <code className="inline-code">FRONTEND_URL</code> on the API, then use Resend
                  below.
                </>
              )}
            </p>
          </div>

          <div className="verify-info">
            <div className="verify-icon" aria-hidden="true">&#9993;</div>
            {verificationEmailSent ? (
              <p>The link expires in 24 hours. Check your spam folder if you don&apos;t see it.</p>
            ) : (
              <p>
                Gmail users: dots in your address are ignored for sign-in. If the API database was
                reset on deploy, your old password no longer applies — register again or restore the
                database file.
              </p>
            )}
          </div>

          {devVerificationUrl && (
            <p className="verify-dev-link">
              <a href={devVerificationUrl} className="link-btn">
                Open verification link (local development)
              </a>
            </p>
          )}

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
                setDevVerificationUrl("");
                setVerificationEmailSent(true);
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
            {mode === "forgot"
              ? "Enter your email and we will send a link to reset your password."
              : mode === "login"
              ? "Sign in to access the clinical analysis platform."
              : "Create an account to get started."}
          </p>
        </div>

        {mode === "forgot" ? (
          <p className="forgot-back">
            <button
              type="button"
              className="link-btn"
              onClick={goToLogin}
            >
              ← Back to sign in
            </button>
          </p>
        ) : (
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
        )}

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

          {(mode === "login" || mode === "register") && (
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
          )}

          {mode === "login" && (
            <p className="forgot-row">
              <button
                type="button"
                className="link-btn"
                onClick={() => {
                  setMode("forgot");
                  setError("");
                  setForgotFlowSuccess(false);
                }}
              >
                Forgot password?
              </button>
            </p>
          )}

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {mode === "login" && error && (
            <p className="muted login-hint">
              Using Gmail? Dots in the address are treated as equivalent. If this is a hosted API,
              accounts live in the server database — a reset there means you need to register again
              or restore backups.
            </p>
          )}

          <button type="submit" disabled={loading}>
            {loading
              ? "Please wait..."
              : mode === "forgot"
              ? "Send reset link"
              : mode === "login"
              ? "Sign In"
              : "Create Account"}
          </button>
        </form>

        {mode !== "forgot" && (
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
        )}
      </div>
    </div>
  );
}
