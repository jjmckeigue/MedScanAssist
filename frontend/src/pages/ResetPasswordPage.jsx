import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { resetPasswordWithToken } from "../api";

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get("token")?.trim() ?? "";

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    if (!token) {
      setError("This reset link is missing a token. Open the link from your email again.");
      return;
    }
    setLoading(true);
    try {
      await resetPasswordWithToken(token, password);
      setDone(true);
    } catch (err) {
      setError(err.message || "Could not reset password.");
    } finally {
      setLoading(false);
    }
  };

  if (done) {
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
            <h1>Password updated</h1>
            <p className="muted">You can now sign in with your new password.</p>
          </div>
          <div className="verify-actions">
            <Link to="/" className="verify-signin-link">
              Go to Sign In
            </Link>
          </div>
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
            onError={(e) => { e.currentTarget.style.display = "none"; }}
          />
          <h1>Set a new password</h1>
          <p className="muted">
            Choose a strong password (8+ characters with upper, lower, and a digit).
          </p>
        </div>

        {!token && (
          <p className="error" role="alert">
            This page needs a valid reset link from your email. Request a new link from the sign-in
            page.
          </p>
        )}

        <form onSubmit={handleSubmit} className="login-form">
          <label className="field">
            <span>New password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Min 8 chars, upper + lower + digit"
              required
              minLength={8}
              autoComplete="new-password"
              disabled={!token}
            />
          </label>
          <label className="field">
            <span>Confirm password</span>
            <input
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              placeholder="Repeat new password"
              required
              minLength={8}
              autoComplete="new-password"
              disabled={!token}
            />
          </label>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          <button type="submit" disabled={loading || !token}>
            {loading ? "Please wait..." : "Update password"}
          </button>
        </form>

        <p className="login-footer muted">
          <Link to="/" className="verify-signin-link">
            Back to Sign In
          </Link>
        </p>
      </div>
    </div>
  );
}
