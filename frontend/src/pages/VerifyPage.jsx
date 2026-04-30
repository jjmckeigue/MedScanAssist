import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { verifyEmail } from "../api";

export default function VerifyPage() {
  const [searchParams] = useSearchParams();
  const [status, setStatus] = useState("verifying");
  const [message, setMessage] = useState("");

  const token = searchParams.get("token")?.trim() ?? "";

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setMessage("No verification token provided.");
      return;
    }

    let cancelled = false;

    verifyEmail(token)
      .then((data) => {
        if (cancelled) return;
        setStatus("success");
        setMessage(data.message || "Email verified successfully!");
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus("error");
        setMessage(err.message || "Verification failed.");
      });

    return () => { cancelled = true; };
  }, [token]);

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
          <h1>Email Verification</h1>
        </div>

        <div className="verify-info">
          {status === "verifying" && (
            <>
              <div className="verify-icon spinning" aria-hidden="true">&#8635;</div>
              <p>Verifying your email address...</p>
            </>
          )}
          {status === "success" && (
            <>
              <div className="verify-icon success" aria-hidden="true">&#10003;</div>
              <p>{message}</p>
            </>
          )}
          {status === "error" && (
            <>
              <div className="verify-icon error-icon" aria-hidden="true">&#10007;</div>
              <p>{message}</p>
            </>
          )}
        </div>

        <div className="verify-actions">
          <a href="/" className="verify-signin-link">
            Go to Sign In
          </a>
        </div>
      </div>
    </div>
  );
}
