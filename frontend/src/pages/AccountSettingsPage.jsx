import { useCallback, useEffect, useState } from "react";
import ConfirmModal from "../components/ConfirmModal";
import {
  changePassword,
  deactivateAccount,
  getSessions,
  logoutAllSessions,
  updateProfile,
} from "../api";

function formatUtcLabel(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export default function AccountSettingsPage({ currentUser, onProfileUpdated, onAuthClear, onSignOut }) {
  const [fullName, setFullName] = useState(currentUser?.full_name || "");
  const [profileStatus, setProfileStatus] = useState("");
  const [profileSaving, setProfileSaving] = useState(false);

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [passwordStatus, setPasswordStatus] = useState("");
  const [passwordSaving, setPasswordSaving] = useState(false);

  const [sessions, setSessions] = useState([]);
  const [sessionsError, setSessionsError] = useState("");
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [logoutAllBusy, setLogoutAllBusy] = useState(false);
  const [signOutBusy, setSignOutBusy] = useState(false);

  const [deactivateBusy, setDeactivateBusy] = useState(false);

  const [confirmKind, setConfirmKind] = useState(null);
  const [modalError, setModalError] = useState("");

  useEffect(() => {
    setFullName(currentUser?.full_name || "");
  }, [currentUser]);

  const loadSessions = useCallback(async () => {
    setSessionsError("");
    setSessionsLoading(true);
    try {
      const data = await getSessions();
      setSessions(Array.isArray(data) ? data : []);
    } catch (e) {
      setSessionsError(e?.message || "Could not load sessions.");
      setSessions([]);
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const onSaveProfile = async (e) => {
    e.preventDefault();
    setProfileStatus("");
    setProfileSaving(true);
    try {
      await updateProfile(fullName.trim());
      setProfileStatus("Profile saved.");
      await onProfileUpdated?.();
    } catch (err) {
      setProfileStatus(err?.message || "Save failed.");
    } finally {
      setProfileSaving(false);
    }
  };

  const onSavePassword = async (e) => {
    e.preventDefault();
    setPasswordStatus("");
    setPasswordSaving(true);
    try {
      await changePassword(currentPassword, newPassword);
      setPasswordStatus("Password updated.");
      setCurrentPassword("");
      setNewPassword("");
    } catch (err) {
      setPasswordStatus(err?.message || "Password change failed.");
    } finally {
      setPasswordSaving(false);
    }
  };

  const closeModal = () => {
    if (signOutBusy || logoutAllBusy || deactivateBusy) return;
    setModalError("");
    setConfirmKind(null);
  };

  const handleConfirmSignOut = async () => {
    setSignOutBusy(true);
    setModalError("");
    try {
      await onSignOut?.();
      setConfirmKind(null);
    } catch (err) {
      setModalError(err?.message || "Could not sign out.");
    } finally {
      setSignOutBusy(false);
    }
  };

  const handleConfirmLogoutAll = async () => {
    setLogoutAllBusy(true);
    setModalError("");
    setSessionsError("");
    try {
      await logoutAllSessions();
      setConfirmKind(null);
      onAuthClear?.();
    } catch (err) {
      const msg = err?.message || "Could not sign out all sessions.";
      setModalError(msg);
      setSessionsError(msg);
    } finally {
      setLogoutAllBusy(false);
    }
  };

  const handleConfirmDeactivate = async () => {
    setDeactivateBusy(true);
    setModalError("");
    try {
      await deactivateAccount();
      setConfirmKind(null);
      onAuthClear?.();
    } catch (err) {
      setModalError(err?.message || "Deactivation failed.");
    } finally {
      setDeactivateBusy(false);
    }
  };

  if (!currentUser) {
    return (
      <div className="settings-page">
        <p className="muted">Loading account…</p>
      </div>
    );
  }

  return (
    <div className="settings-page">
      <ConfirmModal
        open={confirmKind === "signout"}
        title="Sign out?"
        confirmLabel="Sign out"
        onCancel={closeModal}
        onConfirm={handleConfirmSignOut}
        busy={signOutBusy}
        error={modalError}
      >
        <p className="muted modal-text">
          You will need to sign in again to use MedScanAssist on this browser. Other devices stay signed in
          until you use sign out everywhere there too.
        </p>
      </ConfirmModal>

      <ConfirmModal
        open={confirmKind === "logout-all"}
        title="Sign out everywhere?"
        confirmLabel="Sign out everywhere"
        onCancel={closeModal}
        onConfirm={handleConfirmLogoutAll}
        confirmClassName="danger-ghost"
        busy={logoutAllBusy}
        error={modalError}
      >
        <p className="muted modal-text">
          This ends every active session on all browsers and devices. You will need to sign in again everywhere.
        </p>
      </ConfirmModal>

      <ConfirmModal
        open={confirmKind === "deactivate"}
        title="Deactivate account?"
        confirmLabel="Deactivate account"
        onCancel={closeModal}
        onConfirm={handleConfirmDeactivate}
        confirmClassName="btn-account-exit"
        busy={deactivateBusy}
        error={modalError}
      >
        <p className="muted modal-text">
          Your account will be closed and your email released. You can register again with the same address after
          completing verification.
        </p>
      </ConfirmModal>

      <h2 className="settings-title">Account settings</h2>
      <p className="muted settings-lead">
        Signed in as <strong>{currentUser?.email}</strong>
      </p>

      <div className="settings-stack">
        <section className="card settings-section">
          <h3>Profile</h3>
          <form onSubmit={onSaveProfile} className="login-form settings-form">
            <label className="field">
              <span>Full name</span>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                minLength={1}
                maxLength={200}
                required
                autoComplete="name"
              />
            </label>
            {profileStatus && (
              <p className={profileStatus.includes("failed") ? "error" : "muted"} role="status">
                {profileStatus}
              </p>
            )}
            <button type="submit" disabled={profileSaving}>
              {profileSaving ? "Saving…" : "Save profile"}
            </button>
          </form>
        </section>

        <section className="card settings-section">
          <h3>Change password</h3>
          <form onSubmit={onSavePassword} className="login-form settings-form">
            <label className="field">
              <span>Current password</span>
              <input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
                autoComplete="current-password"
              />
            </label>
            <label className="field">
              <span>New password</span>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
                placeholder="Min 8 chars, upper + lower + digit"
              />
            </label>
            {passwordStatus && (
              <p className={passwordStatus.includes("failed") || passwordStatus.includes("incorrect") ? "error" : "muted"} role="status">
                {passwordStatus}
              </p>
            )}
            <button type="submit" disabled={passwordSaving}>
              {passwordSaving ? "Updating…" : "Update password"}
            </button>
          </form>
        </section>

        <section className="card settings-section">
          <h3>Sign out</h3>
          <button
            type="button"
            className="btn-account-exit"
            onClick={() => {
              setModalError("");
              setConfirmKind("signout");
            }}
          >
            Sign out
          </button>
        </section>

        <section className="card settings-section">
          <h3>Sessions</h3>
          <div className="settings-section-fill">
            {sessionsError && (
              <p className="error" role="alert">
                {sessionsError}
              </p>
            )}
            {sessionsLoading ? (
              <p className="muted">Loading sessions…</p>
            ) : sessions.length === 0 ? (
              <p className="muted">No active refresh sessions.</p>
            ) : (
              <ul className="session-list">
                {sessions.map((s, idx) => (
                  <li
                    key={`${s.created_at_utc}-${s.expires_at_utc}-${idx}`}
                    className={`session-item${s.is_current ? " current" : ""}`}
                  >
                    <strong>{s.is_current ? "This device" : "Other session"}</strong>
                    <div className="session-meta">
                      Started {formatUtcLabel(s.created_at_utc)} · Expires {formatUtcLabel(s.expires_at_utc)}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="settings-actions">
            <button type="button" className="ghost" onClick={loadSessions} disabled={sessionsLoading}>
              Refresh list
            </button>
            <button
              type="button"
              className="danger-ghost"
              onClick={() => {
                setModalError("");
                setConfirmKind("logout-all");
              }}
              disabled={logoutAllBusy || sessionsLoading}
            >
              Sign out everywhere
            </button>
          </div>
        </section>

        <section className="card settings-section settings-danger-zone settings-section-span-full">
          <h3>Deactivate accounts</h3>
          <button
            type="button"
            className="btn-account-exit"
            onClick={() => {
              setModalError("");
              setConfirmKind("deactivate");
            }}
            disabled={deactivateBusy}
          >
            Deactivate account
          </button>
        </section>
      </div>
    </div>
  );
}
