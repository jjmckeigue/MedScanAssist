import { useEffect, useState } from "react";
import { getAdminActivity, getAdminStats, getAdminUsers } from "../api";

const PAGE_SIZE = 25;

export default function AdminPage() {
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [activity, setActivity] = useState([]);
  const [page, setPage] = useState(0);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = async (nextPage = page) => {
    setError("");
    try {
      const [statsRes, usersRes, activityRes] = await Promise.all([
        getAdminStats(),
        getAdminUsers(PAGE_SIZE, nextPage * PAGE_SIZE),
        getAdminActivity(14),
      ]);
      setStats(statsRes);
      setUsers(usersRes.users || []);
      setTotal(usersRes.total ?? 0);
      setActivity(activityRes.scans_per_day || []);
      setPage(nextPage);
    } catch (err) {
      setError(err?.message || "Failed to load admin data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) {
    return <p className="muted" role="status">Loading admin dashboard…</p>;
  }

  const maxScans = activity.reduce((m, r) => Math.max(m, r.count || 0), 0) || 1;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <>
      {error && <p className="error" role="alert">{error}</p>}

      <section className="card">
        <div className="history-header">
          <h2 style={{ margin: 0 }}>Admin dashboard</h2>
          <button type="button" className="ghost" onClick={() => refresh(page)}>
            Refresh
          </button>
        </div>
        <p className="muted" style={{ marginTop: "0.4rem" }}>
          Aggregate, no-PHI view of MedScanAssist usage. Restricted to users with
          the <code>admin</code> role.
        </p>
      </section>

      <section className="grid metrics-grid" aria-label="User metrics">
        <article className="card metric-card">
          <h3>Total users</h3>
          <p className="metric-value">{stats?.total_users ?? 0}</p>
          <p className="muted metric-sub">{stats?.active_users ?? 0} active</p>
        </article>
        <article className="card metric-card">
          <h3>Verified</h3>
          <p className="metric-value">{stats?.verified_users ?? 0}</p>
          <p className="muted metric-sub">
            {stats && stats.total_users > 0
              ? `${((stats.verified_users / stats.total_users) * 100).toFixed(0)}% of users`
              : "—"}
          </p>
        </article>
        <article className="card metric-card">
          <h3>Signups (7d)</h3>
          <p className="metric-value">{stats?.signups_last_7d ?? 0}</p>
          <p className="muted metric-sub">{stats?.signups_last_30d ?? 0} in 30d</p>
        </article>
        <article className="card metric-card">
          <h3>Total scans</h3>
          <p className="metric-value">{stats?.total_scans ?? 0}</p>
          <p className="muted metric-sub">
            {stats?.scans_last_24h ?? 0} in last 24h
          </p>
        </article>
        <article className="card metric-card">
          <h3>Scans (7d)</h3>
          <p className="metric-value">{stats?.scans_last_7d ?? 0}</p>
          <p className="muted metric-sub">{stats?.scans_last_30d ?? 0} in 30d</p>
        </article>
        <article className="card metric-card">
          <h3>Predictions</h3>
          <p className="metric-value">
            {stats?.pneumonia_count ?? 0}
            <span className="muted" style={{ fontSize: "0.8rem", marginLeft: "0.4rem" }}>pneumonia</span>
          </p>
          <p className="muted metric-sub">
            {stats?.normal_count ?? 0} normal
          </p>
        </article>
      </section>

      <section className="card" aria-label="Scans per day">
        <div className="history-header">
          <h3>Scans per day (last 14 days, UTC)</h3>
        </div>
        {activity.length === 0 ? (
          <p className="muted">No scans recorded yet.</p>
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              gap: "0.4rem",
              height: "120px",
              padding: "0.4rem 0",
            }}
          >
            {activity.map((d) => {
              const heightPct = Math.max(4, (d.count / maxScans) * 100);
              return (
                <div
                  key={d.day}
                  style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: "0.25rem",
                  }}
                  title={`${d.day}: ${d.count}`}
                >
                  <div
                    style={{
                      width: "100%",
                      background: "var(--primary)",
                      height: `${heightPct}%`,
                      borderRadius: "0.3rem 0.3rem 0 0",
                      minHeight: "2px",
                    }}
                    aria-hidden="true"
                  />
                  <span
                    className="muted"
                    style={{ fontSize: "0.7rem", whiteSpace: "nowrap" }}
                  >
                    {d.day.slice(5)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="card" aria-label="Users">
        <div className="history-header">
          <h3>Recent users</h3>
          <span className="muted">
            {total} total · page {page + 1} of {totalPages}
          </span>
        </div>
        {users.length === 0 ? (
          <p className="muted">No users yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="history-table">
              <caption className="sr-only">Registered users</caption>
              <thead>
                <tr>
                  <th scope="col">Signed up (UTC)</th>
                  <th scope="col">Email</th>
                  <th scope="col">Full name</th>
                  <th scope="col">Role</th>
                  <th scope="col">Verified</th>
                  <th scope="col">Active</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id}>
                    <td>{new Date(u.created_at_utc).toLocaleString()}</td>
                    <td>{u.email}</td>
                    <td>{u.full_name}</td>
                    <td>
                      <span className={`pill ${u.role === "admin" ? "ok" : "neutral"}`}>
                        {u.role}
                      </span>
                    </td>
                    <td>
                      <span className={`pill ${u.is_verified ? "ok" : "warn"}`}>
                        {u.is_verified ? "yes" : "no"}
                      </span>
                    </td>
                    <td>
                      <span className={`pill ${u.is_active ? "ok" : "warn"}`}>
                        {u.is_active ? "yes" : "no"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div
          style={{
            display: "flex",
            gap: "0.5rem",
            justifyContent: "flex-end",
            marginTop: "0.6rem",
          }}
        >
          <button
            type="button"
            className="ghost"
            disabled={page <= 0}
            onClick={() => refresh(page - 1)}
          >
            ← Previous
          </button>
          <button
            type="button"
            className="ghost"
            disabled={page + 1 >= totalPages}
            onClick={() => refresh(page + 1)}
          >
            Next →
          </button>
        </div>
      </section>
    </>
  );
}
