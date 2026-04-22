import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getPatients, createPatient } from "../api";

export default function PatientsPage() {
  const [patients, setPatients] = useState([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({
    first_name: "", last_name: "", date_of_birth: "", medical_record_number: "", notes: "",
  });
  const [formError, setFormError] = useState("");
  const [saving, setSaving] = useState(false);

  const refresh = async (query = search) => {
    setError("");
    try {
      const data = await getPatients(query);
      setPatients(data.patients || []);
      setTotal(data.total || 0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    const timer = setTimeout(() => refresh(search), 250);
    return () => clearTimeout(timer);
  }, [search]);

  const onSubmit = async (e) => {
    e.preventDefault();
    setFormError("");
    if (!formData.first_name.trim() || !formData.last_name.trim()) {
      setFormError("First and last name are required.");
      return;
    }
    setSaving(true);
    try {
      await createPatient({
        ...formData,
        date_of_birth: formData.date_of_birth || null,
        medical_record_number: formData.medical_record_number || null,
      });
      setFormData({ first_name: "", last_name: "", date_of_birth: "", medical_record_number: "", notes: "" });
      setShowForm(false);
      await refresh();
    } catch (err) {
      setFormError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <section className="panel">
        <div className="panel-heading">
          <h2>Patient Profiles</h2>
          <p>Manage patient records and track X-ray analyses over time.</p>
        </div>

        <div className="patients-toolbar">
          <label htmlFor="patient-search" className="sr-only">Search patients</label>
          <input
            id="patient-search"
            className="patient-search-input wide"
            type="text"
            placeholder="Search by name or MRN..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button type="button" onClick={() => setShowForm(!showForm)}>
            {showForm ? "Cancel" : "Add Patient"}
          </button>
        </div>

        {showForm && (
          <form className="patient-form" onSubmit={onSubmit}>
            <div className="form-row">
              <label className="field">
                <span>First Name *</span>
                <input
                  type="text" required maxLength={100}
                  value={formData.first_name}
                  onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                />
              </label>
              <label className="field">
                <span>Last Name *</span>
                <input
                  type="text" required maxLength={100}
                  value={formData.last_name}
                  onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                />
              </label>
            </div>
            <div className="form-row">
              <label className="field">
                <span>Date of Birth</span>
                <input
                  type="date"
                  value={formData.date_of_birth}
                  onChange={(e) => setFormData({ ...formData, date_of_birth: e.target.value })}
                />
              </label>
              <label className="field">
                <span>Medical Record Number</span>
                <input
                  type="text" maxLength={50} placeholder="e.g. MRN-001234"
                  value={formData.medical_record_number}
                  onChange={(e) => setFormData({ ...formData, medical_record_number: e.target.value })}
                />
              </label>
            </div>
            <label className="field">
              <span>Notes</span>
              <textarea
                rows={2} maxLength={2000} placeholder="Optional clinical notes..."
                value={formData.notes}
                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              />
            </label>
            {formError && <p className="error">{formError}</p>}
            <button type="submit" disabled={saving}>{saving ? "Saving..." : "Create Patient"}</button>
          </form>
        )}
      </section>

      {error && <p className="error" role="alert">{error}</p>}

      <section className="card">
        <div className="history-header">
          <h3>{total} Patient{total !== 1 ? "s" : ""}</h3>
        </div>
        {loading ? (
          <p className="muted" role="status">Loading patients...</p>
        ) : error && patients.length === 0 ? (
          <p className="error" role="alert">Failed to load patients. Please try again.</p>
        ) : patients.length === 0 ? (
          <p className="muted">No patients found. Add a patient to get started.</p>
        ) : (
          <div className="table-wrap">
            <table className="history-table">
              <caption className="sr-only">Patient profiles</caption>
              <thead>
                <tr>
                  <th scope="col">Name</th>
                  <th scope="col">MRN</th>
                  <th scope="col">Date of Birth</th>
                  <th scope="col">Created</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p) => (
                  <tr key={p.id}>
                    <td>
                      <Link to={`/patients/${p.id}`} className="patient-link">
                        {p.last_name}, {p.first_name}
                      </Link>
                    </td>
                    <td>{p.medical_record_number || "-"}</td>
                    <td>{p.date_of_birth || "-"}</td>
                    <td>{new Date(p.created_at_utc).toLocaleDateString()}</td>
                    <td>
                      <Link to={`/patients/${p.id}`} className="ghost view-btn">View</Link>
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
