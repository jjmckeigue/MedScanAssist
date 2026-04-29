const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS || 60000);
const HEALTH_TIMEOUT_MS = Number(import.meta.env.VITE_HEALTH_TIMEOUT_MS || 90000);

// ---- Token management ----

export function getAccessToken() {
  return localStorage.getItem("access_token") || "";
}

export function getRefreshToken() {
  return localStorage.getItem("refresh_token") || "";
}

export function clearTokens() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

export function isAuthenticated() {
  return Boolean(getAccessToken());
}

async function refreshAccessToken() {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;

  try {
    const res = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!res.ok) {
      clearTokens();
      return false;
    }

    const data = await res.json();
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    return true;
  } catch {
    clearTokens();
    return false;
  }
}

// ---- Core fetch helpers ----

async function parseJsonSafely(response) {
  return response.json().catch(() => ({}));
}

function authHeaders() {
  const token = getAccessToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const mergedHeaders = { ...authHeaders(), ...(options.headers || {}) };

  try {
    return await fetch(url, {
      ...options,
      headers: mergedHeaders,
      signal: controller.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Request timed out. The server may still be starting up — please try again.");
    }
    throw new Error("Network request failed. Check API connectivity.");
  } finally {
    clearTimeout(timeoutId);
  }
}

async function fetchWithAuth(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  let response = await fetchWithTimeout(url, options, timeoutMs);

  if (response.status === 401 && getRefreshToken()) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      response = await fetchWithTimeout(url, options, timeoutMs);
    }
  }

  return response;
}

async function uploadImage(endpoint, file, query = {}) {
  const formData = new FormData();
  formData.append("file", file);
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.append(key, String(value));
    }
  });
  const suffix = params.toString() ? `?${params.toString()}` : "";

  const response = await fetchWithAuth(`${API_BASE_URL}${endpoint}${suffix}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Request failed");
  }

  return parseJsonSafely(response);
}

// ---- Public API ----

export const healthCheck = async () => {
  const response = await fetchWithAuth(`${API_BASE_URL}/health`, {}, HEALTH_TIMEOUT_MS);
  if (!response.ok) {
    throw new Error("Health check failed");
  }
  return parseJsonSafely(response);
};

export const getModelInfo = async () => {
  const response = await fetchWithAuth(`${API_BASE_URL}/model-info`, {}, HEALTH_TIMEOUT_MS);
  if (!response.ok) {
    throw new Error("Model info request failed");
  }
  return parseJsonSafely(response);
};

export const getHistory = async (limit = 50) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/history?limit=${limit}`);
  if (!response.ok) {
    throw new Error("History request failed");
  }
  return parseJsonSafely(response);
};

export const getHistorySummary = async () => {
  const response = await fetchWithAuth(`${API_BASE_URL}/history/summary`);
  if (!response.ok) {
    throw new Error("History summary request failed");
  }
  return parseJsonSafely(response);
};

export const predictImage = (file, threshold) => uploadImage("/predict", file, { threshold });
export const generateGradCam = (file) => uploadImage("/gradcam", file);
export const analyzeImage = (file, threshold, patientId) =>
  uploadImage("/analyze", file, { threshold, patient_id: patientId });

export const submitFeedback = async (recordId, feedback) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/history/${recordId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feedback }),
  });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Feedback submission failed");
  }
  return parseJsonSafely(response);
};

// ---- Patient API ----

export const getPatients = async (search = "", limit = 100, offset = 0) => {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (search) params.append("search", search);
  const response = await fetchWithAuth(`${API_BASE_URL}/patients?${params}`);
  if (!response.ok) throw new Error("Failed to load patients");
  return parseJsonSafely(response);
};

export const getPatient = async (id) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/patients/${id}`);
  if (!response.ok) throw new Error("Patient not found");
  return parseJsonSafely(response);
};

export const createPatient = async (data) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/patients`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Failed to create patient");
  }
  return parseJsonSafely(response);
};

export const updatePatient = async (id, data) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/patients/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Failed to update patient");
  }
  return parseJsonSafely(response);
};

export const deletePatient = async (id) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/patients/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error("Failed to delete patient");
  return parseJsonSafely(response);
};

export const getPatientProgression = async (id) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/patients/${id}/progression`);
  if (!response.ok) throw new Error("Failed to load progression data");
  return parseJsonSafely(response);
};

export const imageUrl = (filename) => `${API_BASE_URL}/images/${filename}`;

// ---- Auth API ----

export const getCurrentUser = async () => {
  const response = await fetchWithAuth(`${API_BASE_URL}/auth/me`);
  if (!response.ok) throw new Error("Not authenticated");
  return parseJsonSafely(response);
};

export const verifyEmail = async (token) => {
  const response = await fetchWithTimeout(`${API_BASE_URL}/auth/verify?token=${encodeURIComponent(token)}`);
  const data = await parseJsonSafely(response);
  if (!response.ok) throw new Error(data.detail || "Verification failed");
  return data;
};

export const resendVerification = async (email) => {
  const response = await fetchWithTimeout(`${API_BASE_URL}/auth/resend-verification`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
  const data = await parseJsonSafely(response);
  if (!response.ok) throw new Error(data.detail || "Failed to resend verification");
  return data;
};

export const logoutUser = async () => {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return;
  await fetchWithAuth(`${API_BASE_URL}/auth/logout`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });
};

export const updateProfile = async (fullName) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/auth/me`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ full_name: fullName }),
  });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Could not update profile");
  }
  return parseJsonSafely(response);
};

export const changePassword = async (currentPassword, newPassword) => {
  const response = await fetchWithAuth(`${API_BASE_URL}/auth/change-password`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    const msg = typeof payload.detail === "string" ? payload.detail : "Password change failed";
    throw new Error(msg);
  }
  return parseJsonSafely(response);
};

export const getSessions = async () => {
  const headers = { ...authHeaders() };
  const refreshToken = getRefreshToken();
  if (refreshToken) headers["X-Refresh-Token"] = refreshToken;

  const response = await fetchWithAuth(`${API_BASE_URL}/auth/sessions`, { headers });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Could not load sessions");
  }
  return parseJsonSafely(response);
};

export const logoutAllSessions = async () => {
  const response = await fetchWithAuth(`${API_BASE_URL}/auth/logout-all`, {
    method: "POST",
  });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Could not sign out all sessions");
  }
  return parseJsonSafely(response);
};

export const deactivateAccount = async () => {
  const response = await fetchWithAuth(`${API_BASE_URL}/auth/me`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Could not deactivate account");
  }
  return parseJsonSafely(response);
};
