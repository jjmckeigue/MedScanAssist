const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS || 20000);

async function parseJsonSafely(response) {
  return response.json().catch(() => ({}));
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Request timed out. Please try again.");
    }
    throw new Error("Network request failed. Check API connectivity.");
  } finally {
    clearTimeout(timeoutId);
  }
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

  const response = await fetchWithTimeout(`${API_BASE_URL}${endpoint}${suffix}`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const payload = await parseJsonSafely(response);
    throw new Error(payload.detail || "Request failed");
  }

  return parseJsonSafely(response);
}

export const healthCheck = async () => {
  const response = await fetchWithTimeout(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("Health check failed");
  }
  return parseJsonSafely(response);
};

export const getModelInfo = async () => {
  const response = await fetchWithTimeout(`${API_BASE_URL}/model-info`);
  if (!response.ok) {
    throw new Error("Model info request failed");
  }
  return parseJsonSafely(response);
};

export const getHistory = async (limit = 50) => {
  const response = await fetchWithTimeout(`${API_BASE_URL}/history?limit=${limit}`);
  if (!response.ok) {
    throw new Error("History request failed");
  }
  return parseJsonSafely(response);
};

export const getHistorySummary = async () => {
  const response = await fetchWithTimeout(`${API_BASE_URL}/history/summary`);
  if (!response.ok) {
    throw new Error("History summary request failed");
  }
  return parseJsonSafely(response);
};

export const predictImage = (file, threshold) => uploadImage("/predict", file, { threshold });
export const generateGradCam = (file) => uploadImage("/gradcam", file);
