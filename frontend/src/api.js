const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function uploadImage(endpoint, file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed");
  }

  return response.json();
}

export const healthCheck = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("Health check failed");
  }
  return response.json();
};

export const predictImage = (file) => uploadImage("/predict", file);
export const generateGradCam = (file) => uploadImage("/gradcam", file);
