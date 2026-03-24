from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str


class PredictionResponse(BaseModel):
    predicted_label: str = Field(description="Predicted class label.")
    confidence: float = Field(description="Confidence score for predicted class.")
    probabilities: dict[str, float] = Field(description="Probability by class label.")
    threshold: float = Field(description="Decision threshold used for positive class.")


class GradCamResponse(BaseModel):
    predicted_label: str
    confidence: float
    heatmap_base64: str = Field(description="Base64-encoded PNG of Grad-CAM overlay.")
