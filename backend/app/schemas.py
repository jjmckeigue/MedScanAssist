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
    inference_mode: str = Field(description="`checkpoint` when model weights are loaded, else `placeholder`.")
    model_arch: str = Field(description="Model architecture used by inference service.")
    checkpoint_loaded: bool = Field(description="True when a checkpoint was successfully loaded.")


class GradCamResponse(BaseModel):
    predicted_label: str
    confidence: float
    heatmap_base64: str = Field(description="Base64-encoded PNG of Grad-CAM overlay.")
    inference_mode: str
    model_arch: str
    checkpoint_loaded: bool


class ModelInfoResponse(BaseModel):
    inference_mode: str
    model_arch: str
    checkpoint_loaded: bool
    class_names: list[str]
    image_size: int
    default_threshold: float
    checkpoint_path: str
    best_epoch: int | None = None
    best_val_acc: float | None = None
    best_val_loss: float | None = None
