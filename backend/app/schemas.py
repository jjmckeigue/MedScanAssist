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
    analysis_id: int | None = Field(default=None, description="Persistent history record ID.")


class GradCamResponse(BaseModel):
    predicted_label: str
    confidence: float
    heatmap_base64: str = Field(description="Base64-encoded PNG of Grad-CAM overlay.")
    inference_mode: str
    model_arch: str
    checkpoint_loaded: bool
    lung_focus_score: float = Field(
        description="Fraction of Grad-CAM activation concentrated in a heuristic lung ROI (0-1)."
    )
    off_lung_attention_ratio: float = Field(
        description="Fraction of Grad-CAM activation outside the heuristic lung ROI (0-1)."
    )
    explainability_warning: str | None = Field(
        default=None,
        description="Warning message when Grad-CAM attention appears off-lung or low-confidence.",
    )


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


class AnalysisHistoryRecord(BaseModel):
    id: int
    created_at_utc: str
    file_name: str | None = None
    predicted_label: str
    confidence: float
    threshold: float
    inference_mode: str
    model_arch: str
    checkpoint_loaded: bool
    probabilities: dict[str, float]


class AnalysisHistorySummary(BaseModel):
    total_reviews: int
    pneumonia_count: int
    normal_count: int
    avg_confidence: float
