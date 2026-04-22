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
    gradcam_mode: str = Field(
        description="'real' when computed from model activations, 'synthetic' when using placeholder heatmap."
    )
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


class AnalyzeResponse(BaseModel):
    """Combined prediction + Grad-CAM in a single response (one forward pass)."""

    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    threshold: float
    inference_mode: str
    model_arch: str
    checkpoint_loaded: bool
    analysis_id: int | None = None
    heatmap_base64: str = Field(description="Base64-encoded PNG of Grad-CAM overlay.")
    gradcam_mode: str = Field(
        description="'real' when computed from model activations, 'synthetic' when using placeholder heatmap."
    )
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
    temperature: float = Field(default=1.0, description="Temperature scaling factor for calibrated probabilities.")


class FeedbackRequest(BaseModel):
    feedback: str = Field(description="Clinician feedback: 'correct', 'incorrect', or 'clear'.")


class FeedbackResponse(BaseModel):
    id: int
    feedback: str | None


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
    feedback: str | None = None
    patient_id: int | None = None
    image_path: str | None = None


class AnalysisHistorySummary(BaseModel):
    total_reviews: int
    pneumonia_count: int
    normal_count: int
    avg_confidence: float
    feedback_correct: int = 0
    feedback_incorrect: int = 0
    feedback_accuracy: float | None = Field(
        default=None, description="Correct / (correct + incorrect) among reviewed predictions, or null if none reviewed."
    )


# ---- Patient schemas ----


class PatientCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    date_of_birth: str | None = Field(default=None, description="ISO date string (YYYY-MM-DD).")
    medical_record_number: str | None = Field(default=None, max_length=50)
    notes: str = Field(default="", max_length=2000)


class PatientUpdate(BaseModel):
    first_name: str | None = Field(default=None, min_length=1, max_length=100)
    last_name: str | None = Field(default=None, min_length=1, max_length=100)
    date_of_birth: str | None = None
    medical_record_number: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=2000)


class PatientResponse(BaseModel):
    id: int
    created_at_utc: str
    updated_at_utc: str
    first_name: str
    last_name: str
    date_of_birth: str | None = None
    medical_record_number: str | None = None
    notes: str = ""


class PatientListResponse(BaseModel):
    patients: list[PatientResponse]
    total: int


class PatientDetailResponse(PatientResponse):
    analyses: list[AnalysisHistoryRecord] = []
    analysis_count: int = 0


class ProgressionPoint(BaseModel):
    id: int
    created_at_utc: str
    predicted_label: str
    confidence: float
    threshold: float


class PatientProgressionResponse(BaseModel):
    patient_id: int
    points: list[ProgressionPoint]


class DriftBin(BaseModel):
    bin: str
    baseline_prop: float
    recent_prop: float
    psi_contribution: float


class DriftReport(BaseModel):
    psi: float | None = Field(description="Population Stability Index (None if insufficient data).")
    drift_detected: bool
    message: str
    baseline_count: int
    recent_count: int
    bins: list[DriftBin] = []
