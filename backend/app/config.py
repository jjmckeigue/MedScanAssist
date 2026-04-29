from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="MedScanAssist API", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    dataset_root: Path = Field(default=Path("./data/raw/chest_xray"), alias="DATASET_ROOT")
    upload_dir: Path = Field(default=Path("./backend/uploads"), alias="UPLOAD_DIR")
    checkpoint_path: Path = Field(
        default=Path("./backend/checkpoints/best_model.onnx"), alias="CHECKPOINT_PATH"
    )
    model_meta_path: Path = Field(
        default=Path("./backend/checkpoints/model_meta.json"), alias="MODEL_META_PATH"
    )
    history_db_path: Path = Field(default=Path("./backend/artifacts/history.db"), alias="HISTORY_DB_PATH")

    model_arch: str = Field(default="densenet121", alias="MODEL_ARCH")
    class_names: str = Field(default="normal,pneumonia", alias="CLASS_NAMES")
    image_size: int = Field(default=224, alias="IMAGE_SIZE")
    default_threshold: float = Field(default=0.45, alias="DEFAULT_THRESHOLD")
    require_checkpoint: bool = Field(default=False, alias="REQUIRE_CHECKPOINT")
    max_upload_bytes: int = Field(default=8 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    jwt_secret_key: str = Field(default="change-me-in-production", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")

    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="medscanassist@gmail.com", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from_name: str = Field(default="MedScanAssist", alias="SMTP_FROM_NAME")
    verification_token_expire_hours: int = Field(default=24, alias="VERIFICATION_TOKEN_EXPIRE_HOURS")
    frontend_url: str = Field(default="http://localhost:5173", alias="FRONTEND_URL")

    api_key: str = Field(default="", alias="API_KEY")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def class_name_list(self) -> list[str]:
        return [name.strip() for name in self.class_names.split(",") if name.strip()]


settings = Settings()
