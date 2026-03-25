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
        default=Path("./backend/checkpoints/best_model.pt"), alias="CHECKPOINT_PATH"
    )

    model_arch: str = Field(default="densenet121", alias="MODEL_ARCH")
    class_names: str = Field(default="normal,pneumonia", alias="CLASS_NAMES")
    image_size: int = Field(default=224, alias="IMAGE_SIZE")
    default_threshold: float = Field(default=0.5, alias="DEFAULT_THRESHOLD")
    require_checkpoint: bool = Field(default=False, alias="REQUIRE_CHECKPOINT")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    @property
    def class_name_list(self) -> list[str]:
        return [name.strip() for name in self.class_names.split(",") if name.strip()]


settings = Settings()
