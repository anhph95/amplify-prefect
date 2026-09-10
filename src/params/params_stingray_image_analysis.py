from enum import Enum

from pydantic import BaseModel, Field


DEFAULT_IMAGE = "ghcr.io/whoigit/stingray-image-analysis:latest"


class TimestampMode(str, Enum):
    fast = "fast"
    details = "details"


class StingrayCruiseParams(BaseModel):
    cruise: str = Field(..., description="Cruise name, for example HRS2609")
    cruise_date: str = Field(..., description="Cruise media date in YYYYMMDD format")
    cruise_collection: str = Field("NESLTER", description="Collection prefix used by the cruise media directory")
    camera_stream: str = Field(..., description="Camera stream directory name")
    stingray_data_root: str = Field(..., description="Host Stingray data directory containing media lists and dashboard data")
    video_suffix: str = Field(".avi", description="Video filename suffix")
    timestamp_mode: TimestampMode = Field(TimestampMode.fast, description="Timestamp mode used in the shared video and frame-list filenames")
    sensor_dataset: str = Field("stingray_NESLTER", description="Dashboard sensor dataset directory")
    abundance_dataset: str = Field("shadowgraph", description="Dashboard image-abundance dataset directory")
    image: str = Field(DEFAULT_IMAGE, description="Image-analysis container image")


class FrameTimestampParams(BaseModel):
    video_data_root: str = Field(..., description="Host directory containing cruise video collections")
    file_limit: int | None = Field(None, gt=0, description="Maximum videos to scan; empty scans every video")
    max_workers: int | None = Field(None, gt=0, description="Parallel timestamp workers; empty uses the visible CPU count minus one")


class ImageAbundanceParams(BaseModel):
    workspace_dir: str = Field(..., description="Host workspace for intermediate image-abundance products")
    class_yaml: str = Field(..., description="Host path to the model class-name YAML file")
    label_dirs: list[str] = Field(default_factory=list, description="Host directories containing prediction labels")
    merge_labels: bool = Field(True, description="Merge YOLO label files before computing abundance")
    score_thresh: float = Field(0.7, ge=0.0, le=1.0, description="Minimum detection confidence")
    bin_width: int = Field(5, gt=0, description="Abundance time-bin width")
    volume_per_frame: float = Field(0.00225, gt=0.0, description="Sample volume represented by one frame")
    add_ci: bool = Field(False, description="Add confidence intervals to abundance output")
    jobs: int | None = Field(None, gt=0, description="Parallel label-conversion jobs; empty uses the visible CPU count minus one")
