from pydantic import BaseModel, Field
from typing import Optional, Union
from enum import Enum


class BinType(str, Enum):
    I_BINS = "I"
    D_BINS = "D"


# Feature selection configuration for IFCB flow metric training
class IFCBFeatureConfig(BaseModel):
    # Spatial Statistics Features
    use_mean_x: bool = Field(True, description="Use mean x-coordinate feature")
    use_mean_y: bool = Field(True, description="Use mean y-coordinate feature")
    use_std_x: bool = Field(True, description="Use standard deviation of x-coordinates feature")
    use_std_y: bool = Field(True, description="Use standard deviation of y-coordinates feature")
    use_median_x: bool = Field(True, description="Use median x-coordinate feature")
    use_median_y: bool = Field(True, description="Use median y-coordinate feature")
    use_iqr_x: bool = Field(True, description="Use interquartile range of x-coordinates feature")
    use_iqr_y: bool = Field(True, description="Use interquartile range of y-coordinates feature")
    
    # Distribution Shape Features
    use_ratio_spread: bool = Field(True, description="Use ratio of y-direction IQR to x-direction IQR feature")
    use_core_fraction: bool = Field(True, description="Use fraction of points within IQR box feature")
    
    # Clipping Detection Features
    use_duplicate_fraction: bool = Field(True, description="Use fraction of points sharing same y-value feature")
    use_max_duplicate_fraction: bool = Field(True, description="Use largest fraction of points with identical y-values feature")
    
    # Histogram Uniformity Features
    use_cv_x: bool = Field(True, description="Use coefficient of variation for x-coordinate histogram feature")
    use_cv_y: bool = Field(True, description="Use coefficient of variation for y-coordinate histogram feature")
    
    # Statistical Moments Features
    use_skew_x: bool = Field(True, description="Use skewness of x-coordinate distribution feature")
    use_skew_y: bool = Field(True, description="Use skewness of y-coordinate distribution feature")
    use_kurt_x: bool = Field(True, description="Use kurtosis of x-coordinate distribution feature")
    use_kurt_y: bool = Field(True, description="Use kurtosis of y-coordinate distribution feature")
    
    # PCA Orientation Features
    use_angle: bool = Field(True, description="Use principal component angle feature")
    use_eigen_ratio: bool = Field(True, description="Use ratio of first to second eigenvalue feature")
    
    # Edge Features
    use_left_edge_fraction: bool = Field(True, description="Use fraction of points near left frame edge feature")
    use_right_edge_fraction: bool = Field(True, description="Use fraction of points near right frame edge feature")
    use_top_edge_fraction: bool = Field(True, description="Use fraction of points near top frame edge feature")
    use_bottom_edge_fraction: bool = Field(True, description="Use fraction of points near bottom frame edge feature")
    use_total_edge_fraction: bool = Field(True, description="Use total fraction of points near any frame edge feature")
    
    # Temporal Features
    use_t_y_var: bool = Field(True, description="Use variance of rolling mean of y-coordinates feature")

    # Trigger Statistics Features
    # Only exists in images built from code that defines roi_trigger_fraction; on
    # images without it the emitted config key is ignored. Must be emitted
    # explicitly to disable, since the extractor treats an absent key as enabled.
    use_roi_trigger_fraction: bool = Field(True, description="Use fraction of trigger events that yielded a detected ROI feature")

# Parameters relevant to IFCB flow metric training
class IFCBTrainingParams(BaseModel):
    data_dir: str = Field(..., description="Directory containing IFCB point cloud data")
    output_dir: str = Field(..., description="Directory where trained model will be saved")
    id_file: Optional[str] = Field(None, description="File containing list of IDs to load (one PID per line)")
    n_jobs: int = Field(-1, description="Number of parallel jobs for load/extraction phase (-1 uses all CPUs)")
    contamination: float = Field(0.1, description="Expected fraction of anomalous distributions")
    aspect_ratio: float = Field(1.36, description="Camera frame aspect ratio (width/height)")
    chunk_size: int = Field(100, description="Number of PIDs to process in each chunk")
    model_filename: str = Field("classifier.pkl", description="Filename for the trained model")
    max_samples: Union[int, float, str] = Field("auto", description="Number of samples to draw from X to train each base estimator")
    max_features: Union[int, float] = Field(1.0, description="Number of features to draw from X to train each base estimator")
    bin_type: BinType = Field(..., description="Type of bins to train on: 'I' for I-bins or 'D' for D-bins")

    validate_bin_paths: bool = Field(
        True,
        description=(
            "Only use bins located at {year}/{day}/{pid}.adc with the directories "
            "agreeing with the PID's own date. Excludes calibration and scratch "
            "trees (beads, temp, data_temp, skip), whose files have valid PIDs, and "
            "excludes duplicate copies at nested paths. Set False for the previous "
            "behaviour of taking every .adc found at any depth."
        ),
    )

    # Docker image
    ifcb_image: str = Field(
        "ghcr.io/whoigit/ifcb-flow-metric:main",
        description=(
            "Docker image used for training. Determines which features exist; the "
            "trained model records its own feature names, so scoring follows it."
        ),
    )

    # Feature selection configuration
    feature_config: IFCBFeatureConfig = Field(default_factory=IFCBFeatureConfig, description="Feature selection configuration")


# Parameters relevant to IFCB flow metric inference/scoring
class IFCBInferenceParams(BaseModel):
    data_dir: str = Field(..., description="Directory containing IFCB point cloud data")
    output_dir: str = Field(..., description="Directory where inference results will be saved")
    model_path: str = Field(..., description="Path to the trained model file")
    id_file: Optional[str] = Field(None, description="File containing list of IDs to score (one PID per line)")
    n_jobs: int = Field(-1, description="Number of parallel jobs for load/extraction phase (-1 uses all CPUs)")
    aspect_ratio: float = Field(1.36, description="Camera frame aspect ratio (width/height)")
    chunk_size: int = Field(100, description="Number of PIDs to process in each chunk")
    output_filename: str = Field("scores.csv", description="Filename for the output CSV file")


# Parameters relevant to IFCB flow metric evaluation
class IFCBEvaluationParams(BaseModel):
    csv1_path: str = Field(..., description="Path to first CSV file with anomaly scores")
    csv2_path: str = Field(..., description="Path to second CSV file with anomaly scores")
    output_dir: str = Field(..., description="Directory where the violin plot will be saved")
    output_filename: str = Field("violin_plot.png", description="Filename for the output violin plot")
    title: str = Field("Score Distributions by Category (Violin Plot)", description="Title for the violin plot")
    name1: str = Field("Dataset 1", description="Name for first distribution")
    name2: str = Field("Dataset 2", description="Name for second distribution")


# Parameters for full IFCB flow metric evaluation (bad vs normal data)
class IFCBFullEvaluationParams(BaseModel):
    bad_i_data_dir: str = Field(..., description="Directory containing subdirectories of known bad I bin data")
    bad_d_data_dir: str = Field(..., description="Directory containing subdirectories of known bad D bin data")
    normal_data_dir: str = Field(..., description="Directory containing normal/unknown IFCB bin data (mixed I and D)")
    i_model_path: str = Field(..., description="Path to the trained IFCB model file for I bins")
    d_model_path: str = Field(..., description="Path to the trained IFCB model file for D bins")
    output_dir: str = Field(..., description="Directory where all evaluation outputs will be saved")

    validate_bin_paths: bool = Field(
        True,
        description=(
            "Only use normal-data bins located at {year}/{day}/{pid}.adc with the "
            "directories agreeing with the PID's own date. Excludes calibration and "
            "scratch trees (beads, temp, data_temp, skip), whose files have valid "
            "PIDs, and excludes duplicate copies at nested paths. Set False for the "
            "previous behaviour of taking every .adc found at any depth."
        ),
    )

    # Docker image
    ifcb_image: str = Field(
        "ghcr.io/whoigit/ifcb-flow-metric:main",
        description=(
            "Docker image used for inference and evaluation. Must be built from code "
            "that computes the same features the models were trained on, since "
            "scikit-learn validates feature count but not feature identity."
        ),
    )

    # Optional parameters for inference
    n_jobs: int = Field(-1, description="Number of parallel jobs for feature extraction")
    aspect_ratio: float = Field(1.36, description="Camera frame aspect ratio")
    chunk_size: int = Field(100, description="Number of PIDs to process in each chunk")
    
    # Optional parameters for visualization
    plot_title_prefix: str = Field("Anomaly Score Distribution", description="First part of the violin plot title")
    normal_data_name: str = Field("Not Known Bad", description="Label for normal data in plot")
