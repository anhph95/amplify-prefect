# AMPLIfy Prefect Pipeline

A Prefect server for orchestrating machine learning training and inference runs. Supports YOLO (training and inference), SegGPT (inference), ONNX models (inference), and IFCB flow metric (training). After setting up the system with Docker, users can monitor and run the workflows using the UI accessible in a browser.

## Setup

### 1. Environment Configuration

Copy the example environment file and fill in your values:
```bash
cp .env.example .env
```

Edit `.env` with your specific values:
- `POSTGRES_USERNAME`: Your PostgreSQL username
- `POSTGRES_PASSWORD`: Your PostgreSQL password  
- `EXTERNAL_HOST_NAME`: External hostname of your machine
- `PROVENANCE_STORE_URL`: URL for provenance store
- `MEDIASTORE_URL`: URL for your media store
- `MEDIASTORE_TOKEN`: Authentication token for media store

### 2. Launch PostgreSQL Database

Use Docker Compose to start the PostgreSQL container:
```bash
docker compose up -d postgres
```

### 3. Python Environment Setup

Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

### 4. Configure and Start Prefect Server

Load environment variables and configure Prefect:
```bash
# Load environment variables
source .env

# Set Prefect configuration
prefect config set PREFECT_SERVER_API_HOST="$EXTERNAL_HOST_NAME"
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://$POSTGRES_USERNAME:$POSTGRES_PASSWORD@localhost:5432/prefect"

# Start Prefect server
prefect server start
```

### 5. Deploy Workflows

In separate terminal windows, deploy the workflows you want to use:

**For ONNX Inference:**
```bash
source .venv/bin/activate
source .env
python src/flows/onnx_inference.py
```

**For YOLO Inference:**
```bash
source .venv/bin/activate  
source .env
python src/flows/yolo_inference.py
```

**For IFCB Flow Metric Training:**
```bash
source .venv/bin/activate
source .env
python src/flows/ifcb_training.py
```

**For Slim Feature Extraction:**
```bash
source .venv/bin/activate
source .env
python src/flows/extract_slim_features.py
```

**For Stingray Frame Timestamps and Image Abundance:**

```bash
source .venv/bin/activate
source .env
# Run each deployment in its own terminal.
python src/flows/stingray_frame_timestamps.py
python src/flows/stingray_image_abundance.py
```

## Using the Workflows

Navigate to the Prefect UI in your browser at `http://{EXTERNAL_HOST_NAME}:4200`

### ONNX Inference Workflow

The ONNX inference workflow requires the following parameters in the Prefect UI. It runs `ghcr.io/whoigit/ifcb-inference:latest` by default, and switches to `ghcr.io/whoigit/ifcb-inference-embeddings:latest` when embeddings are enabled or score output uses `.h5`/`.parquet`.

**ONNXInferenceParams:**
- `model`: Path to the ONNX model file
- `input_dir`: Directory containing input data
- `output_dir`: Directory where results will be saved
- `batch` (optional): Batch size for inference
- `classes` (optional): Specific classes to process
- `outfile` (optional): Custom output filename
- `subfolder_type` (optional): Choose `model-name` or `run-date` output layout
- `force_notorch` (optional): Force non-PyTorch backend
- `cpuonly` (optional): Force CPU-only inference (default: false)
- `cuda_visible_devices`: GPU devices to use (default: "0,1,2,3")
- `ensure_softmax` (optional): Ensure softmax is applied to model output (default: true)
- `embeddings` (optional): Write penultimate-layer embeddings alongside scores
- `embeddings_only` (optional): Skip the score CSV and write only embeddings
- `embeddings_outfile` (optional): Custom embeddings filename

The embeddings options require an ONNX model that has already been modified to expose the embedding tensor as a second graph output.

### YOLO Inference Workflow

The YOLO inference workflow requires two parameter sets:

**YOLOInferenceParams:**
- `data_dir`: Directory containing input images/videos
- `output_dir`: Directory where results will be saved
- `model_weights_path`: Path to YOLO model weights (.pt file)
- `device`: Compute device for inference (e.g., "0" for GPU 0, "cpu")
- `agnostic_nms`: Class-agnostic Non-Maximum Suppression (default: true)
- `iou`: IoU threshold for NMS to eliminate overlapping boxes (default: 0.5)
- `conf`: Minimum confidence threshold for detections (default: 0.1)
- `imgsz`: Image size for inference (default: 1280)
- `batch`: Batch size for processing multiple inputs (default: 16)
- `half`: Half-precision (FP16) inference for speed (default: false)
- `max_det`: Maximum detections allowed per image (default: 300)
- `vid_stride`: Frame stride for video processing (default: 1)
- `stream_buffer`: Queue frames vs drop old frames (default: false)
- `visualize`: Visualize model features during inference (default: false)
- `augment`: Test-time augmentation for improved robustness (default: false)
- `classes`: Filter predictions to specific class IDs (optional)
- `retina_masks`: High-resolution segmentation masks (default: false)
- `embed`: Extract feature vectors from specified layers (optional)
- `name`: Name for prediction run subdirectory (optional)
- `verbose`: Display detailed inference logs (default: true)
- `ext`: File extension to scan for inference inputs (default: ".avi")
- `max_files`: Maximum number of discovered files to process (optional)
- `skip_validation`: Skip OpenCV validation before inference (default: false)
- `validation_workers`: Parallel OpenCV validation workers; 0 chooses an automatic worker count (default: 0)

**YOLOVisualizationParams:**
- `show`: Display annotated images/videos in window (default: false)
- `save`: Save annotated images/videos to file (default: false)
- `save_frames`: Save individual video frames as images (default: false)
- `save_txt`: Save detection results in text format (default: false)
- `save_conf`: Include confidence scores in saved text files (default: false)
- `save_crop`: Save cropped images of detections (default: false)
- `show_labels`: Display labels for each detection (default: true)
- `show_conf`: Display confidence scores alongside labels (default: true)
- `show_boxes`: Draw bounding boxes around detected objects (default: true)

### IFCB Flow Metric Training Workflow

The IFCB flow metric training workflow requires the following parameters in the Prefect UI:

**IFCBTrainingParams:**
- `data_dir`: Directory containing IFCB point cloud data
- `output_dir`: Directory where trained model will be saved
- `id_file` (optional): File containing list of IDs to load (one PID per line)
- `n_jobs`: Number of parallel jobs for load/extraction phase (-1 uses all CPUs, default: -1)
- `contamination`: Expected fraction of anomalous distributions (default: 0.1)
- `aspect_ratio`: Camera frame aspect ratio (width/height, default: 1.36)
- `chunk_size`: Number of PIDs to process in each chunk (default: 100)
- `model_filename`: Filename for the trained model (default: "classifier.pkl")

### Slim Feature Extraction Workflow

The slim feature extraction workflow supports two extraction sources:

- `storage`: preserves the existing S3/VastDB-capable flow. This mode uses `extract_features_image` and supports blob storage, feature storage, and GPU batch-processing parameters.
- `main`: runs `ghcr.io/whoigit/ifcb-features:latest`. This mode writes local `<bin>_features_v4.csv` and `<bin>_blobs_v4.zip` files in `output_directory` and only passes `data_directory`, `output_directory`, and optional `bins` to the container.

**ExtractSlimFeaturesParams:**
- `extract_features_source`: Choose `storage` or `main` (default: `storage`)
- `extract_features_image`: Docker image for `storage` mode; ignored in `main` mode
- `data_directory`: Directory containing IFCB data
- `output_directory`: Directory where local outputs are written
- `bins` (optional): Specific bin names to process
- `blob_storage_mode`: Blob storage mode for `storage` source, either `local` or `s3`
- `s3_bucket`, `s3_url`, `s3_prefix`: S3 settings for `storage` source
- `feature_storage_mode`: Feature storage mode for `storage` source, either `local` or `vastdb`
- `vastdb_bucket`, `vastdb_schema`, `vastdb_table`, `vastdb_url`: VastDB settings for `storage` source
- `batch_processing`, `min_batch_size`, `max_batch_size`, `gpu_device`: GPU batch-processing settings for `storage` source

### Stingray Image Analysis Workflows

The timestamp and abundance deployments accept their cruise settings directly in
the Prefect UI. Prefect creates the container configuration for the run; users do
not need to place a cruise configuration file on the host or in the image-analysis
repository.

**StingrayCruiseParams:**

- `cruise`, `cruise_date`, `cruise_collection`, `camera_stream`: identify the cruise media
- `stingray_data_root`: host root containing media lists and dashboard data
- `video_suffix`, `timestamp_mode`, `sensor_dataset`, `abundance_dataset`: shared artifact settings
- `image`: image-analysis container image

**FrameTimestampParams:**

- `video_data_root`: host root containing cruise video directories
- `file_limit`: optional test limit; empty scans all videos
- `max_workers`: optional worker count; empty uses all available CPUs

**ImageAbundanceParams:**

- `workspace_dir`: host directory for intermediate image-abundance products
- `class_yaml`: host model class-name YAML
- `label_dirs`: host prediction-label directories to merge
- `merge_labels`: merge labels before abundance processing
- `score_thresh`, `bin_width`, `volume_per_frame`, `add_ci`: abundance controls
- `jobs`: optional label-conversion worker count

## YOLO Training Data Format

For YOLO training, the data directory should contain a `dataset.yaml` file:

```yaml
path: /data # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path') 
test: images/test # test images (optional)

names:
  0: person
  1: bicycle
  2: car
```

Ensure your directory structure matches:
```
data_dir/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/ (optional)
└── labels/
    ├── train/
    ├── val/
    └── test/ (optional)
``` 
