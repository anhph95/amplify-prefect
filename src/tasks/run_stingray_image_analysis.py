import os
import shlex
import tempfile
from pathlib import Path

import docker
from prefect import get_run_logger, task

from src.params.params_stingray_image_analysis import (
    FrameTimestampParams,
    ImageAbundanceParams,
    StingrayCruiseParams,
)


CONTAINER_CONFIG = "/run/cruise.conf.sh"
CONTAINER_STINGRAY_DATA = "/data/stingray"
CONTAINER_VIDEO_DATA = "/data/videos"
CONTAINER_WORKSPACE = "/app/workspace"


def _existing_path(value: str, description: str, directory: bool) -> Path:
    """Resolve a required host input before handing it to Docker."""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if directory and not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")
    return path


def _shell_value(value: object) -> str:
    """Quote one generated shell-config value without evaluating user input."""
    return shlex.quote(str(value))


def _shared_config(params: StingrayCruiseParams) -> list[str]:
    """Build paths shared by timestamp and abundance from Prefect parameters."""
    run_name = f"{params.cruise_date}_{params.cruise}"
    timestamp_mode = params.timestamp_mode.value
    media_list_dir = f"{CONTAINER_STINGRAY_DATA}/media_list/{params.camera_stream}"
    abundance_workspace = f"{CONTAINER_WORKSPACE}/abundance/{params.camera_stream}"
    values = {
        "CRUISE": params.cruise,
        "CRUISE_DATE": params.cruise_date,
        "CRUISE_COLLECTION": params.cruise_collection,
        "CAMERA_STREAM": params.camera_stream,
        "VIDEO_SUFFIX": params.video_suffix,
        "RUN_NAME": run_name,
        "TIMESTAMP_MODE": timestamp_mode,
        "CVISION_ENV": "/app/.venv/cvision",
        "VIDEO_DATA_ROOT": CONTAINER_VIDEO_DATA,
        "STINGRAY_DATA_ROOT": CONTAINER_STINGRAY_DATA,
        "VIDEO_INPUT_DIR": f"{CONTAINER_VIDEO_DATA}/{params.cruise_collection}_{params.cruise}/{params.camera_stream}",
        "MEDIA_LIST_DIR": media_list_dir,
        "ABUNDANCE_WORKSPACE_DIR": abundance_workspace,
        "VIDEO_LIST_CSV": f"{media_list_dir}/{run_name}_video_list_{timestamp_mode}.csv",
        "FRAME_LIST_CSV": f"{media_list_dir}/{run_name}_frame_list_{timestamp_mode}.csv",
        "DETECTIONS_CSV": f"{abundance_workspace}/{run_name}_detection_labels.csv",
        "CLASS_MAP_CSV": f"{abundance_workspace}/{run_name}_class_map.csv",
        "SENSOR_CSV": f"{CONTAINER_STINGRAY_DATA}/dashboard_data/data/{params.sensor_dataset}/{run_name}.csv",
        "ABUNDANCE_OUT_CSV": f"{CONTAINER_STINGRAY_DATA}/dashboard_data/data/{params.abundance_dataset}/{run_name}.csv",
    }
    return [f"{name}={_shell_value(value)}" for name, value in values.items()]


def _write_config(lines: list[str]) -> Path:
    """Create the short-lived config consumed by the existing shell job."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="stingray-image-analysis-",
        suffix=".conf.sh",
        delete=False,
    ) as config_file:
        config_file.write("\n".join(lines))
        config_file.write("\n")
        return Path(config_file.name)


def _run_container(
    image: str,
    script: str,
    config_lines: list[str],
    volumes: dict[str, dict[str, str]],
) -> None:
    """Run one canonical image-analysis shell job and stream its output."""
    logger = get_run_logger()
    config_path = _write_config(config_lines)
    volumes[str(config_path)] = {"bind": CONTAINER_CONFIG, "mode": "ro"}
    client = docker.from_env()

    logger.info("Running %s from %s", script, image)
    try:
        output = client.containers.run(
            image,
            ["bash", script, CONTAINER_CONFIG],
            volumes=volumes,
            working_dir="/app",
            user=f"{os.getuid()}:{os.getgid()}",
            remove=True,
            detach=False,
            stdout=True,
            stderr=True,
            stream=True,
        )
        for line in output:
            logger.info(line.decode("utf-8").rstrip())
    except docker.errors.ContainerError as error:
        message = error.stderr.decode("utf-8") if error.stderr else "No stderr"
        logger.error("Container failed: %s", message)
        raise RuntimeError(
            f"Docker container failed with exit code {error.exit_status}"
        ) from error
    finally:
        config_path.unlink(missing_ok=True)


@task(log_prints=True)
def run_frame_timestamps(
    cruise_params: StingrayCruiseParams,
    timestamp_params: FrameTimestampParams,
) -> None:
    """Generate shared video and frame lists from Prefect UI parameters."""
    video_root = _existing_path(timestamp_params.video_data_root, "Video data root", True)
    stingray_root = _existing_path(cruise_params.stingray_data_root, "Stingray data root", True)

    config_lines = _shared_config(cruise_params)
    timestamp_values = {
        "TIMESTAMP_FILE_LIMIT": timestamp_params.file_limit or "",
        "TIMESTAMP_MAX_WORKERS": timestamp_params.max_workers or "",
    }
    config_lines.extend(
        f"{name}={_shell_value(value)}" for name, value in timestamp_values.items()
    )
    config_lines.append(f"TIMESTAMP_SUFFIXES=({_shell_value(cruise_params.video_suffix)})")

    volumes = {
        str(video_root): {"bind": CONTAINER_VIDEO_DATA, "mode": "ro"},
        str(stingray_root): {"bind": CONTAINER_STINGRAY_DATA, "mode": "rw"},
    }
    _run_container(
        cruise_params.image,
        "frame_timestamps.sh",
        config_lines,
        volumes,
    )


@task(log_prints=True)
def run_image_abundance(
    cruise_params: StingrayCruiseParams,
    abundance_params: ImageAbundanceParams,
) -> None:
    """Merge detection labels and compute abundance from Prefect UI parameters."""
    stingray_root = _existing_path(cruise_params.stingray_data_root, "Stingray data root", True)
    class_yaml = _existing_path(abundance_params.class_yaml, "Class YAML", False)
    workspace = Path(abundance_params.workspace_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if abundance_params.merge_labels and not abundance_params.label_dirs:
        raise ValueError("label_dirs must contain at least one directory when merge_labels is enabled")

    volumes = {
        str(stingray_root): {"bind": CONTAINER_STINGRAY_DATA, "mode": "rw"},
        str(workspace): {"bind": CONTAINER_WORKSPACE, "mode": "rw"},
        str(class_yaml): {"bind": "/inputs/classes.yaml", "mode": "ro"},
    }
    container_label_dirs = []
    for index, label_dir_value in enumerate(abundance_params.label_dirs):
        label_dir = _existing_path(label_dir_value, "Label directory", True)
        container_path = f"/inputs/labels/{index}"
        volumes[str(label_dir)] = {"bind": container_path, "mode": "ro"}
        container_label_dirs.append(container_path)

    config_lines = _shared_config(cruise_params)
    abundance_values = {
        "CLASS_YAML": "/inputs/classes.yaml",
        "MERGE_LABELS": int(abundance_params.merge_labels),
        "SCORE_THRESH": abundance_params.score_thresh,
        "BIN_WIDTH": abundance_params.bin_width,
        "VOLUME_PER_FRAME": abundance_params.volume_per_frame,
        "ADD_CI": int(abundance_params.add_ci),
        "JOBS": abundance_params.jobs or "",
    }
    config_lines.extend(
        f"{name}={_shell_value(value)}" for name, value in abundance_values.items()
    )
    labels = " ".join(_shell_value(path) for path in container_label_dirs)
    config_lines.append(f"LABEL_DIRS=({labels})")

    _run_container(
        cruise_params.image,
        "image_abundance.sh",
        config_lines,
        volumes,
    )
