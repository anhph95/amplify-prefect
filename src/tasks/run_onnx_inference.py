import os

import docker
from prefect import get_run_logger
from prefect import task

from src.params.params_onnx import ONNXInferenceParams

DEFAULT_SCORE_OUTFILE = "{MODEL_NAME}/{SUBPATH}/{BIN}.csv"
DEFAULT_EMBEDDINGS_OUTFILE = "{MODEL_NAME}/{SUBPATH}/{BIN}.emb.parquet"
RUN_DATE_SCORE_OUTFILE = "{RUN_DATE}/{SUBPATH}/{BIN}.csv"
RUN_DATE_EMBEDDINGS_OUTFILE = "{RUN_DATE}/{SUBPATH}/{BIN}.emb.parquet"


def _resolve_outfile_pattern(onnx_inference_params: ONNXInferenceParams) -> str:
    if onnx_inference_params.outfile is not None:
        return onnx_inference_params.outfile
    if onnx_inference_params.subfolder_type == "run-date":
        return RUN_DATE_SCORE_OUTFILE
    return DEFAULT_SCORE_OUTFILE


def _resolve_embeddings_outfile_pattern(
    onnx_inference_params: ONNXInferenceParams,
) -> str:
    if onnx_inference_params.embeddings_outfile is not None:
        return onnx_inference_params.embeddings_outfile
    if onnx_inference_params.subfolder_type == "run-date":
        return RUN_DATE_EMBEDDINGS_OUTFILE
    return DEFAULT_EMBEDDINGS_OUTFILE


def _build_command_args(
    onnx_inference_params: ONNXInferenceParams, model_container_path: str
) -> list[str]:
    command_args: list[str] = []

    if onnx_inference_params.batch is not None:
        command_args.extend(["--batch", str(onnx_inference_params.batch)])
    if onnx_inference_params.classes is not None:
        command_args.extend(["--classes", "/app/classes.txt"])

    command_args.extend(["--outdir", "/app/outputs"])

    outfile_pattern = _resolve_outfile_pattern(onnx_inference_params)
    if outfile_pattern != DEFAULT_SCORE_OUTFILE:
        command_args.extend(["--outfile", outfile_pattern])

    emit_embeddings = (
        onnx_inference_params.embeddings or onnx_inference_params.embeddings_only
    )
    if emit_embeddings:
        command_args.append("--embeddings")
        embeddings_pattern = _resolve_embeddings_outfile_pattern(onnx_inference_params)
        if embeddings_pattern != DEFAULT_EMBEDDINGS_OUTFILE:
            command_args.extend(["--embeddings-outfile", embeddings_pattern])
        if onnx_inference_params.embeddings_only:
            command_args.append("--embeddings-only")

    if onnx_inference_params.ensure_softmax is False:
        command_args.append("--skip-ensure-softmax")

    if onnx_inference_params.force_notorch:
        command_args.append("--notorch")
    if onnx_inference_params.cpuonly:
        command_args.append("--cpuonly")

    command_args.extend([model_container_path, "/app/inputs"])
    return command_args


@task(log_prints=True)
def run_onnx_inference(onnx_inference_params: ONNXInferenceParams, onnx_image: str):
    """
    Run inference with an ONNX model in a Docker container.
    """

    client = docker.from_env()
    logger = get_run_logger()

    model_filename = os.path.basename(onnx_inference_params.model)
    model_container_path = f"/app/models/{model_filename}"

    volumes = {
        onnx_inference_params.model: {"bind": model_container_path, "mode": "ro"},
        onnx_inference_params.input_dir: {"bind": "/app/inputs", "mode": "rw"},
        onnx_inference_params.output_dir: {"bind": "/app/outputs", "mode": "rw"},
    }

    if onnx_inference_params.classes is not None:
        volumes[onnx_inference_params.classes] = {
            "bind": "/app/classes.txt",
            "mode": "ro",
        }

    environment = {"CUDA_VISIBLE_DEVICES": onnx_inference_params.cuda_visible_devices}
    command_args = _build_command_args(onnx_inference_params, model_container_path)

    logger.info("Running container with command: %s", " ".join(command_args))

    try:
        container_kwargs = {
            "volumes": volumes,
            "environment": environment,
            "ipc_mode": "host",
            "remove": True,
            "detach": False,
            "stdout": True,
            "stderr": True,
            "stream": True,
        }
        if not onnx_inference_params.cpuonly:
            container_kwargs["device_requests"] = [
                docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])
            ]

        container = client.containers.run(onnx_image, command_args, **container_kwargs)

        for line in container:
            logger.info(line.decode("utf-8").rstrip())

    except docker.errors.ContainerError as e:
        logger.error(
            "Container failed with stderr: %s",
            e.stderr.decode("utf-8") if e.stderr else "No stderr",
        )
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error("Unexpected error running container: %s", str(e))
        raise
