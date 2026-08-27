from prefect import task
import docker
import os

from prefect import get_run_logger

from src.params.params_extract_slim_features import ExtractSlimFeaturesParams, SlimFeaturesSource


MAIN_SLIM_FEATURES_IMAGE = "ghcr.io/whoigit/ifcb-features:latest"


def resolve_extract_slim_features_image(extract_features_params: ExtractSlimFeaturesParams) -> str:
    if extract_features_params.extract_features_source == SlimFeaturesSource.main:
        return MAIN_SLIM_FEATURES_IMAGE

    if not extract_features_params.extract_features_image:
        raise ValueError("extract_features_image is required when extract_features_source='storage'")

    return extract_features_params.extract_features_image


def build_extract_slim_features_command(extract_features_params: ExtractSlimFeaturesParams) -> list[str]:
    command_args = [
        "/app/data",
        "/app/output"
    ]

    if extract_features_params.extract_features_source == SlimFeaturesSource.main:
        command_args.extend(["--workers", str(extract_features_params.workers)])
        if extract_features_params.bins:
            command_args.extend(["--bins"] + extract_features_params.bins)
        return command_args

    # Add blob storage arguments
    command_args.extend([
        "--blob-storage-mode", extract_features_params.blob_storage_mode
    ])

    if extract_features_params.blob_storage_mode == "s3":
        if not extract_features_params.s3_bucket or not extract_features_params.s3_url:
            raise ValueError("s3_bucket and s3_url are required when blob_storage_mode='s3'")

        command_args.extend([
            "--s3-bucket", extract_features_params.s3_bucket,
            "--s3-url", extract_features_params.s3_url,
            "--s3-prefix", extract_features_params.s3_prefix
        ])

    # Add feature storage arguments
    command_args.extend([
        "--feature-storage-mode", extract_features_params.feature_storage_mode
    ])

    if extract_features_params.feature_storage_mode == "vastdb":
        vastdb_url = extract_features_params.vastdb_url or extract_features_params.s3_url
        if (
            not extract_features_params.vastdb_bucket
            or not extract_features_params.vastdb_schema
            or not extract_features_params.vastdb_table
            or not vastdb_url
        ):
            raise ValueError(
                "vastdb_bucket, vastdb_schema, vastdb_table, and vastdb_url or s3_url "
                "are required when feature_storage_mode='vastdb'"
            )

        command_args.extend([
            "--vastdb-bucket", extract_features_params.vastdb_bucket,
            "--vastdb-schema", extract_features_params.vastdb_schema,
            "--vastdb-table", extract_features_params.vastdb_table,
            "--vastdb-url", vastdb_url
        ])

    # Add optional bins flag if provided
    if extract_features_params.bins:
        command_args.extend(["--bins"] + extract_features_params.bins)

    # Add GPU batch processing arguments if enabled
    if (
        extract_features_params.extract_features_source == SlimFeaturesSource.storage
        and extract_features_params.batch_processing
    ):
        command_args.extend([
            "--batch-processing",
            "--min-batch-size", str(extract_features_params.min_batch_size),
            "--max-batch-size", str(extract_features_params.max_batch_size)
        ])
        if extract_features_params.gpu_device is not None:
            command_args.extend(["--gpu-device", str(extract_features_params.gpu_device)])

    return command_args


@task(log_prints=True)
def run_extract_slim_features(extract_features_params: ExtractSlimFeaturesParams):
    """
    Run IFCB feature extraction in a Docker container.

    The main image runs extract_features_batch.py, which writes the MATLAB-compatible
    day-based layout under the output directory; the storage image is unchanged.
    """

    client = docker.from_env()
    logger = get_run_logger()
    extract_features_image = resolve_extract_slim_features_image(extract_features_params)

    # Set up volumes
    volumes = {
        extract_features_params.data_directory: {'bind': '/app/data', 'mode': 'ro'},
        extract_features_params.output_directory: {'bind': '/app/output', 'mode': 'rw'}
    }

    # Set up environment variables
    environment = {
        'PYTHONUNBUFFERED': '1',  # Disable Python output buffering for real-time logs
    }

    if extract_features_params.extract_features_source == SlimFeaturesSource.storage:
        from prefect_aws import AwsCredentials

        # Load AWS credentials from Prefect block for the storage-capable image.
        aws_credentials = AwsCredentials.load("vasts3-creds")
        environment.update({
            'AWS_ACCESS_KEY_ID': aws_credentials.aws_access_key_id,
            'AWS_SECRET_ACCESS_KEY': aws_credentials.aws_secret_access_key.get_secret_value(),
        })

    # Build command arguments; each image's ENTRYPOINT already supplies the script
    # (extract_features_batch.py for the main image).
    command_args = build_extract_slim_features_command(extract_features_params)

    logger.info(f'Running container with command: {" ".join(command_args)}')
    logger.info(f'Using Docker image: {extract_features_image}')

    # Configure GPU device requests if batch processing is enabled
    device_requests = []
    if (
        extract_features_params.extract_features_source == SlimFeaturesSource.storage
        and extract_features_params.batch_processing
    ):
        if extract_features_params.gpu_device is not None:
            # Request specific GPU device
            device_requests = [docker.types.DeviceRequest(
                device_ids=[str(extract_features_params.gpu_device)],
                capabilities=[["gpu"]]
            )]
        else:
            # Request all available GPUs
            device_requests = [docker.types.DeviceRequest(
                device_ids=["all"],
                capabilities=[["gpu"]]
            )]
        logger.info(f"GPU device requests configured: {device_requests}")

    try:
        # Get current user's UID and GID to ensure output files are owned by the user
        uid = os.getuid()
        gid = os.getgid()

        # Run container in detached mode to stream logs properly
        container = client.containers.run(
            extract_features_image,
            command_args,
            volumes=volumes,
            environment=environment,
            user=f"{uid}:{gid}",
            device_requests=device_requests if device_requests else None,
            detach=True  # Run detached and stream logs separately
        )

        # Stream logs in real-time
        try:
            for log_line in container.logs(stream=True, follow=True):
                logger.info(log_line.decode('utf-8').rstrip())

            # Wait for container to finish
            result = container.wait()
            exit_code = result['StatusCode']

            if exit_code != 0:
                raise RuntimeError(f"Docker container failed with exit code {exit_code}")

        finally:
            # Clean up container
            try:
                container.remove()
            except:
                pass
            
    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise
