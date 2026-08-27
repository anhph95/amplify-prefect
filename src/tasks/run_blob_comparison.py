from prefect import task
import docker
import os

from prefect import get_run_logger
from prefect_aws import AwsCredentials

from src.params.params_feature_validation import FeatureValidationParams


@task(log_prints=True)
def run_blob_comparison(validation_params: FeatureValidationParams):
    """
    Run IFCB blob comparison in a Docker container.

    Compares predicted blobs against ground truth blobs pixel-by-pixel and creates
    Prefect artifacts with comparison results including visualizations.
    """

    client = docker.from_env()
    logger = get_run_logger()

    # Skip if blob comparison is disabled
    if not validation_params.enable_blob_comparison:
        logger.info("Blob comparison is disabled, skipping")
        return

    # Load AWS credentials from Prefect block
    aws_credentials = AwsCredentials.load("vasts3-creds")

    # Set up volumes - mount output directory
    blob_output_dir = os.path.join(validation_params.output_directory, "blob_comparison")
    os.makedirs(blob_output_dir, exist_ok=True)

    volumes = {
        blob_output_dir: {'bind': '/app/blob_comparison_output', 'mode': 'rw'}
    }

    # Set up environment variables for S3 credentials
    environment = {
        'AWS_ACCESS_KEY_ID': aws_credentials.aws_access_key_id,
        'AWS_SECRET_ACCESS_KEY': aws_credentials.aws_secret_access_key.get_secret_value(),
        'PYTHONUNBUFFERED': '1',
    }

    # Build command arguments
    command_args = [
        "compare_blobs.py",
        "--pred-bucket", validation_params.blob_pred_bucket,
        "--gt-bucket", validation_params.blob_gt_bucket,
        "--s3-url", validation_params.blob_s3_url,
        "--pred-prefix", validation_params.blob_pred_prefix,
        "--gt-prefix", validation_params.blob_gt_prefix,
        "--output-dir", "/app/blob_comparison_output",
        "--top-n-worst", str(validation_params.blob_top_n_worst),
    ]

    # Add optional sample IDs filter
    if validation_params.sample_ids:
        command_args.extend(["--sample-ids"] + validation_params.sample_ids)

    logger.info(f'Running blob comparison container with command: {" ".join(command_args)}')

    try:
        # Get current user's UID and GID to ensure output files are owned by the user
        uid = os.getuid()
        gid = os.getgid()

        # Run container in detached mode to stream logs properly
        container = client.containers.run(
            validation_params.validation_image,
            command_args,
            volumes=volumes,
            environment=environment,
            user=f"{uid}:{gid}",
            detach=True
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

        logger.info("✓ Blob comparison completed successfully")

    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise
