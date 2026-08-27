from prefect import task
import docker
import os

from prefect import get_run_logger

from src.params.params_ifcb_flow_metric import IFCBInferenceParams


@task(log_prints=True)
def run_ifcb_flow_metric_inference(ifcb_inference_params: IFCBInferenceParams, ifcb_image: str):
    """
    Run IFCB flow metric inference/scoring in a Docker container.
    """
    
    client = docker.from_env()
    logger = get_run_logger()
    
    # Set up volumes
    volumes = {
        ifcb_inference_params.data_dir: {'bind': '/app/data', 'mode': 'ro'},
        ifcb_inference_params.output_dir: {'bind': '/app/output', 'mode': 'rw'},
        ifcb_inference_params.model_path: {'bind': '/app/model.pkl', 'mode': 'ro'}
    }
    
    # Mount id_file if provided
    id_file_container_path = None
    if ifcb_inference_params.id_file is not None:
        id_file_container_path = '/app/ids.txt'
        volumes[ifcb_inference_params.id_file] = {'bind': id_file_container_path, 'mode': 'ro'}
    
    # Build command arguments
    command_args = [
        "python", "score.py",
        "/app/data",
        "--n-jobs", str(ifcb_inference_params.n_jobs),
        "--aspect-ratio", str(ifcb_inference_params.aspect_ratio),
        "--chunk-size", str(ifcb_inference_params.chunk_size),
        "--model", "/app/model.pkl",
        "--output", f"/app/output/{ifcb_inference_params.output_filename}"
    ]
    
    # Add optional id-file flag if provided
    if ifcb_inference_params.id_file is not None:
        command_args.extend(["--id-file", id_file_container_path])
    
    logger.info(f'Running container with command: {" ".join(command_args)}')
    
    try:
        # Get current user's UID and GID to ensure output files are owned by the user
        uid = os.getuid()
        gid = os.getgid()
        
        container = client.containers.run(
            ifcb_image,
            command_args,
            volumes=volumes,
            user=f"{uid}:{gid}",
            remove=True,
            detach=False,
            stdout=True,
            stderr=True,
            stream=True
        )
        
        # Stream output
        for line in container:
            logger.info(line.decode('utf-8').rstrip())
            
    except docker.errors.ContainerError as e:
        logger.error(f"Container failed with stderr: {e.stderr.decode('utf-8') if e.stderr else 'No stderr'}")
        raise RuntimeError(f"Docker container failed with exit code {e.exit_status}")
    except Exception as e:
        logger.error(f"Unexpected error running container: {str(e)}")
        raise
