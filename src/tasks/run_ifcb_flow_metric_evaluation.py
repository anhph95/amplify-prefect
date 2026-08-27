from prefect import task
import docker
import os

from prefect import get_run_logger

from src.params.params_ifcb_flow_metric import IFCBEvaluationParams


@task(log_prints=True)
def run_ifcb_flow_metric_evaluation(ifcb_evaluation_params: IFCBEvaluationParams, ifcb_image: str):
    """
    Run IFCB flow metric evaluation by creating a violin plot comparing two score distributions.
    """
    
    client = docker.from_env()
    logger = get_run_logger()
    
    # Set up volumes
    volumes = {
        ifcb_evaluation_params.csv1_path: {'bind': '/app/csv1.csv', 'mode': 'ro'},
        ifcb_evaluation_params.csv2_path: {'bind': '/app/csv2.csv', 'mode': 'ro'},
        ifcb_evaluation_params.output_dir: {'bind': '/app/output', 'mode': 'rw'}
    }
    
    # Build command arguments
    command_args = [
        "python", "create_violin_plot.py",
        "/app/csv1.csv",
        "/app/csv2.csv",
        "--output", f"/app/output/{ifcb_evaluation_params.output_filename}",
        "--title", ifcb_evaluation_params.title,
        "--name1", ifcb_evaluation_params.name1,
        "--name2", ifcb_evaluation_params.name2
    ]
    
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