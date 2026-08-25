from prefect import task
import docker
import os

from prefect import get_run_logger

from src.prov import on_task_complete
from src.params.params_ifcb_flow_metric import IFCBTrainingParams
from src.utils.bin_utils import create_bin_type_id_file


def generate_feature_config_yaml(params: IFCBTrainingParams) -> str:
    """Generate YAML configuration string from user feature parameters."""
    feature_config = params.feature_config
    config = {
        'spatial_stats': {
            'mean_x': feature_config.use_mean_x,
            'mean_y': feature_config.use_mean_y,
            'std_x': feature_config.use_std_x,
            'std_y': feature_config.use_std_y,
            'median_x': feature_config.use_median_x,
            'median_y': feature_config.use_median_y,
            'iqr_x': feature_config.use_iqr_x,
            'iqr_y': feature_config.use_iqr_y,
        },
        'distribution_shape': {
            'ratio_spread': feature_config.use_ratio_spread,
            'core_fraction': feature_config.use_core_fraction,
        },
        'clipping_detection': {
            'duplicate_fraction': feature_config.use_duplicate_fraction,
            'max_duplicate_fraction': feature_config.use_max_duplicate_fraction,
        },
        'histogram_uniformity': {
            'cv_x': feature_config.use_cv_x,
            'cv_y': feature_config.use_cv_y,
        },
        'statistical_moments': {
            'skew_x': feature_config.use_skew_x,
            'skew_y': feature_config.use_skew_y,
            'kurt_x': feature_config.use_kurt_x,
            'kurt_y': feature_config.use_kurt_y,
        },
        'pca_orientation': {
            'angle': feature_config.use_angle,
            'eigen_ratio': feature_config.use_eigen_ratio,
        },
        'edge_features': {
            'left_edge_fraction': feature_config.use_left_edge_fraction,
            'right_edge_fraction': feature_config.use_right_edge_fraction,
            'top_edge_fraction': feature_config.use_top_edge_fraction,
            'bottom_edge_fraction': feature_config.use_bottom_edge_fraction,
            'total_edge_fraction': feature_config.use_total_edge_fraction,
        },
        'temporal': {
            't_y_var': feature_config.use_t_y_var,
        },
        'trigger_stats': {
            'roi_trigger_fraction': feature_config.use_roi_trigger_fraction,
        }
    }
    
    import yaml
    return yaml.dump(config)




@task(on_completion=[on_task_complete], log_prints=True)
def run_ifcb_training(ifcb_training_params: IFCBTrainingParams, ifcb_image: str):
    """
    Run IFCB flow metric model training in a Docker container.
    """
    
    client = docker.from_env()
    logger = get_run_logger()
    
    # Set up volumes
    volumes = {
        ifcb_training_params.data_dir: {'bind': '/app/data', 'mode': 'ro'},
        ifcb_training_params.output_dir: {'bind': '/app/output', 'mode': 'rw'}
    }
    
    # Create ID file based on bin type selection or use provided ID file
    id_file_container_path = None
    temp_id_file = None
    
    if ifcb_training_params.id_file is not None:
        # User provided their own ID file
        id_file_container_path = '/app/ids.txt'
        volumes[ifcb_training_params.id_file] = {'bind': id_file_container_path, 'mode': 'ro'}
        logger.info(f'Using provided ID file: {ifcb_training_params.id_file}')
    else:
        # Create ID file based on bin type selection
        logger.info(f'Creating ID file for {ifcb_training_params.bin_type.value}-bins')
        temp_id_file, num_bins = create_bin_type_id_file(
            ifcb_training_params.data_dir,
            ifcb_training_params.bin_type.value,
            validate_paths=ifcb_training_params.validate_bin_paths,
            logger=logger,
        )
        if temp_id_file is None:
            raise ValueError(f"No {ifcb_training_params.bin_type.value}-bins found in directory {ifcb_training_params.data_dir}")
        logger.info(f'Created ID file for {num_bins} {ifcb_training_params.bin_type.value}-bins: {temp_id_file}')
        id_file_container_path = '/app/ids.txt'
        volumes[temp_id_file] = {'bind': id_file_container_path, 'mode': 'ro'}
    
    # Build command arguments
    command_args = [
        "python", "train.py",
        "/app/data",
        "--n-jobs", str(ifcb_training_params.n_jobs),
        "--contamination", str(ifcb_training_params.contamination),
        "--aspect-ratio", str(ifcb_training_params.aspect_ratio),
        "--chunk-size", str(ifcb_training_params.chunk_size),
        "--max-samples", str(ifcb_training_params.max_samples),
        "--max-features", str(ifcb_training_params.max_features),
        "--model", f"/app/output/{ifcb_training_params.model_filename}"
    ]
    
    # Add id-file flag (always present now, either user-provided or generated)
    command_args.extend(["--id-file", id_file_container_path])
    
    # Generate and add feature configuration
    feature_config_yaml = generate_feature_config_yaml(ifcb_training_params)
    command_args.extend(["--config", feature_config_yaml])
    
    logger.info(f'Running container with command: {" ".join(command_args)}')
    
    try:
        container = client.containers.run(
            ifcb_image,
            command_args,
            volumes=volumes,
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
    finally:
        # Clean up temporary ID file if created
        if temp_id_file and os.path.exists(temp_id_file):
            os.unlink(temp_id_file)
            logger.info(f"Cleaned up temporary ID file: {temp_id_file}")
