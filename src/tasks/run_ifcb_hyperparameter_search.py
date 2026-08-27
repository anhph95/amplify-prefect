from prefect import task, get_run_logger
import os
import itertools
from typing import List, Union
import numpy as np

from src.params.params_ifcb_hyperparameter_search import IFCBHyperparameterSearchParams, HyperparameterRange
from src.params.params_ifcb_flow_metric import IFCBTrainingParams
from src.flows.ifcb_training import ifcb_training_flow


def _generate_values_from_range(param_range: HyperparameterRange) -> List[Union[float, int]]:
    """Generate list of values from a hyperparameter range specification."""
    if param_range.values is not None:
        return param_range.values
    elif param_range.min_val is not None and param_range.max_val is not None and param_range.steps is not None:
        return np.linspace(param_range.min_val, param_range.max_val, param_range.steps).tolist()
    else:
        raise ValueError("HyperparameterRange must specify either 'values' or 'min_val', 'max_val', and 'steps'")


def _create_parameter_combinations(search_params: IFCBHyperparameterSearchParams) -> List[dict]:
    """Generate all parameter combinations for the hyperparameter search."""
    param_grid = {}
    
    # Add contamination values if specified
    if search_params.contamination_range is not None:
        param_grid['contamination'] = _generate_values_from_range(search_params.contamination_range)
    else:
        param_grid['contamination'] = [search_params.default_contamination]
    
    # Add max_samples values if specified
    if search_params.max_samples_range is not None:
        param_grid['max_samples'] = _generate_values_from_range(search_params.max_samples_range)
    else:
        param_grid['max_samples'] = [search_params.default_max_samples]
    
    # Add max_features values if specified
    if search_params.max_features_range is not None:
        param_grid['max_features'] = _generate_values_from_range(search_params.max_features_range)
    else:
        param_grid['max_features'] = [search_params.default_max_features]
    
    
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = []
    
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


def _create_output_subdir(base_output_dir: str, param_combo: dict) -> str:
    """Create a subdirectory name based on parameter combination."""
    subdir_name = "_".join([f"{key}_{value}" for key, value in param_combo.items()])
    subdir_path = os.path.join(base_output_dir, subdir_name)
    os.makedirs(subdir_path, exist_ok=True)
    return subdir_path


@task(log_prints=True)
def run_ifcb_hyperparameter_search(search_params: IFCBHyperparameterSearchParams):
    """
    Run IFCB hyperparameter search by training models with different parameter combinations.
    """
    logger = get_run_logger()
    
    # Generate all parameter combinations
    param_combinations = _create_parameter_combinations(search_params)
    logger.info(f"Generated {len(param_combinations)} parameter combinations")
    
    # Track results
    results = []
    
    for i, param_combo in enumerate(param_combinations):
        logger.info(f"Running combination {i+1}/{len(param_combinations)}: {param_combo}")
        
        # Create output subdirectory for this combination
        output_subdir = _create_output_subdir(search_params.base_output_dir, param_combo)
        
        # Create IFCB training parameters for this combination
        training_params = IFCBTrainingParams(
            data_dir=search_params.data_dir,
            output_dir=output_subdir,
            id_file=search_params.id_file,
            n_jobs=search_params.n_jobs,
            contamination=param_combo['contamination'],
            aspect_ratio=search_params.aspect_ratio,
            chunk_size=search_params.chunk_size,
            model_filename=search_params.model_filename,
            max_samples=param_combo['max_samples'],
            max_features=param_combo['max_features']
        )
        
        try:
            # Run IFCB training flow as subflow
            ifcb_training_flow(training_params)
            
            result = {
                'combination': param_combo,
                'output_dir': output_subdir,
                'status': 'success'
            }
            logger.info(f"Successfully completed combination {i+1}")
            
        except Exception as e:
            logger.error(f"Failed combination {i+1}: {str(e)}")
            result = {
                'combination': param_combo,
                'output_dir': output_subdir,
                'status': 'failed',
                'error': str(e)
            }
        
        results.append(result)
    
    # Log summary
    successful_runs = [r for r in results if r['status'] == 'success']
    failed_runs = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"Hyperparameter search completed: {len(successful_runs)} successful, {len(failed_runs)} failed")
    
    if failed_runs:
        logger.warning(f"Failed combinations: {[r['combination'] for r in failed_runs]}")
    
    return results