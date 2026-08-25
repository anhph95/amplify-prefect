from prefect import flow, get_run_logger
import os

from src.params.params_ifcb_flow_metric import IFCBFullEvaluationParams, IFCBInferenceParams, IFCBEvaluationParams
from src.tasks.pull_images import pull_images
from src.tasks.run_ifcb_flow_metric_inference import run_ifcb_flow_metric_inference
from src.tasks.run_ifcb_flow_metric_evaluation import run_ifcb_flow_metric_evaluation
from src.tasks.merge_csv_files import merge_csv_files
from src.utils.bin_utils import create_bin_type_id_file


@flow(name="IFCB Flow Metric Full Evaluation", log_prints=True)
def ifcb_full_evaluation_flow(ifcb_full_evaluation_params: IFCBFullEvaluationParams):
    """
    Flow for full IFCB flow metric evaluation comparing known bad vs normal data.
    
    This flow:
    1. Finds all subdirectories in the bad I and bad D data directories
    2. For each subdirectory:
       - Runs inference with appropriate model (I model for I data, D model for D data)
       - Runs inference on normal data (separated into I and D bins)
       - Creates violin plots comparing bad vs normal for each bin type
       - Creates merged plots combining I and D results
    """
    
    logger = get_run_logger()
    
    # Create output directory if it doesn't exist
    os.makedirs(ifcb_full_evaluation_params.output_dir, exist_ok=True)
    logger.info(f"Output directory: {ifcb_full_evaluation_params.output_dir}")
    
    # Define the Docker image
    ifcb_image = ifcb_full_evaluation_params.ifcb_image
    logger.info(f"Using Docker image: {ifcb_image}")

    # Pull the latest image
    pull_images([ifcb_image])
    
    # Check that bad data directories exist and count bins
    bad_i_bins = len([f for f in os.listdir(ifcb_full_evaluation_params.bad_i_data_dir) 
                     if f.endswith('.adc')])
    bad_d_bins = len([f for f in os.listdir(ifcb_full_evaluation_params.bad_d_data_dir) 
                     if f.endswith('.adc')])
    
    logger.info(f"Found {bad_i_bins} bad I bins in {ifcb_full_evaluation_params.bad_i_data_dir}")
    logger.info(f"Found {bad_d_bins} bad D bins in {ifcb_full_evaluation_params.bad_d_data_dir}")
    
    # Run inference on normal data with separate I and D models
    normal_i_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, "normal_data_i_bins_scores.csv")
    normal_d_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, "normal_data_d_bins_scores.csv")
    
    logger.info("Running inference on normal data with separate I and D models...")
    
    # Create temporary ID files for I and D bins
    temp_i_file, normal_i_bins = create_bin_type_id_file(
        ifcb_full_evaluation_params.normal_data_dir,
        "I",
        validate_paths=ifcb_full_evaluation_params.validate_bin_paths,
        logger=logger,
    )
    temp_d_file, normal_d_bins = create_bin_type_id_file(
        ifcb_full_evaluation_params.normal_data_dir,
        "D",
        validate_paths=ifcb_full_evaluation_params.validate_bin_paths,
        logger=logger,
    )
    
    logger.info(f"Found {normal_i_bins} I bins and {normal_d_bins} D bins in normal data")
    
    try:
        # Run inference on I bins if any exist
        if normal_i_bins > 0:
            logger.info(f"Processing {normal_i_bins} normal I bins with I model...")
            normal_i_inference_params = IFCBInferenceParams(
                data_dir=ifcb_full_evaluation_params.normal_data_dir,
                output_dir=ifcb_full_evaluation_params.output_dir,
                model_path=ifcb_full_evaluation_params.i_model_path,
                id_file=temp_i_file,
                n_jobs=ifcb_full_evaluation_params.n_jobs,
                aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
                chunk_size=ifcb_full_evaluation_params.chunk_size,
                output_filename="normal_data_i_bins_scores.csv"
            )
            run_ifcb_flow_metric_inference(normal_i_inference_params, ifcb_image)
        
        # Run inference on D bins if any exist
        if normal_d_bins > 0:
            logger.info(f"Processing {normal_d_bins} normal D bins with D model...")
            normal_d_inference_params = IFCBInferenceParams(
                data_dir=ifcb_full_evaluation_params.normal_data_dir,
                output_dir=ifcb_full_evaluation_params.output_dir,
                model_path=ifcb_full_evaluation_params.d_model_path,
                id_file=temp_d_file,
                n_jobs=ifcb_full_evaluation_params.n_jobs,
                aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
                chunk_size=ifcb_full_evaluation_params.chunk_size,
                output_filename="normal_data_d_bins_scores.csv"
            )
            run_ifcb_flow_metric_inference(normal_d_inference_params, ifcb_image)
            
    finally:
        # Clean up temporary files
        if temp_i_file and os.path.exists(temp_i_file):
            os.unlink(temp_i_file)
        if temp_d_file and os.path.exists(temp_d_file):
            os.unlink(temp_d_file)
    
    logger.info(f"Normal data inference completed: {normal_i_bins} I bins, {normal_d_bins} D bins")
    
    # Run inference on bad I data directory (if it has bins)
    if bad_i_bins > 0:
        logger.info(f"Processing {bad_i_bins} bad I bins with I model...")
        
        bad_i_csv_filename = "bad_i_bins_scores.csv"
        bad_i_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, bad_i_csv_filename)
        
        bad_i_inference_params = IFCBInferenceParams(
            data_dir=ifcb_full_evaluation_params.bad_i_data_dir,
            output_dir=ifcb_full_evaluation_params.output_dir,
            model_path=ifcb_full_evaluation_params.i_model_path,
            id_file=None,  # No ID file needed since directory only contains I bins
            n_jobs=ifcb_full_evaluation_params.n_jobs,
            aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
            chunk_size=ifcb_full_evaluation_params.chunk_size,
            output_filename=bad_i_csv_filename
        )
        run_ifcb_flow_metric_inference(bad_i_inference_params, ifcb_image)
        logger.info("Bad I bins inference completed")
    
    # Run inference on bad D data directory (if it has bins)
    if bad_d_bins > 0:
        logger.info(f"Processing {bad_d_bins} bad D bins with D model...")
        
        bad_d_csv_filename = "bad_d_bins_scores.csv"
        bad_d_csv_path = os.path.join(ifcb_full_evaluation_params.output_dir, bad_d_csv_filename)
        
        bad_d_inference_params = IFCBInferenceParams(
            data_dir=ifcb_full_evaluation_params.bad_d_data_dir,
            output_dir=ifcb_full_evaluation_params.output_dir,
            model_path=ifcb_full_evaluation_params.d_model_path,
            id_file=None,  # No ID file needed since directory only contains D bins
            n_jobs=ifcb_full_evaluation_params.n_jobs,
            aspect_ratio=ifcb_full_evaluation_params.aspect_ratio,
            chunk_size=ifcb_full_evaluation_params.chunk_size,
            output_filename=bad_d_csv_filename
        )
        run_ifcb_flow_metric_inference(bad_d_inference_params, ifcb_image)
        logger.info("Bad D bins inference completed")
    
    # Create evaluation plots: separate I bins, D bins, and merged plots
    normal_data_clean = ifcb_full_evaluation_params.normal_data_name.replace(' ', '_').lower()
    
    # 1. I bins only evaluation (if both datasets have I bins)
    if bad_i_bins > 0 and normal_i_bins > 0:
        i_plot_filename = f"evaluation_bad_i_vs_{normal_data_clean}_I_bins.png"
        i_plot_title = f"{ifcb_full_evaluation_params.plot_title_prefix} (I bins): Bad I Data vs {ifcb_full_evaluation_params.normal_data_name}"
        
        i_evaluation_params = IFCBEvaluationParams(
            csv1_path=bad_i_csv_path,
            csv2_path=normal_i_csv_path,
            output_dir=ifcb_full_evaluation_params.output_dir,
            output_filename=i_plot_filename,
            title=i_plot_title,
            name1="Bad I Data",
            name2=f"{ifcb_full_evaluation_params.normal_data_name} (I bins)"
        )
        
        run_ifcb_flow_metric_evaluation(i_evaluation_params, ifcb_image)
        logger.info(f"Created I bins evaluation plot: {i_plot_filename}")
    
    # 2. D bins only evaluation (if both datasets have D bins)
    if bad_d_bins > 0 and normal_d_bins > 0:
        d_plot_filename = f"evaluation_bad_d_vs_{normal_data_clean}_D_bins.png"
        d_plot_title = f"{ifcb_full_evaluation_params.plot_title_prefix} (D bins): Bad D Data vs {ifcb_full_evaluation_params.normal_data_name}"
        
        d_evaluation_params = IFCBEvaluationParams(
            csv1_path=bad_d_csv_path,
            csv2_path=normal_d_csv_path,
            output_dir=ifcb_full_evaluation_params.output_dir,
            output_filename=d_plot_filename,
            title=d_plot_title,
            name1="Bad D Data",
            name2=f"{ifcb_full_evaluation_params.normal_data_name} (D bins)"
        )
        
        run_ifcb_flow_metric_evaluation(d_evaluation_params, ifcb_image)
        logger.info(f"Created D bins evaluation plot: {d_plot_filename}")
    
    # 3. Merged evaluation (combining I and D bins)
    if (bad_i_bins > 0 or bad_d_bins > 0) and (normal_i_bins > 0 or normal_d_bins > 0):
        # Merge bad data CSV files
        bad_merged_filename = "bad_merged_scores.csv"
        bad_merged_path = os.path.join(ifcb_full_evaluation_params.output_dir, bad_merged_filename)
        
        bad_csv_files = []
        if bad_i_bins > 0:
            bad_csv_files.append(bad_i_csv_path)
        if bad_d_bins > 0:
            bad_csv_files.append(bad_d_csv_path)
            
        merge_csv_files(
            csv_files=bad_csv_files,
            output_path=bad_merged_path
        )
        
        # Merge normal data CSV files
        normal_merged_filename = "normal_data_merged_scores.csv"
        normal_merged_path = os.path.join(ifcb_full_evaluation_params.output_dir, normal_merged_filename)
        
        normal_csv_files = []
        if normal_i_bins > 0:
            normal_csv_files.append(normal_i_csv_path)
        if normal_d_bins > 0:
            normal_csv_files.append(normal_d_csv_path)
            
        merge_csv_files(
            csv_files=normal_csv_files,
            output_path=normal_merged_path
        )
        
        # Create merged evaluation plot
        merged_plot_filename = f"evaluation_bad_vs_{normal_data_clean}_merged.png"
        merged_plot_title = f"{ifcb_full_evaluation_params.plot_title_prefix} (All bins): Bad Data vs {ifcb_full_evaluation_params.normal_data_name}"
        
        merged_evaluation_params = IFCBEvaluationParams(
            csv1_path=bad_merged_path,
            csv2_path=normal_merged_path,
            output_dir=ifcb_full_evaluation_params.output_dir,
            output_filename=merged_plot_filename,
            title=merged_plot_title,
            name1="Bad Data (All bins)",
            name2=f"{ifcb_full_evaluation_params.normal_data_name} (All bins)"
        )
        
        run_ifcb_flow_metric_evaluation(merged_evaluation_params, ifcb_image)
        logger.info(f"Created merged evaluation plot: {merged_plot_filename}")
    
    logger.info(f"Full evaluation completed - processed {bad_i_bins} bad I bins and {bad_d_bins} bad D bins")


if __name__ == "__main__":
    ifcb_full_evaluation_flow.serve(
        name="ifcb-flow-metric-full-evaluation",
        tags=["evaluation", "ifcb", "anomaly-detection", "comparison"],
    )
