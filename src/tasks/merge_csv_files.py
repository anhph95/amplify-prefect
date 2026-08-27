from prefect import task
import pandas as pd
import os
from typing import List, Optional

from prefect import get_run_logger



@task(log_prints=True)
def merge_csv_files(csv_files: List[str], output_path: str, add_source_column: bool = False) -> str:
    """
    Merge multiple CSV files into a single CSV file.
    
    Args:
        csv_files: List of paths to CSV files to merge
        output_path: Path where the merged CSV will be saved
        add_source_column: If True, add a column indicating the source file
        
    Returns:
        Path to the merged CSV file
    """
    logger = get_run_logger()
    
    # Filter out files that don't exist
    existing_files = [f for f in csv_files if os.path.exists(f)]
    
    if not existing_files:
        logger.warning(f"No CSV files exist from the provided list: {csv_files}")
        return None
        
    if len(existing_files) != len(csv_files):
        missing_files = set(csv_files) - set(existing_files)
        logger.warning(f"Some CSV files are missing and will be skipped: {missing_files}")
    
    logger.info(f"Merging {len(existing_files)} CSV files into {output_path}")
    
    # Read and merge all CSV files
    dataframes = []
    for csv_file in existing_files:
        try:
            df = pd.read_csv(csv_file)
            
            if add_source_column:
                # Add source file name as a column
                source_name = os.path.splitext(os.path.basename(csv_file))[0]
                df['source_file'] = source_name
                
            dataframes.append(df)
            logger.info(f"Read {len(df)} rows from {csv_file}")
            
        except Exception as e:
            logger.error(f"Failed to read {csv_file}: {str(e)}")
            continue
    
    if not dataframes:
        logger.error("No CSV files could be read successfully")
        return None
        
    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save merged CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully merged {len(dataframes)} files into {output_path} with {len(merged_df)} total rows")
    
    return output_path
