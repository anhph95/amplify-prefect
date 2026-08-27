from prefect import task
import os
import json
import pandas as pd
from pathlib import Path
import base64

from prefect import get_run_logger
from prefect.artifacts import create_markdown_artifact

from src.params.params_feature_validation import FeatureValidationParams


@task(log_prints=True)
def create_combined_validation_report(validation_params: FeatureValidationParams):
    """
    Create a single comprehensive validation report combining feature metrics and blob comparison.
    """
    logger = get_run_logger()

    markdown_report = "# IFCB Complete Validation Report\n\n"
    markdown_report += "---\n\n"

    # ========== FEATURE VALIDATION SECTION ==========
    markdown_report += "# Part 1: Feature Validation\n\n"

    feature_summary_file = os.path.join(validation_params.output_directory, validation_params.summary_filename)
    feature_results_file = os.path.join(validation_params.output_directory, validation_params.output_filename)

    if os.path.exists(feature_summary_file):
        with open(feature_summary_file, 'r') as f:
            summary = json.load(f)

        # Add data coverage section if available
        data_coverage = ""
        if 'num_rows_analyzed' in summary:
            sample_list = summary.get('samples_analyzed', [])
            data_coverage = f"""## Data Analysis Coverage
- **Total rows analyzed**: {summary['num_rows_analyzed']:,}
- **Unique samples analyzed**: {summary.get('num_samples_analyzed', 0)}
- **Samples**: {', '.join(sample_list) if len(sample_list) <= 20 else ', '.join(sample_list[:20]) + f' ... and {len(sample_list)-20} more'}

---

"""

        markdown_report += f"""{data_coverage}## Overall Statistics
- **Total features compared**: {summary['total_features']}
- **Mean RMSE**: {summary['mean_rmse']:.4f}
- **Median RMSE**: {summary['median_rmse']:.4f}
- **Mean MAE**: {summary['mean_mae']:.4f}
- **Median MAE**: {summary['median_mae']:.4f}
- **Mean R²**: {summary['mean_r2']:.4f}
- **Median R²**: {summary['median_r2']:.4f}
- **Mean Pearson correlation**: {summary['mean_pearson_r']:.4f}
- **Median Pearson correlation**: {summary['median_pearson_r']:.4f}

## Quality Metrics
- **Features with high correlation (r > 0.9)**: {summary['features_with_high_correlation']}
- **Features with low R² (< 0.5)**: {summary['features_with_low_r2']}

## Configuration
- **Predicted**: `{validation_params.pred_bucket}.{validation_params.pred_schema}.{validation_params.pred_table}`
- **Ground Truth**: `{validation_params.gt_bucket}.{validation_params.gt_schema}.{validation_params.gt_table}`

---

"""

    if os.path.exists(feature_results_file):
        metrics_df = pd.read_csv(feature_results_file)

        top_by_r2 = metrics_df.nlargest(10, 'r2')[['feature', 'r2', 'rmse', 'mae', 'pearson_r']]
        worst_by_r2 = metrics_df.nsmallest(10, 'r2')[['feature', 'r2', 'rmse', 'mae', 'pearson_r']]

        markdown_report += "## Top 10 Features by R² Score\n\n"
        markdown_report += top_by_r2.to_markdown(index=False) + "\n\n"
        markdown_report += "---\n\n"

        markdown_report += "## Bottom 10 Features by R² Score\n\n"
        markdown_report += worst_by_r2.to_markdown(index=False) + "\n\n"
        markdown_report += "---\n\n"

    # ========== BLOB COMPARISON SECTION ==========
    if validation_params.enable_blob_comparison:
        markdown_report += "# Part 2: Blob Comparison\n\n"

        blob_output_dir = os.path.join(validation_params.output_directory, "blob_comparison")
        results_file = os.path.join(blob_output_dir, "blob_comparison_results.json")
        csv_file = os.path.join(blob_output_dir, "blob_comparison_details.csv")
        comparison_images_dir = os.path.join(blob_output_dir, "blob_comparisons")

        # Add summary statistics
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)

            summary_stats = results['summary_stats']

            markdown_report += f"""## Overall Statistics
- **Total blobs compared**: {summary_stats['total_blobs']}
- **Mean IoU**: {summary_stats['mean_iou']:.4f}
- **Median IoU**: {summary_stats['median_iou']:.4f}
- **Std Dev IoU**: {summary_stats['std_iou']:.4f}
- **Mean Dice coefficient**: {summary_stats['mean_dice']:.4f}
- **Median Dice coefficient**: {summary_stats['median_dice']:.4f}
- **Mean accuracy**: {summary_stats['mean_accuracy']:.4f}

## Quality Metrics
- **Perfect matches (IoU=1.0)**: {summary_stats['perfect_matches']} ({100*summary_stats['perfect_matches']/summary_stats['total_blobs']:.1f}%)
- **Near-perfect matches (IoU≥0.95)**: {summary_stats['near_perfect_matches']} ({100*summary_stats['near_perfect_matches']/summary_stats['total_blobs']:.1f}%)
- **Poor matches (IoU<0.5)**: {summary_stats['poor_matches']} ({100*summary_stats['poor_matches']/summary_stats['total_blobs']:.1f}%)

## Configuration
- **Predicted Blobs**: `s3://{validation_params.blob_pred_bucket}/{validation_params.blob_pred_prefix}`
- **Ground Truth Blobs**: `s3://{validation_params.blob_gt_bucket}/{validation_params.blob_gt_prefix}`

---

"""

        # Add worst/best matches table
        if os.path.exists(csv_file):
            metrics_df = pd.read_csv(csv_file)

            # Add blob analysis metrics
            num_blob_rows = len(metrics_df)
            unique_blob_samples = metrics_df['sample_id'].nunique() if 'sample_id' in metrics_df.columns else 0
            blob_sample_list = sorted(metrics_df['sample_id'].unique()) if 'sample_id' in metrics_df.columns else []

            markdown_report += f"""## Data Analysis Coverage
- **Total blobs analyzed**: {summary_stats['total_blobs']:,}
- **Total rows analyzed**: {num_blob_rows:,}
- **Unique samples analyzed**: {unique_blob_samples}
- **Samples**: {', '.join(blob_sample_list) if len(blob_sample_list) <= 20 else ', '.join(blob_sample_list[:20]) + f' ... and {len(blob_sample_list)-20} more'}

---

"""

            worst_matches = metrics_df.nsmallest(10, 'iou')[['sample_id', 'roi_number', 'iou', 'dice', 'accuracy', 'diff_pixels']]
            best_matches = metrics_df.nlargest(10, 'iou')[['sample_id', 'roi_number', 'iou', 'dice', 'accuracy', 'diff_pixels']]

            markdown_report += "## Top 10 Worst Blob Matches\n\n"
            markdown_report += worst_matches.to_markdown(index=False) + "\n\n"
            markdown_report += "---\n\n"

            markdown_report += "## Top 10 Best Blob Matches\n\n"
            markdown_report += best_matches.to_markdown(index=False) + "\n\n"
            markdown_report += "---\n\n"

        # Add visualization images
        if os.path.exists(comparison_images_dir):
            image_files = sorted(Path(comparison_images_dir).glob("*.png"))[:validation_params.blob_top_n_worst]

            if image_files:
                markdown_report += "## Visual Comparisons (Worst Matches)\n\n"
                markdown_report += "Red = False Positive, Green = False Negative, White = Correct\n\n"

                for idx, image_path in enumerate(image_files):
                    # Extract info from filename
                    filename = image_path.stem
                    parts = filename.rsplit('_iou', 1)
                    sample_roi = parts[0] if len(parts) > 0 else filename
                    iou_str = parts[1] if len(parts) > 1 else "unknown"

                    # Read and encode image as base64
                    with open(image_path, 'rb') as f:
                        image_data = f.read()

                    image_base64 = base64.b64encode(image_data).decode('utf-8')

                    # Add to markdown
                    markdown_report += f"### {idx+1}. {sample_roi} (IoU={iou_str})\n\n"
                    markdown_report += f'<img src="data:image/png;base64,{image_base64}" alt="{sample_roi}" style="max-width: 100%;"/>\n\n'

    # Create single comprehensive artifact
    create_markdown_artifact(
        key="complete-validation-report",
        markdown=markdown_report,
        description="Complete IFCB Validation Report (Features + Blobs)"
    )

    logger.info(f"✓ Created comprehensive validation report artifact")
