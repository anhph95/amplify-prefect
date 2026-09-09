from prefect import flow

from src.params.params_stingray_image_analysis import (
    FrameTimestampParams,
    StingrayCruiseParams,
)
from src.tasks.pull_images import pull_images
from src.tasks.run_stingray_image_analysis import run_frame_timestamps


@flow(name="Stingray Frame Timestamps", log_prints=True)
def stingray_frame_timestamps(
    cruise_params: StingrayCruiseParams,
    timestamp_params: FrameTimestampParams,
) -> None:
    """Create shared video and frame lists using Prefect UI parameters."""
    pull_images([cruise_params.image])
    run_frame_timestamps(cruise_params, timestamp_params)


if __name__ == "__main__":
    stingray_frame_timestamps.serve(name="stingray-frame-timestamps")
