from prefect import flow

from src.params.params_stingray_image_analysis import (
    ImageAbundanceParams,
    StingrayCruiseParams,
)
from src.tasks.pull_images import pull_images
from src.tasks.run_stingray_image_analysis import run_image_abundance


@flow(name="Stingray Image Abundance", log_prints=True)
def stingray_image_abundance(
    cruise_params: StingrayCruiseParams,
    abundance_params: ImageAbundanceParams,
) -> None:
    """Compute image abundance using Prefect UI parameters."""
    pull_images([cruise_params.image])
    run_image_abundance(cruise_params, abundance_params)


if __name__ == "__main__":
    stingray_image_abundance.serve(name="stingray-image-abundance")
