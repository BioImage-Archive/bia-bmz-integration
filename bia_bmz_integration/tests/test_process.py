import pytest
import numpy as np
from bia_bmz_integration import process,run_model_inference

# tensors = [
#     np.random.rand(3,512,512),
#     np.random.rand(1,512,512),
#     np.random.rand(3,1024,1024),
#     np.random.rand(1,1024,1024),

# ]

models = ["loyal-squid", "noisy-fish", "affable-shark", "powerful-chipmunk", "loyal-parrot", "faithful-chicken", "frank-water-buffalo"]
# datasets = ["S-BIAD1026", "S-BIAD634", "S-BIAD144/S-BIAD916", "BSST666"]

datasets = [
("frank-water-buffalo","https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/23_low.ome.zarr/0","https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/23_high.ome.zarr/0"),

("powerful-chipmunk", "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/experimentA_01_WT_PDMP.ome.zarr/0", "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/experimentA_01_WT_PDMP_cp_mask.ome.zarr/0" ),

("affable-shark","https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0", "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26_mask.ome.zarr/0"),
]


# @pytest.mark.parametrize("bmz_model", models)
@pytest.mark.parametrize("dataset", datasets)
# def test_model_prediction_on_bia(bmz_model,dataset):
def test_model_prediction_on_bia(dataset):
    
    """
    Test the benchmark_bmz_on_bia_data.py main function using all model names, dataset URIs, and mask URI.
    """
    # Call the main function with the test input
    bmz_model, input_uri, ref_uri = dataset
    output_array = process(
        bmz_model=bmz_model,
        ome_zarr_uri=input_uri,
        reference_annotations=ref_uri,
        plot_images=False,  # Disable plot for testing
    )

    # Ensure the output is a numpy array
    # assert isinstance(output_array, np.ndarray), f"Output is not a numpy array for model {bmz_model}"

    # # Ensure the output has content (non-empty array)
    # assert output_array.size > 0, f"Output array is empty for model {bmz_model}"

# @pytest.mark.parametrize("tensor", tensors)
# @pytest.mark.parametrize("bmz_model", models)
# def test_model_prediction(bmz_model,tensor):
#     return run_model_inference(bmz_model,tensor)