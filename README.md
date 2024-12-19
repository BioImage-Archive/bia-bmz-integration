Create conda env with cuda:

```bash
conda env create -f environment.yml -p ./env
conda activate ./env
```

Create venv without cuda for dev:

```bash 
poetry shell
poetry install
```
## run_benchmark_bmz_on_bia_data

Script for running from the BMZ on data from the BIA. This script will also provide benchmarking  if a reference annotation for an image is provided.

## Usage
Analise a remote OME-Zarr image using a BMZ model:

    python run_benchmark_bmz_on_bia_data.py model_DOI/BMZ_ID image_ome_zarr_uri

Benchmark a BMZ model on an annotated BIA image:

    python run_benchmark_bmz_on_bia_data.py model_DOI/BMZ_ID image_ome_zarr_uri annotation_ome_zarr_uri

For example: 

To run a model on an image

    python run_benchmark_bmz_on_bia_data.py "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" 

or to get benchmarking metrics for the model

    python run_benchmark_bmz_on_bia_data.py "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26_mask.ome.zarr/0"

You can select the z-planes, time points or channels to analyse using the `--z_slices`, `--t_slices` and `--channel` options. For example: 

    python run_benchmark_bmz_on_bia_data.py "noisy-fish" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/1135_n_H2BtdTomato.ome.zarr/0" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/1135_n_stain_StarDist_goldGT_180_rotation.ome.zarr/0" --z_slices 170 180

You can crop the image around the center using the `--crop_image` option, for example:

    python run_benchmark_bmz_on_bia_data.py "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" --crop_image 256 256

By default, the input and prediction images are shown you can prevent this with the `--plot_images` option:

    python run_benchmark_bmz_on_bia_data.py "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26_mask.ome.zarr/0" --plot_images False




