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
## run_bmz_on_bia_data

Script for running models from thr BMZ on data from the BIA. 

## Usage

Analise a remote OME-Zarr image using a BMZ model:

    python run_bmz_on_bia_data.py model_DOI/BMZ_ID ome_zarr_uri

For example: 

    python run_bmz_on_bia_data.py "10.5281/zenodo.5764892" "https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD634/05e66f0c-d0a5-4198-b0ae-09f4b738d2fa/05e66f0c-d0a5-4198-b0ae-09f4b738d2fa.zarr/0"

You can select the z-planes, time points or channels to analyse using the `--z_slices`, `--t_slices` and `--channel` options. For example: 

    python run_bmz_on_bia_data.py "10.5281/zenodo.5764892" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/S-BSST410/IM1.zarr/0" --z_slices 42 43 --channel 0

You can crop the image around the center using the `--crop_image` option, for example:

    python run_bmz_on_bia_data.py "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD634/019da5bd-f6b3-445a-b9c7-0f0363133e34/019da5bd-f6b3-445a-b9c7-0f0363133e34.zarr/0" --crop_image 256 256

By default, the input and prediction images are shown you can prevent this with the `--plot_images` option:

    python run_bmz_on_bia_data.py model_DOI ome_zarr_uri --plot_images False




