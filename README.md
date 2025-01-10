# BioImage Archive — BioImage Model Zoo integration

*For testing and deploying BioImage Model Zoo models on BioImage Archive images.*

## Running locally with conda

After cloning, from the top level of this repository, create the conda environment like so:

    conda env create -f env.yaml -p ./env

and this'll also install bia_bmz_integrator. Activate the environment thus:

    conda activate bia-bmz-integration

and once the environment is activated, to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, view parameters with --help:

    bia_bmz_benchmark --help

See *bia_bmz_benchmark*, further below, for more details.

## Running in a container

To build the docker image, from the top level of this repository (note the period):

    docker build -t bia-bmz-integration . 

and start an interative terminal in a container:

    docker run -it bia-bmz-integration

and you should find yourself within the bia-bmz-integration virtual environment, inside the container. Thus, as above, to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, view parameters with --help: to  you can run:

    bia_bmz_benchmark --help

See *bia_bmz_benchmark*, directly below, for more details.

## bia_bmz_benchmark

Script for running from the BMZ on data from the BIA. This script will also provide benchmarking if a reference annotation for an image is provided.

The commands below apply, whether in an interactive shell inside a container, or running locally on your machine.

To analyse a remote OME-Zarr image using a BMZ model:

    bia_bmz_benchmark model_DOI/BMZ_ID image_ome_zarr_uri

Benchmark a BMZ model on an annotated BIA image:

    bia_bmz_benchmark model_DOI/BMZ_ID image_ome_zarr_uri annotation_ome_zarr_uri

For example: 

To run a model on an image:

    bia_bmz_benchmark "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" 

or to get benchmarking metrics for the model:

    bia_bmz_benchmark "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26_mask.ome.zarr/0"

You can select the z-planes, time points or channels to analyse using the `--z_slices`, `--t_slices` and `--channel` options. For example: 

    bia_bmz_benchmark "noisy-fish" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/1135_n_H2BtdTomato.ome.zarr/0" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/1135_n_stain_StarDist_goldGT_180_rotation.ome.zarr/0" --z_slices 170 180

You can crop the image around the center using the `--crop_image` option, for example:

    bia_bmz_benchmark "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" --crop_image 256 256

By default, the input and prediction images are shown you can prevent this with the `--plot_images` option:

    bia_bmz_benchmark "affable-shark" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26.ome.zarr/0" "https://uk1s3.embassy.ebi.ac.uk/bia-zarr-test/normal_26_mask.ome.zarr/0" --plot_images False
