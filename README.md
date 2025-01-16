# BioImage Archive — BioImage Model Zoo integration

*For testing and deploying BioImage Model Zoo models on BioImage Archive images.*

## Running locally with conda

After cloning, from the top level of this repository, create the conda environment like so:

    conda env create -f env.yaml

and this'll also install bia_bmz_integrator. Activate the environment thus:

    conda activate bia-bmz-integration

and once the environment is activated, to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, view parameters with --help:

    bia_bmz_benchmark --help

see *bia_bmz_benchmark*, further below, for more details.

## Running in a container

To build the docker image, from the top level of this repository (note the period):

    docker build -t bia-bmz-integration . 

and start an interactive terminal in a container:

    docker run -it bia-bmz-integration

and you should find yourself within the bia-bmz-integration virtual environment, inside the container. Thus, as above, to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, to view parameters with --help, you can run:

    bia_bmz_benchmark --help

see *bia_bmz_benchmark*, further below, for more details.

## Running with Singularity on a cluster

Once you've built the Docker image (see above), you can save it:

    docker save -o bia-bmz-integration.tar bia-bmz-integration:latest

and, once you've got the resulting .tar file on to a cluster, you can start an interactive session to build a Singularity .sif from it. For example, assuming you're in the same folder as the Docker .tar file:

    salloc --mem=8g --time=00:30:00
    module load singularityce
    singularity build bia-bmz-integration.sif docker-archive://bia-bmz-integration.tar

and again, you can use an interactive session to run a container, for example:

    salloc --mem=32g --time=00:30:00
    module load singularityce
    singularity shell bia-bmz-integration.sif

and you'll find yourself in the container shell. 

Best not to use `singularity run` at time of writing — the way Singularity loads the container means that the last line of the Dockerfile, `CMD ["/bin/bash", "-c", "source bia-bmz-integration/bin/activate && exec bash"]`, which activates the python environment, throws an error — Singularity won't find bia-bmz-integration/bin/activate.

Instead, once inside the container shell as above, first go to the root of the container, then bia-bmz-integration, and execute the environment activation from there:

    cd / 
    cd bia-bmz-integration
    source ./bin/activate

then, as above, to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, view parameters with --help: to  you can run:

    bia_bmz_benchmark --help

see *bia_bmz_benchmark*, directly below, for more details.

## bia_bmz_benchmark

Script for running models from the BMZ on data from the BIA. This script will also provide benchmarking if a reference annotation for an image is provided.

The commands below apply whether in an interactive shell inside a container (including on a cluster), or running locally on your machine.

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
