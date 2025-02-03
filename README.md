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

After running an analysis, the results will be found in `/results`, relative to the top level of this repository, assuming you didn't run the script from another location. See *results*, further below, for details.

## Running for development, in a local conda environment

If you'll want to make changes to the code and have them take effect immediately, install the bia_bmz_integrator package in editable mode. A distinct conda environment exists for this, and if you're running from the top level of this repository, create it like so: 

    conda env create -f ./dev/env.yaml 

and you can activate it like this:

    conda activate bia-bmz-integration-dev

then your changes to any bia_bmz_integrator code will work straight away. 

Again, after running an analysis, the results will be found in `/results`, relative to the top level of this repository, assuming you didn't run the script from another location. See *results*, further below, for details.

## Running in a container

To build the docker image, from the top level of this repository (note the period):

    docker build -t bia-bmz-integration . 

and start an interactive terminal in a container:

    docker run -it bia-bmz-integration

and you should find yourself within the bia-bmz-integration virtual environment, inside the container. Thus, as above, to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, to view parameters with --help, you can run:

    bia_bmz_benchmark --help

see *bia_bmz_benchmark*, further below, for more details.

After running an analysis, the results will be found in `/results`, in the root of the container, assuming you didn't run the script from another location. See *results*, further below, for details.

## Running with an interactive Singularity container on a cluster

Once you've built the Docker image (see above), you can save it:

    docker save -o bia-bmz-integration.tar bia-bmz-integration:latest

and, once you've got the resulting .tar file on to a cluster, for example by using rsync:

    rsync -avz bia-bmz-integration.tar USER@CLUSTER:/DESTINATION_DIR

*where USER should be your username on the cluster, and CLUSTER is the cluster's address, and DESTINATION_DIR is wherever you intend to put the container.*

Then, once logged in to the cluster, you can start an interactive session to build a Singularity .sif from it. For example, assuming you're in the same folder as the Docker .tar file:

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

see *bia_bmz_benchmark*, further below, for more details.

The results will be saved in `DESTINATION_DIR/results` — see *results*, further below.

## Running in batch with Nextflow on a cluster

The file `run_batch.nf` in this repository is a Nextflow script to run an analysis in the Singularity container described above, and is designed to take inputs from a text file, an example of which is provided here in `batch_input.txt`. In this case, each line of the text file is an input to bia_bmz_benchmark (see below), thus allowing you to run an analysis in batch. 

To run the analysis in batch on the cluster, you'll need to copy the three files `nextflow.config`, `run_batch.nf`, and `batch_input.txt`. For example, with rsync:

    rsync -avz nextflow.config USER@CLUSTER:/DESTINATION_DIR

*where, as above, USER should be your username on the cluster, and CLUSTER is the cluster's address, and DESTINATION_DIR is wherever you intend to put the files.*

Note that the scripts here have been written assuming you're running everything from the location you've put the files (i.e. DESTINATION_DIR, above), and moreover, that this location is also where you've put the Singularity container. Thus, you can run a collection of jobs like so:

    salloc --mem=4g --time=00:10:00
    module load nextflow
    nextflow run run_batch.nf -with-singularity ./bia-bmz-integration.sif

and remember, if you want to check your running jobs, if you open another terminal and log in to the cluster, you can use `squeue`.

The results will be saved in `DESTINATION_DIR/results` — see *results*, further below.

## bia_bmz_benchmark

This is the script for running models from the BMZ on data from the BIA. This script will also provide benchmarking if a reference annotation for an image is provided.

The commands below apply whether in an interactive shell inside a container (including on a cluster), or running locally on your machine.

To analyse a remote OME-Zarr image using a BMZ model:

    bia_bmz_benchmark model_DOI/BMZ_ID image_ome_zarr_uri

Benchmark a BMZ model on an annotated BIA image:

    bia_bmz_benchmark model_DOI/BMZ_ID image_ome_zarr_uri annotation_ome_zarr_uri

And note that you can, and should, provide the study accession, and the dataset uuid for the input image, and if providing an annotation image, the dataset uuid for that, too. You can do this with `--study_acc`, `--dataset_uuid`, and `--annotation_dataset_uuid`, respectively. 

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

## Results

The results of a prediction or benchmarking run are saved in a json file, like this: https://github.com/BioImage-Archive/BIA-astro/blob/main/src/data/ai-dataset-benchmarking/model-dataset-table.json. If an annotation image was provided, then the benchmarking metrics — precision, recall, IoU, and dice — will have values. 

The json file is written to `/results/jsons`, one file per run. If you want to put all your results into one json, you can use the command `amalgamate_jsons`, and this'll create a json `all_results` into the `/results` folder. 

Note that `amalgamate_jsons` assumes that you're running it from one level above `/results`. If you run the main benchmarking script from the top level of this repository, and then `amalgamate_jsons` without changing folders, all will be well.

Images are also saved with each analysis run — under `/results/images`, there will be a slice of the input image, one of the prediction image, and if an annotation image was provided, the ground truth image, too. 