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

After running an analysis, the results will be found in `/results`, relative to the top level of this repository, assuming you didn't run the command from another location. See *Results*, further below, for details.

## Running for development, in a local conda environment

If you'll want to make changes to the code and have them take effect immediately, install the bia_bmz_integrator package in editable mode. A distinct conda environment exists for this, and if you're running from the top level of this repository, create it like so: 

    conda env create -f ./dev/env.yaml 

and you can activate it like this:

    conda activate bia-bmz-integration-dev

then your changes to any bia_bmz_integrator code will work straight away. 

Again, after running an analysis, the results will be found in `/results`, relative to the top level of this repository, assuming you didn't run the command from another location. See *Results*, further below, for details.

## Running locally, in batch

There exists in this repository a shell script, `run_batch.sh`, that you can use to line up multiple model/data combinations when running locally (including in development). This script takes as input a text file, and example of which, `batch_input_example.txt` is provided here. Information on the text file is given in *Using Nextflow to run an analysis in the container*, under *Running on a cluster*, below. And again, see *Results*, further below, for details on the output. 

## Running in a container

To build the docker image, from the top level of this repository (note the period):

    docker build -t bia-bmz-integration . 

and start an interactive terminal in a container:

    docker run -it bia-bmz-integration

and you should find yourself within the bia-bmz-integration virtual environment, inside the container. Thus, as above, to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, to view parameters with --help, you can run:

    bia_bmz_benchmark --help

see *bia_bmz_benchmark*, further below, for more details.

After running an analysis, the results will be found in `/results`, in the root of the container, assuming you didn't run the script from another location. See *Results*, further below, for details.

## Running on a cluster

### *Building a Singularity container*

First, you'll need to get the Docker image onto the cluster. So, once you've built the image (see above), you can save it:

    docker save -o bia-bmz-integration.tar bia-bmz-integration:latest

and then copy the resulting .tar file to a cluster, for example by using rsync:

    rsync -avz bia-bmz-integration.tar USER@CLUSTER:/DESTINATION_DIR

*where USER should be your username on the cluster, and CLUSTER is the cluster's address, and DESTINATION_DIR is wherever you intend to put the container.*

Once logged in to the cluster, you'll build a Singularity container from the Docker image. You need to do this in an interactive cluster session, and assuming you're in the same folder as the Docker .tar file, you can do this:

    salloc --mem=8g --time=00:30:00
    module load singularityce
    singularity build bia-bmz-integration.sif docker-archive://bia-bmz-integration.tar

and you'll have your Singularity container, bia-bmz-integration.sif, waiting for you. 

Running the container in interactive mode is a bit of a faff at the moment — problems with write permission — **and it is much simpler to use Nextflow**, even if you're not planning on running a batch analysis. See *Using Nextflow to run an analysis in the container*, next, for that.

But if you really want to run interactively, you can do this:

    salloc --mem=32g --time=00:30:00
    module load singularityce
    singularity shell bia-bmz-integration.sif

and you'll find yourself in the container shell. 

Best not to use `singularity run` at time of writing — the way Singularity loads the container means that the last line of the Dockerfile, `CMD ["/bin/bash", "-c", "source bia-bmz-integration/bin/activate && exec bash"]`, which activates the python environment, throws an error — Singularity won't find bia-bmz-integration/bin/activate.

Instead, once inside the container shell as above, first go to the root of the container, then bia-bmz-integration, and execute the environment activation from there:

    cd / 
    cd bia-bmz-integration
    source ./bin/activate

but, at this point, you'll need to cd back to wherever you want (and have permission) to write your results to (see *Results*, further below), for example, the `DESTINATION_DIR` you copied everything to earlier.

Once you've done that, it's the same as before: to run a model on an image, or to get benchmarking metrics for the model, run `bia_bmz_benchmark`. For example, view parameters with --help: to  you can run:

    bia_bmz_benchmark --help

see *bia_bmz_benchmark*, further below, for more details.

### *Using Nextflow to run an analysis in the container*

The file `bia_bmz.nf` in this repository is a Nextflow script to run an analysis in the Singularity container described above, and is designed to take inputs from a text file, an example of which is provided here in `batch_input_example.txt`. In this file, an example dataset is subject to two different models, "noisy-fish" and "loyal-squid"; each line of the text file is an input to *bia_bmz_benchmark* (see below), thus allowing you to run an analysis in batch, if you have many model/data combinations.

To run this example analysis in batch on the cluster, you'll need to copy the three files `nextflow.config`, `bia_bmz.nf`, and `batch_input_example.txt`. For example, with rsync:

    rsync -avz nextflow.config USER@CLUSTER:/DESTINATION_DIR

*where, as above, USER should be your username on the cluster, and CLUSTER is the cluster's address, and DESTINATION_DIR is wherever you intend to put the files.*

Note that the scripts here have been written assuming you're running everything from the location you've put the files (i.e. DESTINATION_DIR, above), and moreover, that this location is also where you've put the Singularity container. Thus, you can run a collection of jobs like so:

    salloc --mem=4g --time=00:60:00
    module load nextflow
    nextflow run bia_bmz.nf

and remember, if you want to check your running jobs, if you open another terminal and log in to the cluster, you can use `squeue`.

The results will be saved in `DESTINATION_DIR/results` — see *Results*, further below.

If you have more model/data combinations to run, you can add them to the example input text file, or indeed make your own text file. In the latter case, you'll need to modify the `params.input_file` in `nextflow.config` with the path to the new file, for example:

    params.input_file = './my_new_batch_analysis.txt'

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

With each analysis run, some representative images are saved (see *Results*, below), and among these is the central slice of the input image. For a representative display image, this can occasionally be a bit dim — if you want to adjust its brightness, there is the `--adjust_image` option, which can take the value `auto`, for histogram normalisation (with top 1% of intensities excluded from the calculation, see [here](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#:~:text=PIL.ImageOps.-,autocontrast,-(image%3A))), or `gamma` (with a hardcoded gamma value of 1.5), for gamma correction. For example:

    bia_bmz_benchmark emotional-cricket https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD827/db4b8d36-e0d7-4eac-8298-3a6ba12f7806/91ab07f7-e0d6-4c0d-99a1-6318d25fbd4a.ome.zarr/0 --study_acc S-BIAD827 --dataset_uuid b0cef183-49b0-4911-ad36-b3058e65f272 --channel 0 --z_slices 27 37 --plot_images True --adjust_image gamma
    bia_bmz_benchmark hiding-tiger https://uk1s3.embassy.ebi.ac.uk/bia-integrator-data/S-BIAD1269/a02452b5-3d5e-4d7d-9e51-04e39dad4163/4e270840-d939-4405-837b-a846037877b0.ome.zarr/0 --study_acc S-BIA1269 --dataset_uuid d12a9e60-ad30-4ffe-b993-a3f27b72e38b --plot_images True --adjust_image auto 

## Results

The results of a prediction or benchmarking run are saved in a json file, like this: https://github.com/BioImage-Archive/BIA-astro/blob/main/src/data/ai-dataset-benchmarking/model-dataset-table.json. If an annotation image was provided, then the benchmarking metrics — precision, recall, IoU, and dice — will have values. 

The json file is written to `/results/jsons`, one file per run. If you want to put all your results into one json, you can use the command `amalgamate_jsons`, and this'll create a json `all_results` into the `/results` folder. Note that the Nextflow script also runs `amalgamate_jsons` regardless, so you'll always get `all_results` in that case.

Also note that `amalgamate_jsons` assumes that you're running it from one level above `/results`. If you run the main benchmarking script from the top level of this repository, and then `amalgamate_jsons` without changing folders, all will be well.

Images are also saved with each analysis run — under `/results/images`, there will be a slice of the input image, one of the prediction image, and if an annotation image was provided, the ground truth image, too. 