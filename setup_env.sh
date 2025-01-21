#!/bin/bash

# -e exits upon failure, -x prints lines
set -e
set -x

# conda-pack create a standalone environment from the original
conda env create -f /BIA-BMZ/env.yaml
conda install -c conda-forge --override-channels conda-pack
conda-pack -n bia-bmz-integration -o /tmp/env.tar
mkdir /bia-bmz-integration
cd /bia-bmz-integration
tar xf /tmp/env.tar
rm /tmp/env.tar

# now unpack the environment and tidy up
/bia-bmz-integration/bin/conda-unpack
find /bia-bmz-integration -type f -name '*.a' -delete
find /bia-bmz-integration -name '*.pyc' -delete
find /bia-bmz-integration -name '__pycache__' -delete
rm -rf /bia-bmz-integration/share/doc
rm -rf /bia-bmz-integration/pkgs
conda clean -afy
rm -rf /opt/conda/pkgs
