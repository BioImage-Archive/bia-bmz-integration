# build-stage image
FROM continuumio/miniconda3 AS build

# Python output no buffered 
ENV PYTHONUNBUFFERED=1

# set workdir
WORKDIR /BIA-BMZ

# Copy files
COPY env.yaml pyproject.toml README.md /BIA-BMZ/
COPY ./bia_bmz_integrator /BIA-BMZ/bia_bmz_integrator

# Conda env and conda-pack / -unpack
RUN conda env create -f /BIA-BMZ/env.yaml && \
    conda install -c conda-forge --override-channels conda-pack && \
    conda-pack -n bia-bmz-integration -o /tmp/env.tar && \
    mkdir /bia-bmz-integration && cd /bia-bmz-integration && tar xf /tmp/env.tar && \
    rm /tmp/env.tar && \
    /bia-bmz-integration/bin/conda-unpack && \
    find /bia-bmz-integration -type f -name '*.a' -delete && \
    rm -rf /bia-bmz-integration/pkgs && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs

# Runtime-stage image
FROM debian:bookworm-slim AS runtime

# Copy /venv from the previous stage:
COPY --from=build /bia-bmz-integration /bia-bmz-integration

# Again, python output not buffered
# And no .pyc files generated
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Default to bash and run activation of environment
CMD ["/bin/bash", "-c", "source bia-bmz-integration/bin/activate && exec bash"]
