# Build-stage image
FROM continuumio/miniconda3 AS build

WORKDIR /BIA-BMZ

COPY env.yaml pyproject.toml README.md setup_env.sh /BIA-BMZ/
COPY ./bia_bmz_integrator /BIA-BMZ/bia_bmz_integrator

# This script sets up a standalone environment in /bia-bmz-integration
RUN chmod +x /BIA-BMZ/setup_env.sh && /BIA-BMZ/setup_env.sh

# Runtime-stage image
FROM debian:bookworm-slim AS runtime

# Copy over the standalone environment:
COPY --from=build /bia-bmz-integration /bia-bmz-integration

# Default to bash and run activation of environment
CMD ["/bin/bash", "-c", "source bia-bmz-integration/bin/activate && exec bash"]