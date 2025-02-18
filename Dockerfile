# Build-stage image
FROM python:3.10-slim AS build

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    make \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Verify gcc installation
RUN gcc --version

# Install poetry
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.in-project true

# Set workdir
WORKDIR /app

# Copy only dependency files first to cache dependencies
COPY pyproject.toml poetry.lock* README.md ./

# Configure pip to use system gcc
ENV CC=gcc
ENV CXX=g++

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the application
COPY ./bia_bmz_integrator ./bia_bmz_integrator

# Install the project itself
RUN poetry install

# Runtime-stage image
FROM python:3.10-slim AS runtime

# Copy only the built virtual environment and application code
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/bia_bmz_integrator /app/bia_bmz_integrator

# Set workdir
WORKDIR /app

# Set PATH to use the virtualenv by default
ENV PATH="/app/.venv/bin:$PATH"

# Command to run the application
CMD ["bash"]