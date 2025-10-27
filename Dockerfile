FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System Dependencies and OpenSlide.
RUN apt-get update && apt-get install -y \
    # Install OpenSlide.
    openslide-tools \
    # libopenslide0 \
    # Install gcc, g++, make, entre autres.
    build-essential \
    git \
    # Don't need package index files anymore.
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Copy dependency files first for layer caching.
COPY main.py pyproject.toml uv.lock ./
# Install dependencies.
RUN uv sync --frozen --no-dev

# Copy files over.
COPY configs configs/
COPY experiments experiments/
COPY scripts scripts/
COPY ssregpros ssregpros/
COPY tests tests/
COPY main.py .

# Pass Git commit SHA at build time.
ARG SSREGPROS_LATEST_GIT_COMMIT_SHA=""
ENV SSREGPROS_LATEST_GIT_COMMIT_SHA=${SSREGPROS_LATEST_GIT_COMMIT_SHA}

# Set Python path.
ENV PYTHONPATH=/workspace

CMD ["bash"]