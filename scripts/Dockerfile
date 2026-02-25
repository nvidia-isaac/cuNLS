FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG CUDA_VERSION=12.8

# install dependencies; CUDA is optional as it is provided by some base images
RUN CUDA_VERSION_DASH=$(echo "${CUDA_VERSION}" | sed 's/\./-/g') && \
    apt-get update && apt-get install -y --allow-change-held-packages \
    apt-utils \
    g++-13 \
    gcc-13 \
    git \
    cmake \
    software-properties-common \
    curl \
    python3 \
    python3-pip && \
    apt-get install -y --allow-change-held-packages \
    libcusparse-dev-${CUDA_VERSION_DASH} \
    libcusolver-dev-${CUDA_VERSION_DASH} \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY docs/sphinx/requirements.txt /tmp/cunls-docs-requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/cunls-docs-requirements.txt

RUN mkdir /cunls
WORKDIR /cunls
