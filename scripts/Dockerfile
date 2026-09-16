ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Use the distro CMake so Ubuntu 22.04 continuously validates the 3.22 minimum.
RUN apt-get update && apt-get install -y --no-install-recommends \
       ca-certificates apt-utils g++ gcc git cmake ccache \
       software-properties-common curl \
       python3 python3-pip python3-dev python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python venv with build tools (for pycunls wheel builds)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install scikit-build-core nanobind build wheel pre-commit

# Sphinx docs dependencies
COPY docs/sphinx/requirements.txt /tmp/cunls-docs-requirements.txt
RUN pip install -r /tmp/cunls-docs-requirements.txt

RUN mkdir -p /cunls
WORKDIR /cunls
