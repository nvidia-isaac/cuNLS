ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
# Jetson Orin (JetPack 6) needs an L4T-compatible CUDA image; the nvidia/cuda
# arm64 images target SBSA servers and cannot use the Orin iGPU.
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_PORTS_MIRROR=""

COPY scripts/ubuntu-ports-mirror.conf /tmp/ubuntu-ports-mirror.conf

RUN if [ -n "${UBUNTU_PORTS_MIRROR}" ]; then \
      set -eux; \
      find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
          -exec sed -Ei "s#https?://ports\\.ubuntu\\.com/ubuntu-ports#${UBUNTU_PORTS_MIRROR}#g" {} +; \
      install -m 0644 /tmp/ubuntu-ports-mirror.conf /etc/apt/apt.conf.d/99-nvidia-ubuntu-ports-mirror; \
    fi && \
    rm -f /tmp/ubuntu-ports-mirror.conf

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
