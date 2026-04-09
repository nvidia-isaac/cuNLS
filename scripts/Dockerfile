ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# CMake from Kitware APT (system cmake on Ubuntu 22.04 is 3.22, project needs >= 3.24)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gpg wget \
    && wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
       | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" \
       > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --allow-change-held-packages \
       apt-utils g++ gcc git cmake ccache \
       software-properties-common curl \
       python3 python3-pip python3-dev python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python venv with build tools (for pycunls wheel builds)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install scikit-build-core nanobind build wheel

# Sphinx docs dependencies
COPY docs/sphinx/requirements.txt /tmp/cunls-docs-requirements.txt
RUN pip install -r /tmp/cunls-docs-requirements.txt

RUN mkdir -p /cunls
WORKDIR /cunls
