FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Suhas Katari Chaluva Kumar <katarichalusuhas@ufl.edu>"

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Set architecture-related environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TARGET_ARCH=amd64
ENV HOST_ARCH=amd64

RUN dpkg --add-architecture amd64 && \
    apt-get update && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-tk \
    build-essential \
    cmake \
    libxml2-dev \
    libxslt-dev \
    gcc \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && python3 -m pip install lxml==5.2.1 &&\
    # ffmpeg \
    # libsm6 \
    # libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 


WORKDIR /

# Set up Python 3.11 as default
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s $(which python3.11) /usr/bin/python && \
    ln -s $(which python3.11) /usr/bin/python3

# Install pip in virtual environment
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Verify Python installation
RUN which python && \
    python --version

ENV build_path=$PWD/build
ENV mc_path=$PWD/MultiC
RUN mkdir -p $mc_path

COPY . $mc_path/
WORKDIR $mc_path
RUN export CFLAGS="-Wno-incompatible-function-pointer-types -Wno-implicit-function-declaration"


# Install packages in virtual environment with system lxml
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Try to install a specific version of lxml that's known to work
    pip install --no-cache-dir . && \
    pip install --no-cache-dir cuda-python==12.4.0 && \
    pip install --no-cache-dir tensorboard cmake onnx && \
    pip install --no-cache-dir torch==2.4.0 torchaudio torchvision && \
    rm -rf /root/.cache/pip/*

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    pip install -e detectron2

# Verify installations
RUN python --version && pip --version && pip freeze

WORKDIR $mc_path/multic/cli
LABEL entry_path=$mc_path/multic/cli

# Test our entrypoint
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint MultiCompartmentSegment --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]