FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Suhas Katari Chaluva Kumar <katarichalusuhas@ufl.edu>"

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    wget \
    python3-pyqt5 \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    libpython3-dev \
    python-tk \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-tk \
    software-properties-common \
    libssl-dev \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libmemcached-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECKPOINT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get install libxml2-dev libxslt1-dev -y

WORKDIR /

# Set up Python 3.11 as default and create virtual environment
RUN rm -f /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s $(which python3.11) /usr/bin/python && \
    ln -s $(which python3.11) /usr/bin/python3

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip in virtual environment
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN which python && \
    python --version

ENV build_path=$PWD/build

# HistomicsTK specific
ENV mc_path=$PWD/MultiC
RUN mkdir -p $mc_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . $mc_path/
WORKDIR $mc_path

# Install packages in virtual environment
RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools==69.5.1 && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir cuda-python==12.4.0 && \
    pip install --no-cache-dir tensorboard cmake onnx && \
    pip install --no-cache-dir torch==2.4.0 torchaudio torchvision && \
    rm -rf /root/.cache/pip/*

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    pip install -e detectron2

RUN python --version && pip --version && pip freeze

WORKDIR $mc_path/multic/cli
LABEL entry_path=$mc_path/multic/cli

# Test our entrypoint
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint MultiCompartmentSegment --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]