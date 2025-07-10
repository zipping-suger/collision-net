# Use CUDA 12.1 base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    cmake \  
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip and install core packages
RUN pip install --upgrade pip setuptools wheel packaging

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

ENV CUDA_HOME=/usr/local/cuda

# Install torch-scatter (built against PyTorch 2.4.1 + cu121)
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# Install Python dependencies
# Important numpy version for sponv
# Check https://github.com/traveller59/spconv/issues/725 
RUN python3 -m pip install \
    numpy==1.26.4 \ 
    matplotlib \
    geometrout==0.0.3.4 \
    pyquaternion \
    h5py \
    pyyaml \
    sharedarray \
    tensorboardx \
    yapf \
    addict \
    einops \
    scipy \
    plyfile \
    termcolor \
    timm \
    pytorch_lightning \
    trimesh \
    wandb \
    ipython \
    pybullet \
    ikfast-pybind \
    tqdm \
    urchin==0.0.24 \
    yourdfpy \
    shapely \
    rtree \
    triangle \
    meshcat

# Install robofin and spconv
RUN pip install git+https://github.com/fishbotics/robofin.git@v0.0.1 && \
    pip install spconv-cu120

# Install flash-attn dependencies
RUN pip install ninja

# Install flash-attn
RUN pip install flash-attn --no-build-isolation
# RUN git clone https://github.com/Dao-AILab/flash-attention &&\ 
#     cd flash-attention &&\
#     python setup.py install

# Install wanbd
RUN pip install wandb==0.19.11