# CUDA + cuDNN userspace from NVIDIA (no manual repo installs needed)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    git wget curl unzip ca-certificates \
    build-essential g++ cmake \
    libsndfile1 libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 everywhere
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Workspace + notebook fallback
RUN mkdir -p /data
WORKDIR /data
COPY microWakeWord_training_notebook.ipynb /root/

# Startup script (copies default notebook if missing, then launches JupyterLab)
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

EXPOSE 8888

# Launch Lab (tokenless for local dev; set a token if you want auth)
CMD ["/bin/bash", "-lc", "/usr/local/bin/startup.sh && \
     exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir=/data"]