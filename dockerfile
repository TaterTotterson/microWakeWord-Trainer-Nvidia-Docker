# Standard Ubuntu base image.  CUDA base images not needed.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# System deps (+dev headers for building C/C++ extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3.10-dev python3-pip \
    git wget curl unzip ca-certificates git-lfs \
    build-essential g++ cmake \
    libsndfile1 libsndfile1-dev libffi-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Use python3.10 everywhere
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ---- No cuDNN repo meddling needed if using TF 2.17.x ----

# Python deps
# Order is important. onnxruntime, tensorflow and torch have
# to be installed in the order below or their cuda dependencies
# will conflict.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install "numpy==1.26.4" "cython>=0.29.36" \
 && pip install -r /tmp/requirements.txt \
 && pip install "onnxruntime-gpu[cuda]>=1.16.0" \
 && pip install "tensorflow[and-cuda]==2.18.0" \
    "tensorboard==2.18.0" \
    "tensorboard-data-server==0.7.2" \
    "tensorflow-io-gcs-filesystem==0.37.1" \
 && pip install \
      torch==2.7.1 \
      torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu128

# Workspace + notebook fallback
RUN mkdir -p /data
WORKDIR /data
COPY microWakeWord_training_notebook.ipynb /root/

# Startup script (copies default notebook if missing)
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

EXPOSE 8888

CMD ["/bin/bash", "-lc", "/usr/local/bin/startup.sh && \
     exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir=/data"]
