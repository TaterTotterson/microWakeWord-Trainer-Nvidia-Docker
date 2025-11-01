FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates git-lfs \
    libsndfile1 libsndfile1-dev libffi-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip install --no-cache-dir "tensorflow[and-cuda]==2.18.0"

WORKDIR /app
COPY app.py /app/app.py
COPY .streamlit /app/.streamlit

# persistent work dir
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8502

CMD ["streamlit", "run", "/app/app.py"]