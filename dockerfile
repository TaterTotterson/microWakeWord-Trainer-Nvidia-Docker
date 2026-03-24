# Base — CUDA runtime for GPU support on RunPod and similar platforms
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip python-is-python3 \
    git wget curl unzip ca-certificates nano less \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data

# Script root
WORKDIR /root/mww-scripts

# Bash environment
COPY --chown=root:root --chmod=0755 .bashrc /root/

# Root-level scripts
COPY --chown=root:root --chmod=0755 \
    train_wake_word \
    entrypoint.sh \
    github_push.sh \
    requirements.txt \
    /root/mww-scripts/

# CLI folder
COPY --chown=root:root cli/ /root/mww-scripts/cli/

# Make all CLI scripts executable (avoids "Permission denied")
RUN chmod -R a+x /root/mww-scripts/cli

# No args = interactive bash shell; "train <wake_word>" = full pipeline
ENTRYPOINT ["/root/mww-scripts/entrypoint.sh"]
CMD ["/bin/bash", "-l"]
