# Since this is a pure python environment, we don't need to start
# with a huge CUDA image.  A standard Ubuntu image will do.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PATH="/root/mww-scripts:${PATH}"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip python-is-python3 \
    git wget curl unzip ca-certificates nano less \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data

COPY --chown=root:root --chmod=0755 .bashrc /root/
COPY --chown=root:root --chmod=0755 setup_* wake_word_sample* train_wake_word \
        test_python cudainfo system_summary shell.functions requirements.txt /root/mww-scripts/

# Docker and Podman send the CMD a SIGTERM when you "stop" the container.  Unfortunately, bash
# normally doesn't exit when it recieves a SIGTERM so docker/podman has to wait for the "stop"
# to timeout then SIGKILL the container.
# This little scriptlet causes bash to exit immediately when it receives the SIGTERM.
CMD ["/usr/bin/bash", "-c", "exec /usr/bin/bash --rcfile <(echo '[ -f ~/.bashrc ] && source ~/.bashrc ; trap exit SIGTERM ;')" ]
