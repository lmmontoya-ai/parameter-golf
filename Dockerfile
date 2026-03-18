FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        openssh-server \
        rsync \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh \
    && printf 'PermitRootLogin yes\nPasswordAuthentication no\nPubkeyAuthentication yes\n' >> /etc/ssh/sshd_config

WORKDIR /workspace/parameter-golf

COPY requirements.txt /tmp/requirements.txt
RUN grep -v '^torch==' /tmp/requirements.txt > /tmp/requirements-no-torch.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-no-torch.txt

COPY . /workspace/parameter-golf
RUN chmod +x /workspace/parameter-golf/docker/start.sh

EXPOSE 22

ENTRYPOINT ["/workspace/parameter-golf/docker/start.sh"]
CMD ["bash"]
