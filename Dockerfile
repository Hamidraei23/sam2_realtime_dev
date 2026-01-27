ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ARG MODEL_SIZE=base_plus
ARG ROS_DISTRO=humble

FROM ${BASE_IMAGE}

ARG MODEL_SIZE
ARG ROS_DISTRO

ENV DEBIAN_FRONTEND=noninteractive \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=2 \
    GUNICORN_PORT=5000 \
    APP_ROOT=/opt/sam2 \
    PYTHONUNBUFFERED=1 \
    SAM2_BUILD_CUDA=0 \
    MODEL_SIZE=${MODEL_SIZE} \
    ROS_DISTRO=${ROS_DISTRO} \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CONDA_AUTO_ACTIVATE_BASE=false

# ---- System deps + GUI deps for cv2.imshow + Python 3.10 venv tooling ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    pkg-config \
    build-essential \
    libffi-dev \
    ca-certificates \
    curl \
    \
    # OpenCV/Qt (cv2.imshow) runtime deps
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libx11-xcb1 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    \
    # ROS prerequisites
    locales \
    software-properties-common \
    gnupg2 \
    lsb-release \
 && locale-gen en_US.UTF-8 \
 && add-apt-repository universe \
 && rm -rf /var/lib/apt/lists/*

# ---- Create a Python 3.10 venv (ROS Humble rclpy ABI) ----
RUN /usr/bin/python3.10 -m venv --system-site-packages /opt/venv \
 && /opt/venv/bin/python -m pip install --no-cache-dir -U pip setuptools wheel

# Force the venv to be the default python everywhere (build + runtime)
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Also ensure interactive shells keep venv first (even if conda tries)
RUN sed -i '/conda\.sh/d;/conda activate/d' /root/.bashrc 2>/dev/null || true \
 && echo 'export PATH=/opt/venv/bin:$PATH' >> /etc/bash.bashrc \
 && echo 'export PATH=/opt/venv/bin:$PATH' >> /root/.bashrc

# Make app directory structure early (including checkpoints)
RUN mkdir -p ${APP_ROOT}/checkpoints

# Copy ONLY metadata first if you want caching
COPY setup.py README.md ./

# Copy the repo content needed for editable install + server runtime
COPY . ${APP_ROOT}/src

# Install Python package (editable) from the copied repo (into the venv)
WORKDIR ${APP_ROOT}/src
RUN /opt/venv/bin/pip install --no-cache-dir -e ".[interactive-demo]"

# Install OpenCV (cv2) pinned (into the venv)
RUN /opt/venv/bin/pip install --no-cache-dir opencv-python==4.13.0.90

# ---- Install ROS 2 (AFTER interactive-demo install) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}}) main" \
    > /etc/apt/sources.list.d/ros2.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-image-transport \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-vision-opencv \
      ros-${ROS_DISTRO}-image-tools \
      ros-${ROS_DISTRO}-rqt-image-view \
      ros-dev-tools \
 && rm -rf /var/lib/apt/lists/*

# Auto-source ROS 2 setup for every new bash terminal
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc \
 && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

# Install torch + CUDA 12.1 wheels INTO THE SAME PYTHON AS rclpy (the venv)
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ffmpeg fix (only if conda ffmpeg exists)
RUN if [ -e /opt/conda/bin/ffmpeg ]; then rm -f /opt/conda/bin/ffmpeg; ln -sf /bin/ffmpeg /opt/conda/bin/ffmpeg; fi

# Copy backend server + sam2 into final runtime layout
RUN mkdir -p ${APP_ROOT}/server
RUN cp -a ${APP_ROOT}/src/demo/backend/server/. ${APP_ROOT}/server/ \
 && cp -a ${APP_ROOT}/src/sam2 ${APP_ROOT}/server/sam2

# Download SAM 2.1 checkpoints
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_tiny.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_large.pt


ARG USERNAME=hami
ARG USER_UID=1000
ARG USER_GID=1000

RUN set -eux; \
    # create primary group + user
    groupadd -g "${USER_GID}" "${USERNAME}"; \
    useradd -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USERNAME}"; \
    \
    # ensure device-access groups exist (render may not)
    getent group video  >/dev/null || groupadd -r video; \
    getent group render >/dev/null || groupadd -r render; \
    \
    # add user to groups
    usermod -aG video,render "${USERNAME}"; \
    \
    mkdir -p "/home/${USERNAME}/.config"; \
    chown -R "${USERNAME}:${USER_GID}" "/home/${USERNAME}"

# Make ROS + venv available in that user's shell too
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc \
 && echo "export PATH=/opt/venv/bin:\$PATH" >> /home/${USERNAME}/.bashrc \
 && chown ${USERNAME}:${USER_GID} /home/${USERNAME}/.bashrc

WORKDIR ${APP_ROOT}/server
EXPOSE 5000

# Source ROS for the main process too (not just interactive shells)
RUN install -m 0755 /dev/stdin /usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -e
source /opt/ros/${ROS_DISTRO}/setup.bash
exec gunicorn --worker-tmp-dir /dev/shm \
  --worker-class gthread app:app \
  --log-level info \
  --access-logfile /dev/stdout \
  --log-file /dev/stderr \
  --workers "${GUNICORN_WORKERS}" \
  --threads "${GUNICORN_THREADS}" \
  --bind "0.0.0.0:${GUNICORN_PORT}" \
  --timeout 60
EOF

CMD ["/usr/local/bin/entrypoint.sh"]

