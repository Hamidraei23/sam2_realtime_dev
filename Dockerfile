ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
ARG MODEL_SIZE=base_plus

FROM ${BASE_IMAGE}

# ---- Runtime env ----
ENV GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=2 \
    GUNICORN_PORT=5000 \
    APP_ROOT=/opt/sam2 \
    PYTHONUNBUFFERED=1 \
    SAM2_BUILD_CUDA=0 \
    MODEL_SIZE=${MODEL_SIZE} \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
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
 && rm -rf /var/lib/apt/lists/*

# Make app directory structure early (including checkpoints)
RUN mkdir -p ${APP_ROOT}/checkpoints

# Copy ONLY metadata first if you want caching (safe even if you later copy all)
# If these files don't exist in your repo, remove these two lines.
COPY setup.py README.md ./

RUN pip install --upgrade pip setuptools wheel

# Install OpenCV (cv2) pinned to 4.13.0
# (Use this exact version string for pip wheels)
RUN pip install --no-cache-dir opencv-python==4.13.0.90

# Copy the repo content needed for editable install + server runtime
# (This is the "works normally" move: everything needed is in the image at install time.)
COPY . ${APP_ROOT}/src

# Install Python package (editable) from the copied repo
WORKDIR ${APP_ROOT}/src
RUN pip install -e ".[interactive-demo]"

# ffmpeg fix (only if conda ffmpeg exists)
RUN if [ -e /opt/conda/bin/ffmpeg ]; then rm -f /opt/conda/bin/ffmpeg; ln -sf /bin/ffmpeg /opt/conda/bin/ffmpeg; fi

# Copy backend server + sam2 into final runtime layout
# (These paths assume your repo contains them exactly as in your snippet)
RUN mkdir -p ${APP_ROOT}/server
RUN cp -a ${APP_ROOT}/src/demo/backend/server/. ${APP_ROOT}/server/ \
 && cp -a ${APP_ROOT}/src/sam2 ${APP_ROOT}/server/sam2

# Download SAM 2.1 checkpoints
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_tiny.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_small.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt
ADD https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt ${APP_ROOT}/checkpoints/sam2.1_hiera_large.pt

WORKDIR ${APP_ROOT}/server
EXPOSE 5000

# Gunicorn entrypoint
CMD gunicorn --worker-tmp-dir /dev/shm \
    --worker-class gthread app:app \
    --log-level info \
    --access-logfile /dev/stdout \
    --log-file /dev/stderr \
    --workers ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    --bind 0.0.0.0:${GUNICORN_PORT} \
    --timeout 60

