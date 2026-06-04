# Jetson (JetPack 6 / L4T R36.4, ARM64, CUDA 12.8)
FROM dustynv/pytorch:2.7-r36.4.0

WORKDIR /app

# Override Jetson pip index (pypi.jetson-ai-lab.dev doesn't resolve during build)
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=""

# Build tools + OpenGL headless (required by trimesh, spconv, nvdiffrast)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git g++ ninja-build \
    libboost-dev libgl1 libglib2.0-0 libgomp1 \
    libglfw3-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── 1. spconv — no ARM64 pre-built wheels, build from source (~10 min) ────────
RUN git clone --recursive https://github.com/traveller59/spconv.git /opt/spconv \
    && SPCONV_DISABLE_JIT=1 pip install --no-cache-dir /opt/spconv

# ── 2. nvdiffrast — CUDA rasterizer for GLB texture baking ───────────────────
RUN pip install --no-cache-dir \
    'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git'

# ── 3. Clone TRELLIS (use via PYTHONPATH) ─────────────────────────────────────
RUN git clone https://github.com/microsoft/TRELLIS.git /opt/trellis

# ── 4. Runtime Python deps ────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install TRELLIS deps best-effort (skip flash-attn/xformers — ARM64 incompatible)
RUN grep -v -E 'flash.attn|xformers' /opt/trellis/requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin || true

# ── 5. Pipeline modules ───────────────────────────────────────────────────────
COPY image_utils.py mesh_utils.py sdf_generator.py pipeline.py /app/

# ── 6. Runtime environment ────────────────────────────────────────────────────
# Models are downloaded once on the host and mounted at runtime via:
#   -v ~/models/huggingface:/root/.cache/huggingface
# Run `python3 tools/download_models_trellis.py` on the host first.
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONPATH=/opt/trellis:/app
ENV PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning
ENV TRANSFORMERS_VERBOSITY=error

ENTRYPOINT ["python3", "/app/pipeline.py"]
