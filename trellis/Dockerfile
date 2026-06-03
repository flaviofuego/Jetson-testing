# Jetson (JetPack 6.1 / L4T R36.4, ARM64, CUDA 12.8)
# Uses dusty-nv/jetson-containers PyTorch 2.7 image — natively satisfies TRELLIS torch>=2.4
# https://hub.docker.com/r/dustynv/pytorch/tags
FROM dustynv/pytorch:2.7-r36.4.0

WORKDIR /app

# Build tools + OpenGL headless (required by trimesh and spconv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git g++ ninja-build \
    libboost-dev libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── 1. spconv — no ARM64 pre-built wheels, must build from source ─────────────
RUN git clone --recursive https://github.com/traveller59/spconv.git /opt/spconv \
    && SPCONV_DISABLE_JIT=1 pip install --no-cache-dir /opt/spconv

# ── 2. Clone TRELLIS (no pyproject.toml install — use via PYTHONPATH) ─────────
RUN git clone https://github.com/microsoft/TRELLIS.git /opt/trellis

# ── 3. Runtime Python deps ────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install remaining TRELLIS-specific deps from its own requirements (best-effort)
RUN pip install --no-cache-dir -r /opt/trellis/requirements.txt || true

# ── 4. Pipeline modules ───────────────────────────────────────────────────────
COPY image_utils.py mesh_utils.py sdf_generator.py pipeline.py /app/

# ── 5. Pre-download all models at build time (zero network at inference) ──────
ENV HF_HOME=/opt/models/huggingface
RUN python3 - <<'EOF'
from huggingface_hub import snapshot_download
print("Downloading TRELLIS-image-large weights (~3 GB)...")
snapshot_download("JeffreyXiang/TRELLIS-image-large")
print("Downloading rembg U²-Net model (~176 MB)...")
import rembg
rembg.new_session()
print("All models pre-downloaded successfully.")
EOF

# ── 6. Runtime environment ────────────────────────────────────────────────────
ENV PYTHONPATH=/opt/trellis:/app
ENV PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning
ENV TRANSFORMERS_VERBOSITY=error

ENTRYPOINT ["python3", "/app/pipeline.py"]
