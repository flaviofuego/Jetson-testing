# ARM64 base with PyTorch pre-built for Jetson (JetPack 6 / L4T R36.4)
FROM dustynv/pytorch:2.6-r36.4.0-cu128

WORKDIR /app

# Build tools for torchmcubes and coacd (C++ + CUDA on ARM64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git g++ ninja-build \
    libboost-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# ── 1. TripoSR (no setup.py/pyproject.toml — use via PYTHONPATH) ────────────
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /opt/triposr

# ── 2. torchmcubes — built against L4T torch (CUDA available on Jetson) ─────
RUN TORCH_DIR=$(python3 -c \
      "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share/cmake/Torch'))") \
    && CMAKE_ARGS="-DTorch_DIR=${TORCH_DIR}" \
       pip install --no-cache-dir \
       'torchmcubes @ git+https://github.com/tatsy/torchmcubes.git@3381600ddc3d2e4d74222f8495866be5fafbace4'

# ── 3. Runtime deps ──────────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── 4. Pipeline modules ──────────────────────────────────────────────────────
COPY image_utils.py mesh_utils.py sdf_generator.py pipeline.py /app/

# ── 5. Pre-download all models at build time (zero network at inference) ─────
ENV HF_HOME=/opt/models/huggingface
RUN python3 - <<'EOF'
from huggingface_hub import hf_hub_download
print("Downloading TripoSR weights (~1.7 GB)...")
hf_hub_download("stabilityai/TripoSR", "config.yaml")
hf_hub_download("stabilityai/TripoSR", "model.ckpt")
print("Downloading DINO-ViT config...")
hf_hub_download("facebook/dino-vitb16", "config.json")
print("Downloading rembg U²-Net model (~176 MB)...")
import rembg
rembg.new_session()
print("All models pre-downloaded successfully.")
EOF

# ── 6. Runtime environment ───────────────────────────────────────────────────
ENV PYTHONPATH=/opt/triposr:/app
ENV PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning
ENV TRANSFORMERS_VERBOSITY=error

ENTRYPOINT ["python3", "/app/pipeline.py"]
