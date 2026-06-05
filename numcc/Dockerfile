# Jetson (JetPack 6 / L4T R36.4, ARM64, CUDA 12.8)
FROM dustynv/pytorch:2.6-r36.4.0-cu128

WORKDIR /app

# Override Jetson pip index (pypi.jetson-ai-lab.dev doesn't resolve during build)
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=""

# Build tools + OpenGL headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git g++ ninja-build \
    libboost-dev libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── 1. Clone P2C ──────────────────────────────────────────────────────────────
RUN git clone https://github.com/CuiRuikai/Partial2Complete.git /opt/p2c

# ── 2. Clone NU-MCC ───────────────────────────────────────────────────────────
RUN git clone https://github.com/sail-sg/numcc.git /opt/numcc

# ── 3. Python deps ────────────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install P2C deps best-effort (skip packages that conflict with Jetson torch)
RUN if [ -f /opt/p2c/requirements.txt ]; then \
      grep -v -E '^torch|^torchvision' /opt/p2c/requirements.txt \
      | pip install --no-cache-dir -r /dev/stdin || true; \
    fi

# Install NU-MCC deps best-effort
RUN if [ -f /opt/numcc/requirements.txt ]; then \
      grep -v -E '^torch|^torchvision' /opt/numcc/requirements.txt \
      | pip install --no-cache-dir -r /dev/stdin || true; \
    fi

# ── 4. Pipeline modules ───────────────────────────────────────────────────────
COPY pointcloud_utils.py mesh_utils.py sdf_generator.py pipeline.py /app/

# ── 5. Runtime environment ────────────────────────────────────────────────────
# Weights downloaded once on host: python tools/download_models_numcc.py
# Mount at runtime: -v ~/models/numcc:/opt/models:ro
ENV PYTHONPATH=/opt/p2c:/opt/numcc:/app
ENV PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning

ENTRYPOINT ["python3", "/app/pipeline.py"]
