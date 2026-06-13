# Jetson (JetPack 6 / L4T R36.4, ARM64, CUDA 12.8)
# No multi-stage available — no runtime-only Jetson base image.
# Mitigations: --depth=1 clones, strip .git + build artifacts post-compile,
# chain build+cleanup in single RUN to avoid extra layers.

FROM dustynv/pytorch:2.6-r36.4.0-cu128

WORKDIR /app

# Override Jetson pip index (pypi.jetson-ai-lab.dev doesn't resolve during build)
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=""

# Runtime libs only (libboost-dev only needed if P2C/rtree requires it at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake git g++ ninja-build \
        libboost-dev libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── P2C: shallow clone, compile extensions, strip .git + build dirs ───────────
RUN git clone --depth=1 https://github.com/CuiRuikai/Partial2Complete.git /opt/p2c && \
    pip install --no-cache-dir /opt/p2c/extensions/chamfer_dist && \
    pip install --no-cache-dir /opt/p2c/extensions/pointops && \
    rm -rf /opt/p2c/.git \
           /opt/p2c/extensions/chamfer_dist/build \
           /opt/p2c/extensions/pointops/build

# ── NU-MCC: shallow clone, strip .git ────────────────────────────────────────
RUN git clone --depth=1 https://github.com/sail-sg/numcc.git /opt/numcc && \
    rm -rf /opt/numcc/.git

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── pytorch3d — build from source (no ARM64 wheel available) ──────────────────
# Required by P2C (estimate_pointcloud_normals in ManifoldnessConstraint)
RUN pip install --no-cache-dir \
    'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git' || \
    echo "WARNING: pytorch3d build failed — ManifoldnessConstraint will not work"

# ── Pipeline modules ──────────────────────────────────────────────────────────
COPY pointcloud_utils.py mesh_utils.py sdf_generator.py pipeline.py remesh.py /app/

# Weights downloaded once on host: python tools/download_models_numcc.py
# Mount at runtime: -v ~/models/numcc:/opt/models:ro
ENV PYTHONPATH=/opt/p2c:/opt/numcc:/app
ENV PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning

ENTRYPOINT ["python3", "/app/pipeline.py"]
