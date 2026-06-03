# Jetson Testing — TripoSR Asset Pipeline

Converts a single image (PNG/JPG) into a **3D mesh + Drake SDF** ready for manipulation simulation.
The pipeline uses [TripoSR](https://github.com/VAST-AI-Research/TripoSR) for reconstruction and
`coacd` for convex decomposition. Two Docker targets are provided:

| Target | File | Hardware |
|--------|------|----------|
| `triposr` | `triposr/Dockerfile` | Jetson (JetPack 6, CUDA) |
| `triposr:cpu` | `triposr/Dockerfile.cpu` | Any machine, CPU-only |

---

## Output structure

Running the pipeline for an image named `mug.png` produces:

```
assets/
└── mug/
    ├── mug.obj              ← normalized mesh (longest axis = 20 cm)
    ├── mug.sdf              ← Drake SDF with inertia + convex collisions
    └── mug_parts/
        ├── convex_piece_000.obj
        └── convex_piece_001.obj   ← one file per convex part (coacd)
```

---

## Prerequisites

### Jetson (GPU)
- JetPack 6 / L4T R36
- Docker with `nvidia-container-toolkit`

```bash
# Verify GPU access inside Docker
docker run --rm --gpus all nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3 python3 -c "import torch; print(torch.cuda.is_available())"
```

### CPU (any machine)
- Docker

### Quick tests (no Docker)
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (optional, used as runner)

---

## 1. Build the Docker image

Build once; models (~2 GB) are baked into the image at build time.

**Jetson / GPU:**
```bash
docker build -t triposr -f triposr/Dockerfile triposr/
```

**CPU:**
```bash
docker build -t triposr:cpu -f triposr/Dockerfile.cpu triposr/
```

The build downloads TripoSR weights, DINO-ViT config, and the rembg U²-Net model automatically.
No network access is needed at inference time.

---

## 2. Generate an asset

### Via wrapper script (recommended)

```bash
# GPU (Jetson)
python tools/generate_asset.py path/to/image.png --name mug

# CPU
python tools/generate_asset.py path/to/image.png --name mug --cpu
```

Assets are saved to `assets/<name>/` relative to the repo root.

### Direct Docker call

```bash
docker run --rm --gpus all \
  -v /absolute/path/to/images:/input:ro \
  -v /absolute/path/to/assets:/output \
  triposr \
  --input /input/image.png --output /output --name mug
```

---

## 3. Quick tests (no Docker)

These scripts run TripoSR directly on the host. Useful for fast iteration on a development machine.

### Windows CPU

```bash
# 1. Install dependencies (isolated from the Drake environment)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install trimesh Pillow huggingface_hub rembg

# 2. Clone TripoSR and add it to PYTHONPATH
git clone https://github.com/VAST-AI-Research/TripoSR.git
set PYTHONPATH=%cd%\TripoSR   # Windows
# export PYTHONPATH=$PWD/TripoSR  # Linux/macOS

# 3. Run
python quick_test_cpu_windows.py path/to/image.png --name mug
```

### Linux / CUDA

```bash
pip install torch trimesh Pillow huggingface_hub
git clone https://github.com/VAST-AI-Research/TripoSR.git
export PYTHONPATH=$PWD/TripoSR

python quick_test.py path/to/image.png --name mug --device cuda
```

Output goes to `assets/<name>/` relative to the working directory.

---

## 4. uv environment (Drake integration)

`pyproject.toml` defines the full AIRA environment (Drake + manipulation). The TripoSR dependencies
are **not** listed here because they conflict with Drake's pinned versions — use Docker or a
separate `pip` environment for TripoSR as shown above.

```bash
# Install AIRA/Drake environment
uv sync

# Run any Drake script
uv run python <script>.py
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `torch.cuda.is_available()` → False on Jetson | Wrong base image or missing nvidia runtime | Verify JetPack version matches `l4t-pytorch:r36.2.0` |
| `coacd` not found | Running outside Docker | Expected; pipeline falls back to a single convex hull |
| `ModuleNotFoundError: tsr` | TripoSR not in PYTHONPATH | `export PYTHONPATH=/path/to/TripoSR` |
| Build hangs at model download | Network issue during `docker build` | Retry; or pre-pull with `huggingface-cli download stabilityai/TripoSR` |
| Assets saved to wrong path | Script run from wrong directory | Always run from repo root, or pass `--out` explicitly |
