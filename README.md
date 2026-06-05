# Jetson Testing — 3D Asset Pipeline

Converts images into **3D meshes + Drake SDFs** ready for robotic manipulation simulation.
Three pipelines are provided, all running on a Jetson Orin NX (JetPack 6 / L4T R36.4) via Docker.

| Pipeline | Model | Input | Output | Docker image |
|----------|-------|-------|--------|--------------|
| **TripoSR** | Stability AI | 1 image | OBJ + SDF | `triposr` |
| **TRELLIS** | Microsoft | 1 image | GLB + OBJ + SDF | `trellis` |
| **dvlt** | NVIDIA | 1+ images | Gaussian PLY → OBJ | `dvlt:jetson` |

---

## Output structure

```
assets/
└── mug/
    ├── mug.glb              ← textured mesh (TRELLIS only)
    ├── mug.obj              ← normalized mesh (longest axis = 20 cm)
    ├── mug.sdf              ← Drake SDF with inertia + convex collisions
    └── mug_parts/
        ├── convex_piece_000.obj
        └── convex_piece_001.obj

dvlt.cu/meshes/
    └── scene.obj            ← mesh from Gaussian PLY reconstruction
```

---

## Prerequisites

- Jetson Orin NX with JetPack 6 / L4T R36.4, Docker installed
- `ssh jetson` configured (see setup below)
- Models pre-downloaded to host (see each pipeline section)

---

## 1. TripoSR

Reconstructs a 3D mesh from a single image using [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

### Build
```bash
docker build -t triposr -f triposr/Dockerfile triposr/
```

### Download models (once)
```bash
pip3 install huggingface_hub rembg onnxruntime
python3 tools/download_models.py
```

### Run
```bash
python3 tools/generate_asset.py data/images/image.png --name mug
```

### CPU mode (any machine)
```bash
docker build -t triposr:cpu -f triposr/Dockerfile.cpu triposr/
python3 tools/generate_asset.py data/images/image.png --name mug --cpu
```

---

## 2. TRELLIS

Reconstructs a textured 3D mesh using [TRELLIS](https://github.com/microsoft/TRELLIS) (Microsoft).
Build time: ~20–30 min (compiles `spconv` from source for ARM64).

### Build
```bash
docker build -t trellis -f trellis/Dockerfile trellis/
```

### Download models (once)

On Jetson:
```bash
python3 tools/download_models_trellis.py
```

Or faster — download on Windows and transfer:
```powershell
python tools/download_models_trellis_windows.py --token hf_xxxxxxxx
```

### Run
```bash
python3 tools/generate_asset.py data/images/image.png --name mug --model trellis
```

| Flag | Default | Max | Effect |
|------|---------|-----|--------|
| `--steps` | 4 | 12 | Diffusion steps |
| `--texture-size` | 512 | 1024 | GLB texture resolution |

---

## 3. dvlt (Gaussian Splatting)

Reconstructs a 3D Gaussian scene from multiple images using [dvlt](https://github.com/flaviofuego/dvlt.cu).

### Build
```bash
cd dvlt.cu
docker build -f Dockerfile.jetson -t dvlt:jetson .
```

### Setup weights (once)
```bash
mkdir -p ~/dvlt.cu/model
docker run --rm --runtime nvidia -v ~/dvlt.cu/model:/dvlt/model dvlt:jetson --setup
```

### Run
```bash
# Using the wrapper script
./dvlt.cu/run_dvlt.sh ~/Jetson-testing/data/images/taza taza
```

### Convert PLY → OBJ
```bash
python3 tools/ply_to_obj.py ~/dvlt.cu/output/<timestamp>/scene.ply \
  --output ~/dvlt.cu/meshes/object.obj
```

---

## Transferring results to Windows

```powershell
# Assets (TripoSR / TRELLIS)
wsl rsync -av jetson@<IP>:~/Jetson-testing/assets/ /mnt/c/Users/flavi/.../assets/

# dvlt meshes
wsl rsync -av jetson@<IP>:~/dvlt.cu/meshes/ /mnt/c/Users/flavi/.../dvlt.cu/meshes/
```

---

## Cloning

This repo uses `dvlt.cu` as a git submodule:

```bash
git clone --recurse-submodules https://github.com/flaviofuego/Jetson-testing
```

To keep submodules updated automatically:
```bash
git config --global submodule.recurse true
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `permission denied` on Docker socket | `sudo usermod -aG docker $USER` then reconnect SSH |
| `--gpus all` fails on Jetson | Use `--runtime=nvidia` instead |
| Assets owned by root | `sudo chown -R jetson:jetson ~/Jetson-testing/assets` |
| SSH timeout during large transfers | Use `wsl rsync` instead of `scp` |
| Jetson IP changed | `arp -a \| findstr "192.168.1"` in PowerShell |
| pip uses wrong index in Docker build | `ENV PIP_INDEX_URL=https://pypi.org/simple` in Dockerfile |
