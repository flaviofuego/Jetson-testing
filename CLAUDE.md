# Jetson Testing — Asset Pipeline

## Proyecto

Cinco pipelines para generar assets 3D desde imágenes, listos para simulación robótica con Drake. Más un pipeline de estimación de profundidad monocular.

| Pipeline | Modelo | Entrada | Salida |
|----------|--------|---------|--------|
| SAM2 | SAM2.1 small (Meta) | 1 imagen | objeto recortado PNG (fondo blanco) |
| TripoSR | TripoSR (Stability AI) | 1 imagen | OBJ + SDF |
| TRELLIS | TRELLIS-image-large (Microsoft) | 1 imagen | GLB + OBJ + SDF |
| dvlt.cu | DVLT (NVIDIA) | 1+ imágenes | PLY gaussiano → OBJ |
| depth-anything-3 | DA3NESTED-GIANT-LARGE (ByteDance) | 1+ imágenes | depth map + GLB point cloud |

**Flujo típico:** SAM2 → TripoSR o TRELLIS (SAM2 reemplaza a rembg para segmentar el objeto antes de la reconstrucción 3D)

## Hardware / Conexión

- Jetson Orin NX Developer Kit Super, JetPack 6 / L4T R36.4.7, 16 GB RAM unificada
- SSH: `ssh jetson` (usuario: `jetson`, IP dinámica — si falla buscar con `arp -a | findstr "192.168.1"`)
- IP actual: `192.168.1.43` (puede cambiar — DHCP)
- Siempre usar **tmux** en la Jetson para no perder procesos si cae SSH

## Estructura del repo

```
sam2/                   # submodulo fork — segmentación (SAM2.1, Meta)
  Dockerfile.jetson     # GPU ARM64 — base dustynv/pytorch:2.6-r36.4.0-cu128
  Dockerfile.x86        # GPU x86   — base pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
  pipeline.py           # segment → recortar objeto → PNG fondo blanco
depth-anything-3/       # submodulo — estimación de profundidad monocular (DA3, ByteDance 2025)
triposr/
  Dockerfile            # GPU — base dustynv/pytorch:2.6-r36.4.0-cu128
  Dockerfile.cpu        # CPU — base python:3.12-slim
  pipeline.py           # preprocess → TripoSR (FP16) → normalize → coacd → SDF
  image_utils.py
  mesh_utils.py
  sdf_generator.py
  requirements.txt
  requirements.cpu.txt
trellis/
  Dockerfile            # GPU — base dustynv/pytorch:2.7-r36.4.0
  Dockerfile.x86        # GPU x86
  pipeline.py           # preprocess → TRELLIS → GLB → OBJ → coacd → SDF
  image_utils.py
  mesh_utils.py
  sdf_generator.py
  requirements.txt
dvlt.cu/                # submodulo — reconstrucción 3D gaussiana
  run_dvlt.sh           # wrapper para correr dvlt con una carpeta de imágenes
tools/
  generate_asset.py         # wrapper principal (soporta sam2, triposr, trellis, numcc)
  download_models.py        # descarga modelos TripoSR en la Jetson
  download_models_trellis.py        # descarga modelos TRELLIS en la Jetson
  download_models_trellis_windows.py # descarga modelos TRELLIS en Windows + scp a Jetson
  ply_to_obj.py             # convierte PLY gaussiano (dvlt) a OBJ mesh
data/
  images/               # imágenes de entrada
  outputs/              # salidas de SAM2 (viz, máscaras, recortes)
```

## Imágenes Docker

| Tag | Dockerfile | Base | Hardware |
|-----|-----------|------|----------|
| `sam2:x86` | `sam2/Dockerfile.x86` | `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` | x86 GPU |
| `sam2:jetson` | `sam2/Dockerfile.jetson` | `dustynv/pytorch:2.6-r36.4.0-cu128` | Jetson GPU |
| `triposr` | `triposr/Dockerfile` | `dustynv/pytorch:2.6-r36.4.0-cu128` | Jetson GPU |
| `triposr:cpu` | `triposr/Dockerfile.cpu` | `python:3.12-slim` | CPU cualquier máquina |
| `triposr:x86` | `triposr/Dockerfile.x86` | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` | x86 GPU |
| `trellis` | `trellis/Dockerfile` | `dustynv/pytorch:2.7-r36.4.0` | Jetson GPU |
| `trellis:x86` | `trellis/Dockerfile.x86` | - | x86 GPU |
| `dvlt:jetson` | `dvlt.cu/Dockerfile.jetson` | L4T | Jetson GPU |

## Comandos principales

### SAM2 — segmentar objeto

```bash
# x86 (este servidor) — descarga checkpoint la primera vez
mkdir -p ~/models/sam2
cd sam2/checkpoints && bash download_ckpts.sh  # o solo tiny/small

# Segmentar (desde Jetson-testing/)
python3 tools/generate_asset.py data/images/imagen.png --model sam2 --name objeto

# Con todas las máscaras individuales
python3 tools/generate_asset.py data/images/imagen.png --model sam2 --name objeto --all-masks
```

**Outputs en `data/outputs/`:**
- `{nombre}_viz.png` — imagen original + todas las máscaras con bounding boxes
- `{nombre}.png` — objeto principal recortado sobre fondo blanco (listo para TripoSR/TRELLIS)
- `{nombre}_mask{N}.png` — máscaras individuales (con `--all-masks`)

**Configuración elegida:** `sam2.1_hiera_small` + parámetros default del `SAM2AutomaticMaskGenerator`.
Tras comparativa tiny/small/base_plus: small detecta objetos enteros sin fragmentarlos, mismo costo de VRAM (~2.4 GB) y tiempo (~2.4s) que los otros modelos.

**Checkpoints disponibles en `~/models/sam2/`:**

| Modelo | Tamaño | Nota |
|--------|--------|------|
| `sam2.1_hiera_tiny.pt` | 149 MB | más fragmenta |
| `sam2.1_hiera_small.pt` | 176 MB | **usado en pipeline** |
| `sam2.1_hiera_base_plus.pt` | 309 MB | más máscaras, requiere tuning |

**En Jetson** usar `--runtime=nvidia` en lugar de `--gpus all` en el comando docker. El `generate_asset.py` usa `--gpus all` (para x86); para Jetson correr `pipeline.py` directamente dentro del container `sam2:jetson`.

### TripoSR — generar asset
```bash
# Descargar modelos (una sola vez)
python3 tools/download_models.py

# Generar (desde Jetson-testing/)
python3 tools/generate_asset.py data/images/imagen.png --name objeto
```

### TRELLIS — generar asset
```bash
# Descargar modelos en Jetson (una sola vez)
python3 tools/download_models_trellis.py

# Generar (desde Jetson-testing/)
python3 tools/generate_asset.py data/images/imagen.png --name objeto --model trellis
```

### dvlt — reconstrucción gaussiana
```bash
# Setup (una sola vez)
mkdir -p ~/dvlt.cu/model
docker run --rm --runtime nvidia -v ~/dvlt.cu/model:/dvlt/model dvlt:jetson --setup

# Correr con carpeta de imágenes
cd ~/Jetson-testing
./dvlt.cu/run_dvlt.sh ~/Jetson-testing/data/images/taza taza
```

### Depth Anything 3 — depth map + reconstrucción 3D

DA3 está instalado como submódulo en `depth-anything-3/`. El CLI `da3` queda disponible en el venv de UniWhere (`.venv`). El modelo por defecto es `depth-anything/DA3NESTED-GIANT-LARGE-1.1` (~descarga automática desde HF la primera vez).

```bash
# Activar venv de UniWhere donde está instalado
source /home/worker-node-4/Documents/GitHub/UniWhere/.venv/bin/activate

# Depth map visualizado (imagen side-by-side)
da3 image data/images/objeto.jpg \
    --export-format depth_vis \
    --export-dir data/outputs/depthmaps/nombre \
    --auto-cleanup

# Point cloud 3D (GLB) — ajustar conf-thresh-percentile para más/menos puntos
# Valor bajo (5) = más puntos (incluye zonas de baja confianza)
# Valor alto (40, default) = menos puntos pero más limpios
da3 image data/images/objeto.jpg \
    --export-format glb \
    --export-dir data/outputs/depthmaps/nombre_glb \
    --auto-cleanup \
    --conf-thresh-percentile 5.0

# Formatos disponibles: depth_vis, glb, npz, mini_npz, colmap, feat_vis, gs_ply, gs_video
# Combinar con guión: --export-format depth_vis-glb
```

**Outputs:**
- `depth_vis/0000.jpg` — imagen original + depth colorizado (inferno)
- `scene.glb` — point cloud 3D con colores
- `scene.jpg` — preview side-by-side

**Limitaciones imagen única:** solo reconstruye superficies visibles desde esa cámara. Interiores de objetos (ej. fondo de taza) no se reconstruyen. Para eso usar TripoSR o TRELLIS.

**Bugs corregidos en el CLI de DA3** (`depth-anything-3/src/depth_anything_3/cli.py`):
- `reference_view_strategy` → `ref_view_strategy` en llamadas a `run_inference()` (4 ocurrencias, líneas ~383, 462, 549, 626)

**Depth Anything V2** también disponible via `transformers` sin instalación manual:
```bash
# Script standalone (no requiere clonar repo)
uv run /tmp/depth_anything.py input.jpg output_depthmap.png
```

### PLY → OBJ (post-proceso dvlt)
```bash
python3 tools/ply_to_obj.py ~/dvlt.cu/output/<timestamp>/scene.ply --output ~/dvlt.cu/meshes/nombre.obj
```

### Transferir resultados a Windows
```powershell
# Assets TripoSR/TRELLIS
wsl rsync -av jetson@192.168.1.43:~/Jetson-testing/assets/ /mnt/c/Users/flavi/.../assets/

# Meshes dvlt
wsl rsync -av jetson@192.168.1.43:~/dvlt.cu/meshes/ /mnt/c/Users/flavi/.../dvlt.cu/meshes/
```

## Modelos y caché

Los modelos se guardan en el host y se montan en Docker:

| Modelo | Ubicación | Tamaño |
|--------|-----------|--------|
| SAM2.1 small | `~/models/sam2/` | 176 MB |
| TripoSR weights | `~/models/huggingface/` | ~1.7 GB |
| DINO-ViT config | `~/models/huggingface/` | - |
| rembg U2-Net | `~/.u2net/` | ~176 MB |
| TRELLIS-image-large | `~/models/huggingface/` | ~3 GB |
| dvlt weights | `~/dvlt.cu/model/` | ~468 MB |

## Optimizaciones activas (TripoSR)

- **FP16** en GPU (Ampere, nativo en Orin)
- **chunk_size = 262144** (16 GB RAM unificada)
- `sudo nvpmodel -m 0 && sudo jetson_clocks` antes de inferencia

## Notas importantes

- Docker usa `--runtime=nvidia` (no `--gpus all`) en esta Jetson
- Los assets generados por Docker son propiedad de root — usar `sudo chown -R jetson:jetson ~/Jetson-testing/assets` si hay problemas de permisos
- El entorno `pyproject.toml` (uv/Drake) es independiente — NO incluye TripoSR ni TRELLIS (conflictos de versiones)
- `dvlt.cu`, `depth-anything-3` y `sam2` son submódulos git — clonar con `git clone --recurse-submodules`
- `git config --global submodule.recurse true` para que pull/fetch actualice submódulos automáticamente
- `sam2` es un fork de `facebookresearch/sam2` en `cristian10gf/sam2`
- DA3 instalado en el venv de UniWhere (`/home/worker-node-4/Documents/GitHub/UniWhere/.venv`), no tiene venv propio
- xformers instalado pero sin extensiones CUDA (torch 2.12 vs xformers compilado para 2.10) — funciona igual, solo sin memory-efficient attention
- La IP de la Jetson cambia por DHCP — pendiente configurar IP estática en el router
- SAM2 extensión CUDA (`sam2._C`) no compiló en la imagen x86 actual — funciona igual, solo sin post-procesado de huecos (no afecta resultados en la mayoría de casos)
