# Jetson Testing — Asset Pipeline

## Proyecto

Seis pipelines para generar assets 3D desde imágenes o datos RGBD, listos para simulación robótica con Drake. Más un pipeline de estimación de profundidad monocular.

| Pipeline | Modelo | Entrada | Salida |
|----------|--------|---------|--------|
| SAM2 | SAM2.1 small (Meta) | 1 imagen | objeto recortado PNG (fondo blanco) |
| TripoSR | TripoSR (Stability AI) | 1 imagen | OBJ + SDF |
| TRELLIS | TRELLIS-image-large (Microsoft) | 1 imagen | GLB + OBJ + SDF |
| **numcc** | **P2C + NU-MCC (CO3D-V2)** | **depth map + color (RGBD)** | **OBJ + SDF** |
| dvlt.cu | DVLT (NVIDIA) | 1+ imágenes | PLY gaussiano → OBJ |
| depth-anything-3 | DA3NESTED-GIANT-LARGE (ByteDance) | 1+ imágenes | depth map + GLB point cloud |

**Flujo típico imagen:** SAM2 → TripoSR o TRELLIS (SAM2 reemplaza a rembg para segmentar el objeto antes de la reconstrucción 3D)

**Flujo RGBD (cámara de profundidad / simulación Drake):** depth + color → numcc (P2C + NU-MCC) → OBJ + SDF

## Hardware / Conexión

- Jetson Orin NX Developer Kit Super, JetPack 6 / L4T R36.4.7, 16 GB RAM unificada
- SSH: `ssh jetson` (usuario: `jetson`, IP dinámica — si falla buscar con `arp -a | findstr "192.168.1"`)
- IP actual: `192.168.1.43` (puede cambiar — DHCP)
- Siempre usar **tmux** en la Jetson para no perder procesos si cae SSH

## Estructura del repo

```
submodules/
  sam2/                 # submodulo fork — segmentación (SAM2.1, Meta)
    Dockerfile.jetson   # GPU ARM64 — base dustynv/pytorch:2.6-r36.4.0-cu128
    Dockerfile.x86      # GPU x86   — base pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
    pipeline.py         # segment → recortar objeto → PNG fondo blanco
  depth-anything-3/     # submodulo — estimación de profundidad monocular (DA3, ByteDance 2025)
  dvlt.cu/              # submodulo — reconstrucción 3D gaussiana
    run_dvlt.sh         # wrapper para correr dvlt con una carpeta de imágenes
  nksr/                 # submodulo nv-tlabs/nksr — Neural Kernel Surface Reconstruction
models/
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
  numcc/
    Dockerfile.x86        # GPU x86 — base pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
    pipeline.py           # depth → P2C → NU-MCC → surface pts → mesh → coacd → SDF
    remesh.py             # mesh-only desde PLY existente (salta NU-MCC) — corre dentro de Docker
    pointcloud_utils.py   # back-projection depth → nube de puntos
    mesh_utils.py
    sdf_generator.py
    requirements.txt
tools/
  pipeline.py                    # super-pipeline unificado: imagen → output/ estructurado por stages
  generate_asset.py              # wrapper simple (triposr, trellis, sam2, numcc) — casos básicos
  run_numcc.py                   # wrapper numcc: full pipeline o solo remesh, elige nksr/poisson
  capture_realsense.py           # captura RGBD desde Intel RealSense → depth.npy + color.npy + intrinsics.json
  download_models.py             # descarga modelos TripoSR en la Jetson
  download_models_trellis.py     # descarga modelos TRELLIS en la Jetson
  download_models_numcc.py       # descarga pesos P2C (Google Drive) y NU-MCC (S3)
  ply_to_obj.py                  # convierte PLY gaussiano (dvlt) a OBJ mesh
  segment_objects.py             # segmenta escena multi-objeto PLY (dvlt) → OBJ por cluster
  visualize_aira_npy.py          # debug: convierte .npy AIRA a PNG + PLY visualizable
  mesh_filter.py                 # filtros MeshLab individuales (smooth normals, depth smooth, decimation)
  smooth_preserve.py             # suavizado preservando geometría (Taubin + Two Steps) — recomendado
  apply_mlx.py                   # aplica script .mlx exportado desde GUI MeshLab
data/
  images/               # imágenes de entrada
  outputs/              # salidas de SAM2 (viz, máscaras, recortes)
```

## Imágenes Docker

| Tag | Dockerfile | Base runtime | Hardware | Tamaño |
|-----|-----------|--------------|----------|--------|
| `sam2:x86` | `submodules/sam2/Dockerfile.x86-server` | `pytorch:2.5.1-cuda12.1-cudnn9-runtime` | x86 GPU (CUDA 12.1) | **18.1 GB** |
| `sam2:x86-cuda128` | `submodules/sam2/Dockerfile.x86` | `pytorch:2.7.0-cuda12.8-cudnn9-runtime` | x86 GPU (CUDA 12.8+) | ~18 GB |
| `sam2:jetson` | `submodules/sam2/Dockerfile.jetson` | `dustynv/pytorch:2.6-r36.4.0-cu128` | Jetson GPU | — |
| `triposr` | `models/triposr/Dockerfile` | `dustynv/pytorch:2.6-r36.4.0-cu128` | Jetson GPU | — |
| `triposr:cpu` | `models/triposr/Dockerfile.cpu` | `python:3.12-slim` | CPU cualquier máquina | — |
| `triposr:x86` | `models/triposr/Dockerfile.x86` | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` | x86 GPU | — |
| `trellis` | `models/trellis/Dockerfile` | `dustynv/pytorch:2.7-r36.4.0` | Jetson GPU | — |
| `trellis:x86` | `models/trellis/Dockerfile.x86` | — | x86 GPU | — |
| `numcc:x86` | `models/numcc/Dockerfile.x86` | `pytorch:2.1.0-cuda11.8-cudnn8-runtime` | x86 GPU | **15.5 GB** |
| `dvlt:x86` | `submodules/dvlt.cu/Dockerfile.x86` | `nvidia/cuda:12.9.1-base-ubuntu24.04` | x86 GPU | **2 GB** |
| `dvlt:jetson` | `submodules/dvlt.cu/Dockerfile.jetson` | L4T | Jetson GPU | — |

**Todas las imágenes x86 usan multi-stage build** (devel para compilar → runtime para el artifact final). Pesos nunca embebidos — se montan como volumen en runtime.

**Builds x86 (RTX 4000 Ada, desde cero):**

| Imagen | Build time | Reducción vs original |
|--------|-----------|----------------------|
| `sam2:x86` | ~5–6 min | 33.1 GB → 18.1 GB (-45%) |
| `numcc:x86` | ~4–5 min | 31 GB → 15.5 GB (-50%) |
| `dvlt:x86` | ~1 min | 7.76 GB → 2 GB (-74%) |

## Comandos principales

### SAM2 — segmentar objeto

```bash
# x86 (este servidor) — descarga checkpoint la primera vez
mkdir -p ~/models/sam2
cd submodules/sam2/checkpoints && bash download_ckpts.sh  # o solo tiny/small

# Segmentar (desde Jetson-testing/)
python3 tools/generate_asset.py data/images/imagen.png --model sam2 --name objeto

# Con todas las máscaras individuales
python3 tools/generate_asset.py data/images/imagen.png --model sam2 --name objeto --all-masks
```

**Outputs en `data/outputs/`:**
- `{nombre}_viz.png` — imagen original + todas las máscaras con bounding boxes
- `{nombre}.png` — objeto principal recortado sobre fondo blanco (listo para TripoSR/TRELLIS)
- `{nombre}_mask{N}.png` — máscaras individuales (con `--all-masks`)

**Configuración elegida:** `sam2.1_hiera_small` + `SAM2AutomaticMaskGenerator(points_per_side=8)`.
Grid 8×8 = 64 prompts en vez de 32×32 = 1024 (default). 16× menos prompts, calidad idéntica para objeto único centrado. Usar `SAM2ImagePredictor` con un solo punto central NO funciona para objetos huecos (taza): el punto cae dentro de la cavidad y SAM2 segmenta el interior, no el objeto completo.

**Benchmarks SAM2 small en taza.jpeg (1156×868, RTX 4000 Ada, `sam2:x86`):**

| Config | Segmentación | Wall clock | VRAM pico | Calidad |
|--------|-------------|-----------|-----------|---------|
| AMG 32×32 (1024 pts, default) | 2.95s | 7.35s | 6,980 MB | ✓ |
| **AMG 8×8 (64 pts, actual)** | **0.84s** | **5.23s** | **5,918 MB** | ✓ idéntica |

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

### numcc — reconstrucción RGBD (P2C + NU-MCC)

Pipeline para reconstruir objetos desde depth map + imagen de color (RGBD). Especialmente útil con simulaciones Drake (AIRA) o cámaras de profundidad reales.

```bash
# Descargar pesos (una sola vez)
python3 tools/download_models_numcc.py
# → ~/models/numcc/p2c/p2c_checkpoint.pth  (1.9 GB, Google Drive)
# → ~/models/numcc/numcc/numcc_checkpoint.pth  (2.4 GB, S3)

# Construir imagen Docker (una sola vez, ~15 min — compila nksr + P2C CUDA extensions)
docker build -t numcc:x86 -f models/numcc/Dockerfile.x86 .

# Pipeline completo desde foto + depth map
python3 tools/run_numcc.py \
    --image  data/images/objeto.jpg \
    --depth  data/outputs/pipeline/objeto/numcc_input/depth_full.npy \
    --name   objeto \
    --intrinsics data/outputs/pipeline/objeto/numcc_input/intrinsics.json \
    --mask   data/outputs/pipeline/objeto/numcc_input/mask.npy \
    --mesh-method noksr          # o poisson

# Solo remesh desde nube de puntos existente (salta NU-MCC, muy rápido)
python3 tools/run_numcc.py \
    --cloud  assets/objeto/objeto/objeto_numcc_surface.ply \
    --name   objeto \
    --mesh-method noksr

# Correr Docker directamente (máximo control)
docker run --rm --gpus all \
    -v ~/models/numcc:/opt/models:ro \
    -v ~/models/nksr_cache:/root/.cache/torch \
    -v "$(pwd)/data/outputs/pipeline/nombre/numcc_input":/input:ro \
    -v "$(pwd)/assets":/output \
    numcc:x86 \
    --depth /input/depth_full.npy \
    --color /input/nombre_masked.png \
    --mask  /input/mask.npy \
    --intrinsics /input/intrinsics.json \
    --name nombre_objeto \
    --output /output \
    --mesh-method noksr \
    --udf-threshold 0.23
```

**Formatos aceptados:**
- `--depth`: `.npy` (float32, metros) o `.png` (uint16, milímetros)
- `--color`: `.npy` (uint8 HxWx3 o HxWx4 RGBA) o `.png`
- `--intrinsics`: JSON con claves `fx`, `fy`, `cx`, `cy`

**Intrínsecas de la cámara Drake (AIRA):** fx=fy=579.41, cx=319.5, cy=239.5 (imagen 640×480, FOV 45°)

**Mount de modelos:** el pipeline espera `/opt/models` dentro del container — montar como `~/models/numcc:/opt/models:ro`. Montar también `~/models/nksr_cache:/root/.cache/torch` para cachear el checkpoint nksr (~55 MB, se descarga de HuggingFace la primera vez).

**Nota sobre el checkpoint P2C:** el archivo descargado (`p2c_checkpoint.pth`) es un ZIP con checkpoints por categoría ShapeNet (plane/car/chair/lamp/sofa/table/watercraft/cabinet). No corresponde a la arquitectura P2C de CuiRuikai. El pipeline tiene fallback gracioso — si `load_state_dict` falla, pasa la nube de depth directamente a NU-MCC sin completar.

**Parámetros correctos del checkpoint NU-MCC (udf-ep99.pth, CO3D-V2):**
- `n_groups=200` (shape de init_embedding en el checkpoint)
- `nneigh=4` — vecinos de anclaje que el decoder atiende (training default; usar 45 rompe la distribución de atención)
- `nn_seen=4` — vecinos de seen_xyz por query point (training default; -1 → OOM de 95 GiB)
- `udf_threshold=0.23` — threshold de training (0.05 descartaba casi todos los puntos válidos; con 0.23 se obtienen ~40K candidatos)
- `repulsive=1` — fuerzas repulsivas activas en `move_points`
- `seen_xyz`: mapa XYZ 2D `(B, 112, 112, 3)` normalizado a zero-mean/unit-std; inválidos = `float('inf')`
- `seen_images`: 800×800 obligatorio (assert en preprocess_img); canal alpha eliminado si RGBA
- Normalización de seen_xyz: stats computadas sobre píxeles de objeto únicamente (máscara aplicada antes de normalizar)

**`move_points` es crítico:** después del filtro por UDF threshold, cada punto candidato se refina por descenso de gradiente sobre el campo UDF (`udf_n_iter=3` iteraciones). Sin esto la superficie es muy escasa e irregular.

**NU-MCC es densificador/completador multiview, NO generador 360° monocular:**
NU-MCC (Multiview Compressive Coding) necesita varias vistas para completar las zonas ocluidas; desde UNA imagen solo densifica y limpia la superficie visible, no alucina la parte trasera. Esto es comportamiento correcto — no es bug. Consecuencia práctica:
- **Entrada DA3 monocular + objeto cóncavo (ej. taza):** NU-MCC produce cáscara abierta → CoACD genera 200-333 partes → SDF fragmentado. Preferir **TRELLIS o TripoSR** en ese caso.
- **Entrada RGBD real (Drake/AIRA depth cam):** NU-MCC densifica correctamente (ej. 4480→13054 pts) y agrega grosor volumétrico. Caso de uso ideal.
- **Entrada DA3 monocular + objeto convexo (ej. lapicero):** funciona bien, pocos partes CoACD (~7-40).

**Debug dump de NU-MCC:**
```bash
docker run ... numcc:x86 --depth ... --debug-dump /output/debug
# Genera en /output/debug/: seen_image.png, seen_xyz_{x,y,z}.png, anchors.png,
# candidates.png, surface.ply, udf_hist.png, summary.json (pct_completion, n_candidates, etc.)
```

**Métodos de reconstrucción de mesh (`--mesh-method`):**

| Método | Faces típicas | Descripción |
|--------|--------------|-------------|
| `noksr` (default) | ~260K | nksr Neural Kernel Surface Reconstruction — prior aprendido, mejor en zonas dispersas |
| `poisson` | ~140K | Open3D Screened Poisson — más suave, menor conteo de caras, más rápido |

nksr descarga un checkpoint (~55 MB de HuggingFace) en el primer uso. El resultado tiene mayor resolución geométrica pero CoACD genera más partes convexas. Para Drake, ambos son válidos — usar `poisson` si se necesitan menos partes CoACD.

**Outputs en `assets/<nombre>/`:**
- `<nombre>.obj` — mesh final (método elegido)
- `<nombre>.sdf` — listo para Drake
- `<nombre>_numcc_surface.ply` — nube bruta de NU-MCC (espacio normalizado, antes del mesh)
- `<nombre>_object_cloud.ply` — back-projection del depth (espacio métrico, antes de NU-MCC, con colores RGB)
- `<nombre>_pointcloud.npy` — nube de puntos P2C completada
- `<nombre>_parts/` — piezas CoACD

**Tiempo típico (RTX 4000 Ada, 20 GB):**
- Pipeline completo (NU-MCC + nksr): ~70–100s
- Pipeline completo (NU-MCC + poisson): ~45–60s
- Solo remesh desde PLY (nksr): ~15–20s
- Solo remesh desde PLY (poisson): ~5–10s

**E2E benchmarks reales (RTX 4000 Ada, DA3→SAM2→NU-MCC→Drake):**

| Asset | DA3 | SAM2 | NU-MCC | Drake | VRAM pico | Partes CoACD |
|-------|-----|------|--------|-------|-----------|--------------|
| lapicero (convexo) | ~10s / 9GB | ~5s / 6GB | ~55s / 9.4GB | ~1s | 9.4 GB | ~7 (poisson) |
| taza (cóncavo, DA3) | 9.6s / 9.2GB | **5.2s / 5.9GB** | 50.9s / 9.4GB | 0.6s | 9.4 GB | 234 (poisson) / 333 (noksr) |

SAM2 mejorado con AMG 8×8 (antes: ~7.6s / 8.1GB VRAM).

Taza produce muchas partes CoACD porque DA3 monocular solo reconstruye la superficie visible (cáscara abierta). Para taza desde una imagen usar TRELLIS/TripoSR.

### Pipeline completo DA3 → SAM2 → numcc → Drake

Script unificado `tools/pipeline.py` — encadena todos los stages con output organizado por subdirectorios y monitoreo de VRAM:

```bash
# Pipeline completo (primera vez — corre todo)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto

# Saltar stages ya calculados
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --skip-da3 --skip-sam2

# Con visualización interactiva en Meshcat (bloquea hasta Ctrl+C)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --drake-interactive

# Con UDF threshold relajado (recomendado para objetos pequeños/monoculares)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --udf-threshold 0.10

# Desde depth map ya calculado (salta DA3 y SAM2)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --depth path/to/depth.npy --mask path/to/mask.npy --skip-da3 --skip-sam2
```

**Outputs en `output/<nombre>/`:**
- `01_da3/` — NPZ + depth_full.npy + intrinsics.json + depth_vis.png
- `02_sam2/` — segmask.npy + imagen fondo blanco + viz.png
- `03_numcc_inputs/` — depth_full.npy + mask.npy + intrinsics.json + *_masked.png
- `04_pointcloud/` — *_object_cloud.ply + *_numcc_surface.ply
- `05_mesh/` — <nombre>.obj + <nombre>.sdf + <nombre>_parts/
- `06_debug/` — (solo con `--debug`) dump NU-MCC inputs/outputs
- `pipeline_report.json` — tiempos, VRAM pico, parámetros, inventario de archivos
- `vram_profile.csv` — uso de VRAM por stage

### dvlt — reconstrucción gaussiana
```bash
# Setup (una sola vez)
mkdir -p ~/dvlt.cu/model
docker run --rm --runtime nvidia -v ~/dvlt.cu/model:/dvlt/model dvlt:jetson --setup

# Correr con carpeta de imágenes
cd ~/Jetson-testing
./submodules/dvlt.cu/run_dvlt.sh ~/Jetson-testing/data/images/taza taza
```

### Depth Anything 3 — depth map + reconstrucción 3D

DA3 está instalado como submódulo en `submodules/depth-anything-3/`. El CLI `da3` queda disponible en el venv de UniWhere (`.venv`). El modelo por defecto es `depth-anything/DA3NESTED-GIANT-LARGE-1.1` (~descarga automática desde HF la primera vez).

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

**DA3 entrega depth métrico:** DA3NESTED-GIANT-LARGE produce depth en metros directamente (ej. taza a ~43cm → valores [0.30, 0.85]m). **NO renormalizar** a rangos arbitrarios como [0.1, 1.5]m — destruye la proporción Z/XY y distorsiona la nube de puntos vista desde ángulos laterales. Usar `np.clip(depth, 0.05, 20.0)` como máximo.

**Intrínsecos del NPZ:** corresponden a la resolución procesada por DA3 (ej. 504×378 para una imagen original de 1156×868). DA3 escala los intrínsecos proporcionalmente al redimensionar. El NPZ guarda `image` y `depth` ya en resolución procesada — los intrínsecos son consistentes con esa resolución, no con la imagen original.

**Claves del NPZ de DA3:**
- `image`: `(N, H, W, 3)` uint8 — imagen procesada (resolución reducida)
- `depth`: `(N, H, W)` float32 — depth métrico en metros
- `conf`: `(N, H, W)` float32 — mapa de confianza
- `intrinsics`: `(N, 3, 3)` float32 — matriz K estimada por el modelo
- `extrinsics`: `(N, 3, 4)` float32 — pose estimada (identidad para imagen única)

**Bugs corregidos en el CLI de DA3** (`submodules/depth-anything-3/src/depth_anything_3/cli.py`):
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

## Post-procesado de mesh (PyMeshLab)

PyMeshLab instalado en el venv del proyecto (`.venv`). Permite aplicar filtros MeshLab por CLI sin abrir la GUI.

```bash
# Instalar (ya hecho)
uv pip install pymeshlab
```

### smooth_preserve.py — suavizado preservando geometría

Secuencia recomendada para eliminar grumos manteniendo bordes y detalle:
1. Clustering Decimation — reduce faces primero
2. Taubin Smooth — suaviza sin encoger el mesh (mejor que Laplacian)
3. Two Steps Smoothing — alisa zonas planas, preserva aristas por ángulo
4. Smooth Face Normals — limpia normales al final

```bash
# Uso básico
.venv/bin/python3 tools/smooth_preserve.py input.obj output.obj

# Ajustar agresividad (más iter = más liso; angle menor = preserva más bordes)
.venv/bin/python3 tools/smooth_preserve.py input.obj output.obj \
    --taubin-iter 20 --twosteps-iter 5 --feature-angle 30
```

### mesh_filter.py — filtros individuales

```bash
.venv/bin/python3 tools/mesh_filter.py input.obj output.obj \
    --smooth-normals --smooth-iter 4 \
    --depth-smooth --depth-smooth-iter 4 \
    --cluster-decimation --threshold 0.3
```

### apply_mlx.py — aplicar script exportado desde GUI MeshLab

Exportar desde MeshLab GUI: Filters → Show current filter script → Save Script (.mlx)

```bash
.venv/bin/python3 tools/apply_mlx.py input.obj output.obj assets/Scripts/mi_script.mlx
```

**Scripts MLX guardados en `assets/Scripts/`:**
- `script_smoothing.mlx` — secuencia de smooth normals + clustering decimation + depth smooth aplicada en GUI
- `script_smooth_preserve.mlx` — clustering + Taubin + Two Steps (recomendado)

**Nota:** assets generados por Docker son de root — hacer `sudo chown -R $USER:$USER assets/` antes de guardar con PyMeshLab.

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
| P2C checkpoint (zip categorías) | `~/models/numcc/p2c/` | 1.9 GB |
| NU-MCC CO3D-V2 (udf-ep99.pth) | `~/models/numcc/numcc/` | 2.4 GB |
| nksr checkpoint (ks.pth) | `~/models/nksr_cache/torch/hub/checkpoints/` | 55 MB |

## Optimizaciones activas (TripoSR)

- **FP16** en GPU (Ampere, nativo en Orin)
- **chunk_size = 262144** (16 GB RAM unificada)
- `sudo nvpmodel -m 0 && sudo jetson_clocks` antes de inferencia

## Notas importantes

- Docker usa `--runtime=nvidia` (no `--gpus all`) en esta Jetson
- Los assets generados por Docker son propiedad de root — usar `sudo chown -R jetson:jetson ~/Jetson-testing/assets` si hay problemas de permisos
- El entorno `pyproject.toml` (uv/Drake) es independiente — NO incluye TripoSR ni TRELLIS (conflictos de versiones)
- `submodules/dvlt.cu`, `submodules/depth-anything-3`, `submodules/sam2` y `submodules/nksr` son submódulos git — clonar con `git clone --recurse-submodules`
- `git config --global submodule.recurse true` para que pull/fetch actualice submódulos automáticamente
- `sam2` es un fork de `facebookresearch/sam2` en `cristian10gf/sam2`
- DA3 instalado en el venv de UniWhere (`/home/worker-node-4/Documents/GitHub/UniWhere/.venv`), no tiene venv propio; el editable install apunta a `submodules/depth-anything-3/src` vía `_editable_impl_depth_anything_3.pth` — si falla `import depth_anything_3`, verificar/corregir esa ruta en el `.pth`
- xformers instalado pero sin extensiones CUDA (torch 2.12 vs xformers compilado para 2.10) — funciona igual, solo sin memory-efficient attention
- La IP de la Jetson cambia por DHCP — pendiente configurar IP estática en el router
- SAM2 extensión CUDA (`sam2._C`) no compiló en la imagen x86 actual — funciona igual, solo sin post-procesado de huecos (no afecta resultados en la mayoría de casos)
- numcc usa `--gpus all` (x86), no `--runtime=nvidia` (Jetson). El `generate_asset.py` ya lo maneja automáticamente según el modelo
- `tools/pipeline.py` corre Docker numcc con `--user $(uid):$(gid)` para evitar archivos root-owned en el output; usa `tempfile.mkdtemp` por run para el directorio temporal (evita colisiones con runs anteriores de root)
- numcc `Dockerfile.x86` es multi-stage: stage devel compila CUDA extensions (chamfer_dist, pointops, nksr) → stage runtime copia `/opt/conda` completo; requiere `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"` y `numpy<2`
- numcc `pipeline.py` tiene ENTRYPOINT — al correr Docker los argumentos van directo, sin `python3 /app/pipeline.py` delante
- numcc `remesh.py` requiere `--entrypoint python3` para activarse: `docker run --entrypoint python3 numcc:x86 /app/remesh.py ...`
- `generate_asset.py` no soporta `--mask` para numcc — usar `tools/run_numcc.py` o Docker directamente
- nksr submodulo en `submodules/nksr/` (nv-tlabs/nksr) — se compila en Docker build desde `submodules/nksr/package/setup.py` con `--no-build-isolation`; requiere `gitpython` para descargar OpenVDB + Eigen durante el build; tiempo de compilación ~4 min; Docker build ahora usa contexto `.` (raíz del repo): `docker build -t numcc:x86 -f models/numcc/Dockerfile.x86 .`
- nksr wheel server (`nksr.huangjh.tech`) está permanentemente caído (NXDOMAIN); el submodulo es la única vía de instalación
- nksr checkpoint (`ks.pth`, ~55 MB) se descarga automáticamente de HuggingFace en el primer uso — `pipeline.py` monta `~/models/nksr_cache:/nksr_cache` + `-e TORCH_HOME=/nksr_cache` (no `/root/.cache/torch` — incompatible con `--user`)
- El gitignore cubre `/assets/` — ningún asset generado se commitea
- `pipeline.py` y scripts de tools requieren `.venv/bin/python3` (no `python3` del sistema — no tiene numpy)
- sam2 `Dockerfile.x86*` son multi-stage: builder (devel) compila `sam2._C.so` → runtime copia `/opt/conda`; los configs YAML no se instalan con `pip install .` (wheel build ignora MANIFEST.in) — se copian explícitamente a `site-packages/sam2/configs/` para que Hydra los encuentre vía `pkg://sam2`
- dvlt `Dockerfile.x86` usa `nvidia/cuda:base` (solo cudart, ~150 MB) en lugar de `runtime` (cuda-libraries bundle, ~3.5 GB); dvlt solo linka `-lcublasLt` — `libcublas-12-9` se instala explícitamente; no necesita cusparse, curand, ni NCCL
- dvlt benchmarks reales (RTX 4000 Ada, 6 fotos shoe): wall clock 1.35s, inferencia 488ms, VRAM pico 2,580 MB, 973K puntos PLY
- SAM2 pesos **no embebidos** en la imagen — se montan en runtime: `-v ~/models/sam2:/opt/sam2/checkpoints:ro`; sin el mount falla con FileNotFoundError en `load_model()`
