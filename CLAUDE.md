# Jetson Testing — Asset Pipeline

## Proyecto

Seis pipelines para generar assets 3D desde imágenes o datos RGBD, listos para simulación robótica con Drake. Más un pipeline de estimación de profundidad monocular.

| Pipeline | Modelo | Entrada | Salida |
|----------|--------|---------|--------|
| SAM2 | SAM2.1 small (Meta) | 1 imagen | objeto recortado PNG (fondo blanco) |
| TripoSR | TripoSR (Stability AI) | 1 imagen | OBJ + SDF |
| TRELLIS | TRELLIS-image-large (Microsoft) | 1 imagen | GLB + OBJ + SDF |
| **numcc** | **P2C + NU-MCC (CO3D-V2)** | **depth map + color (RGBD)** | **OBJ + OBJ_smoothed** |
| dvlt.cu | DVLT (NVIDIA) | 1+ imágenes | PLY gaussiano → OBJ |
| depth-anything-3 | DA3NESTED-GIANT-LARGE (ByteDance) | 1+ imágenes | depth map + GLB point cloud |

**Flujo típico imagen:** SAM2 → TripoSR o TRELLIS (SAM2 reemplaza a rembg para segmentar el objeto antes de la reconstrucción 3D)

**Flujo RGBD (cámara de profundidad / simulación Drake):** depth + color → numcc (P2C + NU-MCC) → OBJ → smooth (PyMeshLab) → OBJ_smoothed

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
    pipeline.py           # depth → P2C → NU-MCC → surface pts → mesh → (coacd → SDF, skip con --no-sdf)
    remesh.py             # mesh-only desde PLY existente (salta NU-MCC) — corre dentro de Docker
    pointcloud_utils.py   # back-projection depth → nube de puntos
    mesh_utils.py
    sdf_generator.py
    requirements.txt
tools/
  pipeline.py                    # super-pipeline unificado: imagen → output/ estructurado por stages
  pipeline_scripts/
    pipeline_multiview.py        # pipeline N-vistas v2: DA3→SAM2 AMG→FPFH+ICP merge→[NU-MCC]→remesh→smooth
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

**Configuración actual:** `sam2.1_hiera_small` + `SAM2AutomaticMaskGenerator()` (default — `points_per_side=32`, 1024 prompts, `pred_iou_thresh=0.88`, `stability_score_thresh=0.95`).
`SAM2ImagePredictor` con un solo punto central NO funciona para objetos huecos (taza): el punto cae dentro de la cavidad y SAM2 segmenta el interior, no el objeto completo.
`points_per_side=8` (64 prompts) falló en objetos de forma compleja (audífonos): los puntos caen en el fondo visible a través del arco y AMG filtra las masks por bajo IoU → objeto no detectado.

**Merge de masks (merge_centered_masks):** seed = mask más centrada no-background (area < 30% frame). Absorbe iterativamente masks dentro de `dilation_px=30` px de distancia Euclidiana (vía `distance_transform_edt`, O(n) vs OOM de `binary_dilation` a radios grandes). `max_area_frac=0.30` (default) — excluye fondos de escena completa (>60%) pero permite objetos grandes retornados como una sola mask por AMG (ej. taladro=18.22%). Threshold anterior 0.05 fallaba cuando AMG retorna el objeto completo como una sola mask >5%: sin candidatos válidos, el algoritmo elegía ruido de fondo. Configurable vía `--max-area-frac`.

**Thresholds AMG y estructuras delgadas:** AMG filtra como "ruido" masks con IoU < 0.88 o stability < 0.95. El arco del cinto de audífonos genera masks de baja confianza → queda fuera con defaults. Con `--pred-iou-thresh 0.70 --stability-score-thresh 0.80` se detecta el arco (41 masks vs 19, area 91K vs 65K px²). Trade-off: más noise en escenas complejas (centro.jpeg con estos thresholds fusiona objetos del fondo). Usar thresholds bajos solo cuando el objeto tiene estructuras delgadas Y la imagen tiene fondo simple.

**Ángulo de imagen crítico para audífonos:** vista cenital (centro.jpeg) el arco del cinto tiene ~20px de ancho — AMG nunca lo detecta. Vista lateral (derecho.jpeg) el arco es claramente visible y grueso → con thresholds bajos se captura. **Usar `headphones_derecho.jpeg` con `--sam2-pred-iou-thresh 0.70 --sam2-stability-thresh 0.80`.**

**Benchmarks SAM2 small (1156×868, RTX 4000 Ada, `sam2:x86`):**

| Config | Segmentación | Wall clock | VRAM pico | Calidad |
|--------|-------------|-----------|-----------|---------|
| **AMG 32×32, iou=0.88, stab=0.95 (default)** | **2.95s** | **7.35s** | **6,980 MB** | ✓ objetos simples |
| AMG 32×32, iou=0.70, stab=0.80 | 3.95s | ~10s | ~7,000 MB | ✓ audífonos arco visible |
| AMG 8×8 (64 pts, revertido) | 0.84s | 5.23s | 5,918 MB | ✗ falla shapes complejas |

**Checkpoints disponibles en `~/models/sam2/`:**

| Modelo | Tamaño | Nota |
|--------|--------|------|
| `sam2.1_hiera_tiny.pt` | 149 MB | más fragmenta |
| `sam2.1_hiera_small.pt` | 176 MB | **usado en pipeline** |
| `sam2.1_hiera_base_plus.pt` | 309 MB | seed diferente → peor resultado para audífonos (63K vs 91K px²) |

**En Jetson** usar `--runtime=nvidia` en lugar de `--gpus all` en el comando docker. El `generate_asset.py` usa `--gpus all` (para x86); para Jetson correr `pipeline.py` directamente dentro del container `sam2:jetson`.

### SAM2 — selector interactivo (point-click)

Para casos donde AMG elige la máscara equivocada o el fondo contamina el resultado.

```bash
# Prerequisito X11 (una sola vez por sesión)
xhost +local:docker

# Correr selector interactivo
.venv/bin/python3 submodules/sam2/pipeline_interactive.py \
  --input data/images/objeto.jpg \
  --output data/outputs \
  --name objeto
```

**Controles:**
- Click izquierdo → punto positivo (incluir en máscara)
- Click derecho → punto negativo (excluir de máscara)
- `R` → resetear puntos y máscara
- `Enter` → guardar y salir (exit 0) — requiere al menos un click
- `Q` → salir sin guardar (exit 1)

**Output:** mismo formato que `pipeline.py` — `<name>.png` (fondo blanco) + `<name>_segmask.npy` (uint8) + `<name>_viz.png`.
Compatible con `--skip-sam2` en `tools/pipeline.py` copiando los outputs a `output/<name>/02_sam2/`.

### SAM2 — pipeline automático YOLO + SAM2

Para segmentación sin tunear thresholds AMG. YOLO detecta el bbox, SAM2 segmenta con precisión.

```bash
# Objeto en COCO80 (taza, botella, silla, etc.)
.venv/bin/python3 submodules/sam2/pipeline_yolo_sam2.py \
  --input data/images/taza/taza.jpeg \
  --output data/outputs \
  --name taza \
  --class-name cup

# Objeto no en COCO80 (taladro, audífonos) — usar clase más alta confianza
.venv/bin/python3 submodules/sam2/pipeline_yolo_sam2.py \
  --input data/images/taladro.JPG \
  --output data/outputs \
  --name taladro \
  --any-class

# Escena multi-objeto — elegir interactivamente con click
xhost +local:docker
.venv/bin/python3 submodules/sam2/pipeline_yolo_sam2.py \
  --input data/images/escena.jpg \
  --output data/outputs \
  --name objeto \
  --interactive
```

**Flags de selección (uno requerido):**
- `--class-name cup` → filtrar por nombre de clase COCO
- `--class-id 41` → filtrar por ID de clase COCO
- `--any-class` → detección de mayor confianza sin filtro
- `--interactive` → mostrar todas las detecciones, click para elegir (requiere X11)

**COCO80 clases relevantes:** cup=41, bottle=39, chair=56, laptop=63, cell phone=67.
**Taladro y audífonos NO están en COCO80** → usar `--any-class` o `--interactive`.

**Output:** mismo formato que `pipeline.py` — `<name>.png` + `<name>_segmask.npy` + `<name>_viz.png` (viz incluye bbox YOLO en amarillo).
**Modelos YOLO:** `yolo11n.pt` (default, 6 MB) — se descarga automáticamente en `~/models/yolo/` la primera vez.

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

**`--no-sdf` flag:** pasa `--no-sdf` al container para saltar CoACD + SDF (usado por `tools/pipeline.py` siempre). Sin este flag corre CoACD y genera `.sdf` y `_parts/`.

**Floor cap penetration guard:** `_slice_and_cap_at_floor` genera una tapa Delaunay en el plano de soporte. El `merged` polygon usa buffer del 5% para bridgear gaps del boundary ruidoso, pero en objetos cóncavos (audífonos: arco, espacio entre copas) ese buffer llena concavidades y genera triángulos que cruzan las paredes del mesh. Fix: `original_for_guard = unary_union(significant).buffer(buf * 0.15)` (0.75% — solo tolerancia numérica). Después de apilar `all_cap_faces`, se filtra cualquier cara cuyo centroide caiga fuera de `original_for_guard`. Resultado típico audífonos: ~3500 caras penetrantes removidas, ~400 válidas retención.

**Outputs en `assets/<nombre>/` (run_numcc.py directo, sin --no-sdf):**
- `<nombre>.obj` — mesh final (método elegido)
- `<nombre>.sdf` — listo para Drake
- `<nombre>_numcc_surface.ply` — nube bruta de NU-MCC (espacio normalizado, antes del mesh)
- `<nombre>_object_cloud.ply` — back-projection del depth (espacio métrico, antes de NU-MCC, con colores RGB)
- `<nombre>_pointcloud.npy` — nube de puntos P2C completada
- `<nombre>_parts/` — piezas CoACD

**Tiempo típico (RTX 4000 Ada, 20 GB, run_numcc.py directo):**
- Pipeline completo (NU-MCC + nksr): ~18–20s (sin CoACD/SDF) / ~70–100s (con CoACD/SDF)
- Pipeline completo (NU-MCC + poisson): ~10–15s (sin CoACD/SDF) / ~45–60s (con CoACD/SDF)
- Solo remesh desde PLY (nksr): ~15–20s
- Solo remesh desde PLY (poisson): ~5–10s

**Tiempo típico (tools/pipeline.py — DA3+SAM2+numcc+smooth):** ~35–45s objetos simples, ~20s desde --skip-da3 --skip-sam2.

**E2E benchmarks reales (RTX 4000 Ada, DA3→SAM2→NU-MCC+nksr→smooth, sin SDF/CoACD):**

| Asset | DA3 | SAM2 | NU-MCC+nksr | smooth | Total | VRAM pico | Pts superficie |
|-------|-----|------|-------------|--------|-------|-----------|----------------|
| headphones (vista cenital, iou=0.88) | 13.6s / 7.2GB | 9.5s / 4.5GB | 13.2s / 11.2GB | 0.8s | ~37s | 11.2 GB | 13,797 |
| **headphones (vista lateral, iou=0.70)** | **11.4s / 7.2GB** | **14.1s / 5.3GB** | **18.3s / 11.2GB** | **2.3s** | **~46s** | **11.2 GB** | **30,627** |

- Guard removió ~3500 caras penetrantes en audífonos derecho → 428 válidas en cap.
- Audífonos vista cenital: arco no capturado. Vista lateral + thresholds bajos: arco completo.
- Sin SDF/CoACD el pipeline es 3–5× más rápido en el stage numcc.

### Pipeline completo DA3 → SAM2 → numcc → smooth

Script unificado `tools/pipeline.py` — encadena todos los stages con output organizado por subdirectorios y monitoreo de VRAM. **SDF y Drake eliminados del pipeline**; smoothing es stage obligatoria post-numcc.

```bash
# Pipeline completo (primera vez — corre todo)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto

# Desde numcc en adelante (salta DA3 y SAM2, recalcula mesh + smooth)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --skip-da3 --skip-sam2 --no-p2c

# Solo smooth sobre mesh existente (salta todo hasta smooth)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --skip-da3 --skip-sam2 --skip-numcc

# Con UDF threshold relajado (recomendado para objetos pequeños/monoculares)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --udf-threshold 0.10

# Desde depth map ya calculado (salta DA3 y SAM2)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --depth path/to/depth.npy --mask path/to/mask.npy --skip-da3 --skip-sam2

# Objeto con estructuras delgadas (arcos, asas, manijas) — bajar thresholds AMG
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --sam2-pred-iou-thresh 0.70 --sam2-stability-thresh 0.80

# Cambiar variante SAM2 (tiny/small/base_plus)
.venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \
    --sam2-model base_plus
```

**`--skip-numcc` salta también `stage_prepare`** — va directo al smooth sin tocar ningún archivo de depth/mask ni lanzar Docker.

**Outputs en `output/<nombre>/`:**
- `01_da3/` — NPZ + depth_full.npy + intrinsics.json + depth_vis.png
- `02_sam2/` — segmask.npy + imagen fondo blanco + viz.png
- `03_numcc_inputs/` — depth_full.npy + mask.npy + intrinsics.json + *_masked.png
- `04_pointcloud/` — *_object_cloud.ply + *_numcc_surface.ply
- `05_mesh/` — <nombre>.obj + <nombre>_smoothed.obj
- `06_debug/` — (solo con `--debug`) dump NU-MCC inputs/outputs
- `pipeline_report.json` — tiempos, VRAM pico, parámetros, inventario de archivos
- `vram_profile.csv` — uso de VRAM por stage

### Pipeline multiview v2 — DA3 → SAM2 (N vistas) → FPFH+ICP merge → remesh → smooth

Script `tools/pipeline_scripts/pipeline_multiview.py` — recibe directorio con N imágenes (diseñado para N=3), genera OBJ mesh.

**Flujo v2:** DA3 multiview → SAM2 AMG por vista (todas) → FPFH+ICP registration (object-centric, ref=best_view) → bbox fallback para vistas con rmse alto → merge + voxel downsample → [opcional: NU-MCC densificación con `--use-numcc`] → remesh (NKSR o Poisson vía `numcc:x86`) → smooth PyMeshLab

**Cambio principal vs v1:** v1 usaba extrinsics de DA3 para transformar las nubes al world frame → fallaba para vistas wide-baseline (DA3 no generaliza a fotografía de producto, el objeto aparecía en 2+ posiciones). v2 usa FPFH global registration + ICP refinement object-centric (centroid-subtracted): cada nube se centra en su propio centroide, se registra a la nube ref (best_view), sin depender de DA3 extrinsics.

```bash
# Pipeline completo — directorio con imágenes ordenadas
.venv/bin/python3 tools/pipeline_scripts/pipeline_multiview.py \
    --views-dir data/images/headphones_views/ \
    --name headphones \
    [--output-dir output]              # default: ./output
    [--mesh-method noksr|poisson]      # default: noksr
    [--voxel-size 0.005]              # metros, default 0.005 (5mm)
    [--sam2-model small|tiny|base_plus]
    [--sam2-pred-iou-thresh 0.88]
    [--sam2-stability-thresh 0.95]
    [--registration fpfh|icp]         # default: fpfh (icp para vistas narrow-baseline)
    [--icp-threshold 0.02]            # metros, distancia max ICP
    [--use-numcc]                     # Stage 4b: densificar nube con NU-MCC antes del remesh
    [--udf-threshold 0.23]            # solo con --use-numcc

# Objetos con estructuras delgadas (arcos, asas)
.venv/bin/python3 tools/pipeline_scripts/pipeline_multiview.py \
    --views-dir data/images/headphones_views/ \
    --name headphones \
    --sam2-pred-iou-thresh 0.70 --sam2-stability-thresh 0.80

# Skip stages (reusar outputs previos)
    [--skip-da3]      # reusar 01_da3/
    [--skip-sam2]     # reusar 02_sam2/
    [--skip-merge]    # reusar 03_clouds/merged_cloud.ply
    [--skip-smooth]
    [--debug]

# Re-run NU-MCC densificación en nube cached (sin repetir DA3/SAM2/merge)
.venv/bin/python3 tools/pipeline_scripts/pipeline_multiview.py \
    --views-dir data/images/headphones_views/ --name headphones \
    --skip-da3 --skip-sam2 --skip-merge --use-numcc --udf-threshold 0.15
```

**Selección de mejor vista (best_view):** `argmin(centerness_distance_i)` donde `centerness_distance = sqrt((cx_mask - W/2)² + (cy_mask - H/2)²)`. La mask más centrada es la de referencia para registro ICP (`ref_idx == best_idx` enforced).

**Registro FPFH+ICP (Stage 3+4 combinado):**
1. Back-project todas las vistas → nube en frame cámara
2. Restar centroide de cada nube (object-centric)
3. Para cada vista no-ref: FPFH global → coarse_T (maneja rotaciones >90°); luego ICP point-to-plane → fine_T
4. Divergence check: si `inlier_rmse > icp_threshold * 0.5` → usar coarse_T + opcionalmente re-segmentar con bbox
5. Bbox fallback (solo para vistas con rmse alto): proyectar nube ref a image_i via `transform_i.inverse()` → re-run SAM2 bbox → re-register
6. Merge: concatenar ref + aligned clouds → voxel downsample → Y-up → merged_cloud.ply

**Limitación conocida (vistas espejo):** FPFH puede fallar (`fitness≈0`, warning "Too few correspondences after mutual filter") cuando dos vistas son casi espejo (izquierdo vs derecho). En ese caso ICP con identity fallback puede tener rmse≈0 (0 correspondencias) sin divergence flag → la vista se incluye en posición incorrecta. Se puede mejorar bajando `--icp-threshold` o usando `--registration icp` si las vistas son narrow-baseline.

**Stage 4b NU-MCC densificación (`--use-numcc`):** Opcional. Corre `numcc:x86` con `--query-cloud` apuntando al merged_cloud.ply (bypass del meshgrid interno). Usa la best-view como `seen_xyz`. Output: `04_pointcloud/<nombre>_numcc_surface.ply`. Si NU-MCC no produce output, fallback a merged_cloud.ply para el remesh.

**SAM2 `--bbox` mode** (en `submodules/sam2/pipeline.py`): usa `SAM2ImagePredictor.predict(box=..., multimask_output=True)`, selecciona `masks[scores.argmax()]`. Sin `merge_centered_masks`. Activar con `--bbox x1 y1 x2 y2`.

**Outputs en `output/<nombre>/`:**
- `01_da3/` — depth_i.npy + intrinsics_i.json + extrinsics.npy (guardado pero no usado para merge)
- `02_sam2/` — segmask_i.npy + viz_i.png + best_view_idx.txt
- `03_clouds/` — *_cloud_i.ply (por vista, ref-centroid frame, Y-up) + *_merged_cloud.ply
- `04_pointcloud/` — *_numcc_surface.ply (solo con --use-numcc)
- `04_mesh/` — <nombre>.obj + <nombre>_smoothed.obj
- `pipeline_report.json` + `vram_profile.csv`

**Benchmarks E2E v2 (RTX 4000 Ada, 3 vistas headphones_3views/, iou=0.70, stab=0.80, nksr):**

| Stage | Tiempo | VRAM pico |
|-------|--------|-----------|
| DA3 multiview | 12.1s | 9,648 MB |
| SAM2 AMG × 3 | 37.9s | 9,648 MB |
| Merge v2 (FPFH+ICP) | ~3s | ~1,400 MB |
| Remesh nksr | 11.9s | 1,823 MB |
| **Total (sin smooth)** | **~67s** | **9,648 MB** |

- Dataset: `headphones_3views/view_0.jpg` (centro) + `view_1.jpg` (derecho, best) + `view_2.jpg` (izquierdo)
- best_view = view_1 (centerness=42.3)
- View 0: FPFH fitness=0.584, ICP rmse=0.0032 ✓ (buena alineación)
- View 2: FPFH fitness=0.000 (vistas espejo, mutual filter falla), ICP rmse=0.000 (identity fallback)
- Puntos: view_0=31,407 + view_1=17,389 (ref) + view_2=41,333 → merged 90,129 → 15,528 tras downsample
- OBJ final: 44,215 vértices, 84,882 caras (nksr)
- Tests: `tests/multiview/` — 27/27 pasan

**Benchmarks E2E v1 (referencia, DA3 extrinsics, mismos datos):**

| Stage | Tiempo | VRAM pico |
|-------|--------|-----------|
| DA3 multiview | 11.8s | 9,730 MB |
| SAM2 AMG × 3 + bbox | 39.3s | 9,730 MB |
| Merge + voxel | 0.01s | 1,479 MB |
| Remesh nksr | 13.3s | 1,837 MB |
| **Total (sin smooth)** | **~73s** | **9,730 MB** |

- v1 bug: DA3 extrinsics fallaban para wide-baseline → objeto aparecía en 2+ posiciones en merged cloud

**Nota:** DA3 `da3 images <dir>` (subcommand multiview) produce NPZ con `depth(N,H,W)`, `intrinsics(N,3,3)`, `extrinsics(N,3,4)`. No usar `da3 image img1 img2` (no soportado).

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
- `tools/pipeline.py` corre Docker numcc con `--user $(uid):$(gid)` para evitar archivos root-owned en el output; usa `tempfile.mkdtemp` por run para el directorio temporal (evita colisiones con runs anteriores de root); pasa siempre `--no-sdf` (SDF/CoACD eliminados del pipeline); stage Drake eliminada; smoothing (PyMeshLab) es stage obligatoria post-numcc → `<nombre>_smoothed.obj` en `05_mesh/`; `--skip-numcc` también salta `stage_prepare` (va directo al smooth)
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
