# Jetson Testing — TripoSR Asset Pipeline

## Proyecto

Convierte una imagen (PNG/JPG) en un **mesh 3D + Drake SDF** listo para simulación de manipulación robótica.

Stack: TripoSR (reconstrucción 3D) + coacd (descomposición convexa) + Drake SDF.

## Hardware / Conexión

- Jetson Orin NX Developer Kit Super, JetPack 6 / L4T R36
- SSH: `ssh jetson` (usuario: `jetson`, IP dinámica — si falla: `arp -a | findstr "192.168.1"` en PowerShell)
- Dentro de la Jetson: usar **tmux** para no perder procesos si cae SSH (`tmux attach` para reconectar)

## Estructura del repo

```
triposr/
  Dockerfile          # GPU — base l4t-pytorch:r36.2.0-pth2.1-py3
  Dockerfile.cpu      # CPU — base python:3.12-slim
  pipeline.py         # Entrypoint del contenedor (5 pasos: preprocess → TripoSR → normalize → coacd → SDF)
  image_utils.py      # Preprocesamiento de imagen (rembg background removal)
  mesh_utils.py       # normalize_mesh (eje más largo → 20 cm) + decompose_convex (coacd)
  sdf_generator.py    # Genera Drake SDF con inercia + colisiones convexas
  requirements.txt    # Deps GPU (sin torch — viene del base l4t-pytorch)
  requirements.cpu.txt
tools/
  generate_asset.py   # Wrapper host-side: llama Docker, monta volúmenes, guarda en assets/
quick_test.py         # Test directo sin Docker (Linux/CUDA)
quick_test_cpu_windows.py  # Test directo sin Docker (Windows CPU)
pyproject.toml        # Entorno Drake/AIRA (uv) — NO incluye TripoSR (conflictos de versiones)
```

## Imágenes Docker

| Target | Dockerfile | Hardware |
|--------|-----------|----------|
| `triposr` | `triposr/Dockerfile` | Jetson GPU (CUDA, ARM64) — base: dustynv/torch:2.3-r36.4.0 |
| `triposr:cpu` | `triposr/Dockerfile.cpu` | Cualquier máquina, CPU |

## Comandos principales

### Verificar Docker + GPU en la Jetson
```bash
docker run --rm --gpus all nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3 \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

### Build (una sola vez — baja modelos ~2 GB)
```bash
# GPU (en la Jetson)
docker build -t triposr -f triposr/Dockerfile triposr/

# CPU (cualquier máquina)
docker build -t triposr:cpu -f triposr/Dockerfile.cpu triposr/
```

### Generar asset
```bash
# Recomendado — wrapper automático
python tools/generate_asset.py path/to/image.png --name mug        # GPU
python tools/generate_asset.py path/to/image.png --name mug --cpu  # CPU

# Docker directo
docker run --rm --gpus all \
  -v /ruta/absoluta/imagenes:/input:ro \
  -v /ruta/absoluta/assets:/output \
  triposr \
  --input /input/image.png --output /output --name mug
```

### Salida esperada
```
assets/mug/
  mug.obj              # mesh normalizado (eje más largo = 20 cm)
  mug.sdf              # Drake SDF con inercia + colisiones convexas
  mug_parts/
    convex_piece_000.obj
    convex_piece_001.obj
```

## Setup en la Jetson (pasos pendientes)

1. **Verificar Docker con GPU:**
   ```bash
   docker --version
   docker run --rm --gpus all nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3 python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Clonar/transferir el repo** a la Jetson:
   ```bash
   git clone <repo-url> ~/jetson-testing
   # O desde Windows:
   scp -r . jetson:~/jetson-testing
   ```

3. **Build de la imagen Docker GPU** (tarda ~20-30 min, baja ~2 GB de modelos):
   ```bash
   cd ~/jetson-testing
   docker build -t triposr -f triposr/Dockerfile triposr/
   ```

4. **Probar con una imagen:**
   ```bash
   python3 tools/generate_asset.py test_image.png --name test
   ```

## Notas importantes

- El `Dockerfile` GPU usa `l4t-pytorch:r36.2.0-pth2.1-py3` como base — NO instala PyTorch por separado, ya viene en la imagen base de NVIDIA para Jetson.
- `torchmcubes` se compila desde fuente en el build de Docker (requiere cmake, g++, ninja).
- `coacd` puede fallar en ARM64; el pipeline tiene fallback automático a convex hull.
- Los modelos se descargan **en el build** (no en inferencia) — `HF_HOME=/opt/models/huggingface` dentro del contenedor.
- El entorno `pyproject.toml` (uv/Drake) es independiente del entorno TripoSR — tienen conflictos de versiones, por eso TripoSR siempre corre en Docker.
- Para tests rápidos sin Docker en Windows usar `quick_test_cpu_windows.py` con TripoSR clonado en `PYTHONPATH`.
