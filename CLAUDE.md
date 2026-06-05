# Jetson Testing — Asset Pipeline

## Proyecto

Tres pipelines para generar assets 3D desde imágenes, listos para simulación robótica con Drake.

| Pipeline | Modelo | Entrada | Salida |
|----------|--------|---------|--------|
| TripoSR | TripoSR (Stability AI) | 1 imagen | OBJ + SDF |
| TRELLIS | TRELLIS-image-large (Microsoft) | 1 imagen | GLB + OBJ + SDF |
| dvlt.cu | DVLT (NVIDIA) | 1+ imágenes | PLY gaussiano → OBJ |

## Hardware / Conexión

- Jetson Orin NX Developer Kit Super, JetPack 6 / L4T R36.4.7, 16 GB RAM unificada
- SSH: `ssh jetson` (usuario: `jetson`, IP dinámica — si falla buscar con `arp -a | findstr "192.168.1"`)
- IP actual: `192.168.1.43` (puede cambiar — DHCP)
- Siempre usar **tmux** en la Jetson para no perder procesos si cae SSH

## Estructura del repo

```
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
  pipeline.py           # preprocess → TRELLIS → GLB → OBJ → coacd → SDF
  image_utils.py
  mesh_utils.py
  sdf_generator.py
  requirements.txt
dvlt.cu/                # submodulo — reconstrucción 3D gaussiana
  run_dvlt.sh           # wrapper para correr dvlt con una carpeta de imágenes
tools/
  generate_asset.py         # wrapper principal (soporta triposr y trellis)
  download_models.py        # descarga modelos TripoSR en la Jetson
  download_models_trellis.py        # descarga modelos TRELLIS en la Jetson
  download_models_trellis_windows.py # descarga modelos TRELLIS en Windows + scp a Jetson
  ply_to_obj.py             # convierte PLY gaussiano (dvlt) a OBJ mesh
```

## Imágenes Docker

| Tag | Dockerfile | Base | Hardware |
|-----|-----------|------|----------|
| `triposr` | `triposr/Dockerfile` | `dustynv/pytorch:2.6-r36.4.0-cu128` | Jetson GPU |
| `triposr:cpu` | `triposr/Dockerfile.cpu` | `python:3.12-slim` | CPU cualquier máquina |
| `trellis` | `trellis/Dockerfile` | `dustynv/pytorch:2.7-r36.4.0` | Jetson GPU |
| `dvlt:jetson` | `dvlt.cu/Dockerfile.jetson` | L4T | Jetson GPU |

## Comandos principales

### TripoSR — generar asset
```bash
# Descargar modelos (una sola vez)
python3 tools/generate_asset.py data/images/imagen.png --name objeto

# Con más detalle de tiempo
# El output incluye: Xs total al final
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

Los modelos se guardan en el host de la Jetson y se montan en Docker:

| Modelo | Ubicación en Jetson | Tamaño |
|--------|-------------------|--------|
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
- `dvlt.cu` es un submódulo git — clonar con `git clone --recurse-submodules`
- `git config --global submodule.recurse true` para que pull/fetch actualice submódulos automáticamente
- La IP de la Jetson cambia por DHCP — pendiente configurar IP estática en el router
