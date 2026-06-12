#!/usr/bin/env python3
"""
Pipeline numcc: photo + depth map → OBJ + SDF (Drake-ready).

SAM2 corre automáticamente si se provee --image y no se provee --mask.
Para saltarse SAM2 usa --skip-sam2 (corre numcc sin máscara) o provee --mask directamente.

Modos de uso:

  # Pipeline completo — SAM2 automático + NU-MCC
  python3 tools/run_numcc.py \\
      --image  data/aira_input_data/color.png \\
      --depth  data/aira_input_data/depth_image.npy \\
      --name   aira_screwdriver \\
      [--intrinsics data/aira_input_data/intrinsics.json] \\
      [--mesh-method noksr]

  # Máscara ya existente (salta SAM2)
  python3 tools/run_numcc.py \\
      --image  data/aira_input_data/color.png \\
      --depth  data/aira_input_data/depth_image.npy \\
      --mask   data/aira_input_data/mask.npy \\
      --name   aira_screwdriver

  # Sin máscara (escena completa)
  python3 tools/run_numcc.py \\
      --image  data/aira_input_data/color.png \\
      --depth  data/aira_input_data/depth_image.npy \\
      --name   aira_screwdriver \\
      --skip-sam2

  # Solo remesh desde nube de puntos existente (salta NU-MCC)
  python3 tools/run_numcc.py \\
      --cloud  assets/taza/taza/taza_numcc_surface.ply \\
      --name   taza \\
      [--mesh-method noksr]

Salidas en assets/<name>/<name>.obj y assets/<name>/<name>.sdf
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = Path.home() / "models" / "numcc"
SAM2_CKPTS   = Path.home() / "models" / "sam2"
ASSETS_DIR   = PROJECT_ROOT / "assets"
OUTPUTS_DIR  = PROJECT_ROOT / "data" / "outputs"
CACHE_DIR    = Path.home() / "models" / "nksr_cache"
DOCKER_NUMCC = "numcc:x86"
DOCKER_SAM2  = "sam2:x86"


def _abs(p: Path | None) -> Path | None:
    return p.resolve() if p else None


def run(cmd: list[str]) -> int:
    print("▶", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd).returncode


def run_sam2(image: Path, name: str) -> Path:
    """Corre SAM2 sobre image y devuelve la ruta del segmask.npy generado."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{image.parent}:/input:ro",
        "-v", f"{OUTPUTS_DIR}:/output",
        "-v", f"{SAM2_CKPTS}:/opt/sam2/checkpoints:ro",
        "-v", f"{PROJECT_ROOT / 'sam2' / 'pipeline.py'}:/opt/sam2/pipeline.py:ro",
        "--entrypoint", "python3",
        DOCKER_SAM2,
        "/opt/sam2/pipeline.py",
        "--input",  f"/input/{image.name}",
        "--output", "/output",
        "--name",   name,
    ]
    rc = run(cmd)
    if rc != 0:
        sys.exit(f"SAM2 failed (exit {rc})")

    mask_path = OUTPUTS_DIR / f"{name}_segmask.npy"
    if not mask_path.exists():
        sys.exit(f"ERROR: SAM2 no generó la máscara esperada en {mask_path}")

    print(f"  SAM2 mask: {mask_path}")
    return mask_path


def mode_full_pipeline(args):
    """NU-MCC full pipeline: depth + color → surface pts → mesh → SDF."""
    image      = _abs(args.image)
    depth      = _abs(args.depth)
    intrinsics = _abs(args.intrinsics)

    # ── SAM2 step ─────────────────────────────────────────────────────────────
    if args.mask:
        # Mask provided explicitly — skip SAM2
        mask = _abs(args.mask)
        print(f"[SAM2] skip — usando máscara provista: {mask}")
    elif args.skip_sam2:
        # Explicit skip — run without mask
        mask = None
        print("[SAM2] skip — corriendo sin máscara")
    elif image is not None:
        # Auto-run SAM2 to generate mask
        print(f"[SAM2] generando máscara para {image.name}...")
        raw_mask = run_sam2(image, args.name)
        # Copy mask to the depth directory (numcc requires depth + mask in same dir)
        mask = depth.parent / f"{args.name}_segmask.npy"
        shutil.copy2(raw_mask, mask)
        print(f"  máscara copiada a {mask}")
    else:
        mask = None
        print("[SAM2] skip — sin imagen de color")

    # ── Validate input directory constraint ───────────────────────────────────
    input_files = [f for f in [depth, mask, intrinsics] if f is not None]
    if not input_files:
        sys.exit("ERROR: --depth is required for full pipeline mode")

    input_dirs = set(f.parent for f in input_files)
    if len(input_dirs) > 1:
        sys.exit(
            "ERROR: depth, mask, and intrinsics must be in the same directory. "
            f"Found: {input_dirs}"
        )
    input_dir = input_files[0].parent

    # Color image can live in a different directory — mount separately
    if image is not None and image.parent != input_dir:
        color_mount = ["-v", f"{image.parent}:/color_dir:ro"]
        color_arg   = f"/color_dir/{image.name}"
    elif image is not None:
        color_mount = []
        color_arg   = f"/input/{image.name}"
    else:
        color_mount = []
        color_arg   = None

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{MODELS_DIR}:/opt/models:ro",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{ASSETS_DIR}:/output",
        "-v", f"{CACHE_DIR}:/root/.cache/torch",
        *color_mount,
        DOCKER_NUMCC,
        "--depth",  f"/input/{depth.name}",
        "--name",   args.name,
        "--output", "/output",
        "--mesh-method",   args.mesh_method,
        "--udf-threshold", str(args.udf_threshold),
        "--n-query",       str(args.n_query),
        "--udf-n-iter",    str(args.udf_n_iter),
        "--poisson-depth", str(args.poisson_depth),
    ]
    if color_arg:
        cmd += ["--color", color_arg]
    if mask:
        cmd += ["--mask", f"/input/{mask.name}"]
    if intrinsics:
        cmd += ["--intrinsics", f"/input/{intrinsics.name}"]
    if args.no_p2c:
        cmd += ["--no-p2c"]
    if args.no_floor_cap:
        cmd += ["--no-floor-cap"]

    rc = run(cmd)
    if rc != 0:
        sys.exit(f"numcc pipeline failed (exit {rc})")

    asset_path = ASSETS_DIR / args.name
    print(f"\n✓ Asset en: {asset_path}")
    print(f"  {args.name}.obj   [{args.mesh_method}]")
    print(f"  {args.name}.sdf")


def mode_remesh(args):
    """Solo remesh: load PLY → mesh → SDF (salta NU-MCC)."""
    cloud = _abs(args.cloud)
    if not cloud.exists():
        sys.exit(f"ERROR: cloud file not found: {cloud}")

    cloud_dir     = cloud.parent
    output_parent = cloud_dir.parent
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--entrypoint", "python3",
        "-v", f"{cloud_dir}:/asset_in:ro",
        "-v", f"{output_parent}:/output",
        "-v", f"{CACHE_DIR}:/root/.cache/torch",
        DOCKER_NUMCC,
        "/app/remesh.py",
        "--cloud",         f"/asset_in/{cloud.name}",
        "--name",          args.name,
        "--output",        "/output",
        "--mesh-method",   args.mesh_method,
        "--poisson-depth", str(args.poisson_depth),
    ]

    rc = run(cmd)
    if rc != 0:
        sys.exit(f"remesh failed (exit {rc})")

    asset_path = output_parent / args.name
    print(f"\n✓ Asset en: {asset_path}")
    print(f"  {args.name}.obj   [{args.mesh_method}]")
    print(f"  {args.name}.sdf")


def main():
    p = argparse.ArgumentParser(
        description="Pipeline numcc: foto + depth → OBJ + SDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── inputs ────────────────────────────────────────────────────────────────
    inp = p.add_argument_group("inputs (full pipeline)")
    inp.add_argument("--image",      type=Path, help="Imagen de color (.jpg, .png, .npy)")
    inp.add_argument("--depth",      type=Path, help="Depth map (.npy float32 m, o .png uint16 mm)")
    inp.add_argument("--mask",       type=Path, help="Máscara existente (.npy uint8) — salta SAM2")
    inp.add_argument("--intrinsics", type=Path, help="intrinsics.json con fx/fy/cx/cy (opcional)")

    rem = p.add_argument_group("inputs (remesh desde nube existente — salta NU-MCC)")
    rem.add_argument("--cloud",      type=Path,
                     help="PLY de superficie ya calculada (*_numcc_surface.ply)")

    # ── common ────────────────────────────────────────────────────────────────
    p.add_argument("--name",          required=True,
                   help="Nombre del asset (determina carpeta de salida)")
    p.add_argument("--mesh-method",   default="noksr",
                   choices=["poisson", "noksr"],
                   help="Método de reconstrucción de mesh (default: noksr)")
    p.add_argument("--poisson-depth", default=10, type=int,
                   help="Profundidad octree Poisson (default 10)")
    p.add_argument("--skip-sam2",     action="store_true",
                   help="No correr SAM2 aunque no haya --mask (corre numcc sin máscara)")

    # ── numcc pipeline args ───────────────────────────────────────────────────
    p.add_argument("--udf-threshold", default=0.23, type=float,
                   help="NU-MCC UDF threshold (default 0.23)")
    p.add_argument("--n-query",       default=200_000, type=int,
                   help="Puntos de query UDF (default 200000)")
    p.add_argument("--udf-n-iter",    default=3, type=int,
                   help="Iteraciones move_points (default 3)")
    p.add_argument("--no-p2c",        action="store_true",
                   help="Saltar etapa P2C (usar nube de depth directamente)")
    p.add_argument("--no-floor-cap", action="store_true",
                   help="Desactivar la tapa de fondo automática del plano de soporte "
                        "(por defecto activa cuando se usa --mask)")

    args = p.parse_args()

    if args.cloud:
        print(f"[remesh] {args.cloud.name} → {args.name}.obj  [{args.mesh_method}]")
        mode_remesh(args)
    elif args.depth:
        print(f"[full pipeline] {args.depth.name} → {args.name}.obj  [{args.mesh_method}]")
        mode_full_pipeline(args)
    else:
        p.error("Proveer --depth (pipeline completo) o --cloud (solo remesh)")


if __name__ == "__main__":
    main()
