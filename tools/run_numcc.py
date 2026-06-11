#!/usr/bin/env python3
"""
Pipeline numcc: photo + depth map → OBJ + SDF (Drake-ready).

Modos de uso:

  # Pipeline completo (NU-MCC + mesh desde foto y depth map)
  python3 tools/run_numcc.py \\
      --image  data/images/taza.jpg \\
      --depth  data/outputs/pipeline/taza/numcc_input/depth_full.npy \\
      --name   taza \\
      [--intrinsics data/outputs/pipeline/taza/numcc_input/intrinsics.json] \\
      [--mask  data/outputs/pipeline/taza/numcc_input/mask.npy] \\
      [--mesh-method noksr]

  # Solo remesh desde nube de puntos existente (salta NU-MCC)
  python3 tools/run_numcc.py \\
      --cloud  assets/taza/taza/taza_numcc_surface.ply \\
      --name   taza \\
      [--mesh-method noksr]

Salidas en assets/<name>/<name>.obj y assets/<name>/<name>.sdf
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

MODELS_DIR  = Path.home() / "models" / "numcc"
ASSETS_DIR  = Path(__file__).parent.parent / "assets"
CACHE_DIR   = Path.home() / "models" / "nksr_cache"
DOCKER_IMG  = "numcc:x86"


def _abs(p: Path | None) -> Path | None:
    return p.resolve() if p else None


def run(cmd: list[str]) -> int:
    print("▶", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd).returncode


def mode_full_pipeline(args):
    """NU-MCC full pipeline: depth + color → surface pts → mesh → SDF."""
    image      = _abs(args.image)
    depth      = _abs(args.depth)
    mask       = _abs(args.mask)
    intrinsics = _abs(args.intrinsics)

    # Collect all input files and resolve to a single input directory.
    # All files must be under the same directory so we can mount it as /input.
    input_files = [f for f in [depth, mask, intrinsics] if f is not None]
    if not input_files:
        sys.exit("ERROR: --depth is required for full pipeline mode")

    # Allow image in a different directory — mount separately as /color_img
    input_dirs = set(f.parent for f in input_files)
    if len(input_dirs) > 1:
        sys.exit(
            "ERROR: depth, mask, and intrinsics must be in the same directory. "
            f"Found: {input_dirs}"
        )
    input_dir = input_files[0].parent

    # Color image path inside container
    if image is not None and image.parent != input_dir:
        color_mount = ["-v", f"{image.parent}:/color_dir:ro"]
        color_arg   = f"/color_dir/{image.name}"
    elif image is not None:
        color_mount = []
        color_arg   = f"/input/{image.name}"
    else:
        color_mount = []
        color_arg   = None

    output_dir = ASSETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{MODELS_DIR}:/opt/models:ro",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{output_dir}:/output",
        "-v", f"{CACHE_DIR}:/root/.cache/torch",
        *color_mount,
        DOCKER_IMG,
        "--depth",  f"/input/{depth.name}",
        "--name",   args.name,
        "--output", "/output",
        "--mesh-method", args.mesh_method,
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

    rc = run(cmd)
    if rc != 0:
        sys.exit(f"numcc pipeline failed (exit {rc})")

    asset_path = output_dir / args.name
    print(f"\n✓ Asset en: {asset_path}")
    print(f"  {args.name}.obj   [{args.mesh_method}]")
    print(f"  {args.name}.sdf")


def mode_remesh(args):
    """Solo remesh: load PLY → mesh → SDF (salta NU-MCC)."""
    cloud = _abs(args.cloud)
    if not cloud.exists():
        sys.exit(f"ERROR: cloud file not found: {cloud}")

    # asset_dir is the parent of the PLY if it follows */<name>/<name>_*.ply
    # or any directory the user specifies. We mount cloud.parent as /asset.
    cloud_dir = cloud.parent
    output_parent = cloud_dir.parent   # e.g. assets/taza_hq
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--entrypoint", "python3",
        "-v", f"{cloud_dir}:/asset_in:ro",
        "-v", f"{output_parent}:/output",
        "-v", f"{CACHE_DIR}:/root/.cache/torch",
        DOCKER_IMG,
        "/app/remesh.py",
        "--cloud",        f"/asset_in/{cloud.name}",
        "--name",         args.name,
        "--output",       "/output",
        "--mesh-method",  args.mesh_method,
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
    inp.add_argument("--image",      type=Path, help="Foto / imagen de color (.jpg, .png, .npy)")
    inp.add_argument("--depth",      type=Path, help="Depth map (.npy float32 m, o .png uint16 mm)")
    inp.add_argument("--mask",       type=Path, help="Máscara SAM2 (.npy uint8, opcional)")
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

    # ── numcc pipeline args (solo relevantes en modo full) ───────────────────
    p.add_argument("--udf-threshold", default=0.23, type=float,
                   help="NU-MCC UDF threshold (default 0.23)")
    p.add_argument("--n-query",       default=200_000, type=int,
                   help="Puntos de query UDF (default 200000)")
    p.add_argument("--udf-n-iter",    default=3, type=int,
                   help="Iteraciones move_points (default 3)")
    p.add_argument("--no-p2c",        action="store_true",
                   help="Saltar etapa P2C (usar nube de depth directamente)")

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
