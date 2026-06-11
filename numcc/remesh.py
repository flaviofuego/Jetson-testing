"""
Mesh-only step: load a surface point cloud PLY and generate OBJ + SDF.
Runs inside the Docker container via --entrypoint python3 /app/remesh.py.

Usage:
  docker run --rm --gpus all --entrypoint python3 \\
      -v /host/asset/dir:/asset \\
      -v ~/models/nksr_cache:/root/.cache/torch \\
      numcc:x86 /app/remesh.py \\
      --cloud /asset/nombre_numcc_surface.ply \\
      --name nombre \\
      --output /asset \\
      --mesh-method noksr
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from pipeline import (
    _points_to_mesh, _points_to_mesh_noksr, _points_to_mesh_sap,
    _points_to_mesh_lwmr,
)
from mesh_utils import normalize_mesh, decompose_convex
from sdf_generator import generate_sdf


def main():
    p = argparse.ArgumentParser(description="Mesh surface point cloud → OBJ + SDF")
    p.add_argument("--cloud",        required=True, type=Path,
                   help="Surface point cloud PLY (e.g. *_numcc_surface.ply)")
    p.add_argument("--name",         required=True,
                   help="Asset name (used for output file names)")
    p.add_argument("--output",       required=True, type=Path,
                   help="Output directory — files saved to OUTPUT/NAME/")
    p.add_argument("--mesh-method",  default="noksr",
                   choices=["poisson", "noksr", "sap", "lwmr"],
                   help="Mesh reconstruction method: noksr (nksr Neural Kernel, default), "
                        "poisson (Open3D Screened Poisson), "
                        "sap (Shape As Points / DPSR — fast spectral Poisson), "
                        "lwmr (LightweightMR CVPR 2025 — low-poly, ~10-30 min)")
    p.add_argument("--poisson-depth", default=10, type=int,
                   help="Poisson octree depth (default 10; 9=coarse, 11=fine)")
    p.add_argument("--sap-grid-res", default=256, type=int,
                   help="SAP/DPSR grid resolution (default 256)")
    p.add_argument("--sap-sigma", default=2.0, type=float,
                   help="SAP/DPSR Gaussian smoothing (default 2.0)")
    p.add_argument("--lwmr-sdf-iters", default=20_000, type=int,
                   help="LightweightMR SDF iterations (default 20000)")
    p.add_argument("--lwmr-vg-iters", default=8_000, type=int,
                   help="LightweightMR vertex-generation iterations (default 8000)")
    p.add_argument("--lwmr-vertices", default=3_400, type=int,
                   help="LightweightMR output vertex count (default 3400)")
    args = p.parse_args()

    t0 = time.time()

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(args.cloud))
    surface_pts = np.asarray(pcd.points).astype(np.float32)
    if len(surface_pts) == 0:
        sys.exit(f"ERROR: no points in {args.cloud}")
    print(f"Loaded {len(surface_pts):,} pts from {args.cloud}")

    asset_dir = args.output / args.name
    asset_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = asset_dir / f"{args.name}_parts"

    # ── mesh ─────────────────────────────────────────────────────────────────
    if args.mesh_method == "noksr":
        print("Meshing with nksr (Neural Kernel Surface Reconstruction)...")
        raw_mesh = _points_to_mesh_noksr(surface_pts)
    elif args.mesh_method == "sap":
        print(f"Meshing with SAP/DPSR (grid {args.sap_grid_res}^3)...")
        raw_mesh = _points_to_mesh_sap(surface_pts, grid_res=args.sap_grid_res,
                                       sigma=args.sap_sigma)
    elif args.mesh_method == "lwmr":
        print("Meshing with LightweightMR (per-object optimization, slow)...")
        raw_mesh = _points_to_mesh_lwmr(
            surface_pts, args.name, sdf_iters=args.lwmr_sdf_iters,
            vg_iters=args.lwmr_vg_iters, vertices_size=args.lwmr_vertices)
    else:
        print(f"Meshing with Poisson (depth={args.poisson_depth})...")
        raw_mesh = _points_to_mesh(surface_pts, poisson_depth=args.poisson_depth)
    print(f"  {len(raw_mesh.faces):,} faces  [{args.mesh_method}]")

    # ── normalize + export OBJ ────────────────────────────────────────────────
    mesh = normalize_mesh(raw_mesh)
    obj_path = asset_dir / f"{args.name}.obj"
    mesh.export(str(obj_path))
    print(f"Saved: {obj_path}  ({len(mesh.faces):,} faces)")

    # ── convex decomposition (CoACD) ──────────────────────────────────────────
    try:
        parts = decompose_convex(mesh, parts_dir, args.name)
    except Exception as e:
        print(f"CoACD failed ({e}), using convex hull")
        import trimesh
        hull = trimesh.convex.convex_hull(mesh)
        parts_dir.mkdir(parents=True, exist_ok=True)
        fallback = parts_dir / "convex_piece_000.obj"
        hull.export(str(fallback))
        parts = [fallback]
    print(f"{len(parts)} convex part(s)")

    # ── SDF ───────────────────────────────────────────────────────────────────
    relative_parts = [Path(f"{args.name}_parts") / pp.name for pp in parts]
    sdf_path = asset_dir / f"{args.name}.sdf"
    sdf_path.write_text(generate_sdf(args.name, mesh, relative_parts))
    print(f"Saved: {sdf_path}")

    print(f"\nDone ({time.time() - t0:.1f}s)  →  {asset_dir}/")


if __name__ == "__main__":
    main()
