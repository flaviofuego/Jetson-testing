"""
Runs inside the Docker container.
Usage:
  python pipeline.py --depth /input/depth.npy --intrinsics /input/intrinsics.json \
                     --output /output --name mug
  python pipeline.py --depth /input/depth.png --fx 615 --fy 615 --cx 320 --cy 240 \
                     --output /output --name mug
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import time
from pathlib import Path
import numpy as np

from pointcloud_utils import load_depth, depth_to_pointcloud, subsample_pointcloud, load_intrinsics
from mesh_utils import normalize_mesh, decompose_convex
from sdf_generator import generate_sdf


def run_p2c(partial_pts: np.ndarray, n_output: int, weights_dir: Path) -> np.ndarray:
    """Complete partial point cloud with P2C. Returns (M, 3) float32 numpy array."""
    import torch
    import sys as _sys
    _sys.path.insert(0, "/opt/p2c")

    # Adjust import path per docs/numcc-api-notes.md if available
    from models.p2c import P2CModel
    model = P2CModel()

    ckpt = torch.load(str(weights_dir / "p2c_checkpoint.pth"), map_location="cuda")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval().cuda()

    with torch.no_grad():
        pts_t = torch.from_numpy(partial_pts).float().unsqueeze(0).cuda()  # (1, N, 3)
        out = model(pts_t)

        if isinstance(out, dict):
            completed = out.get("coarse_output", out.get("fine_output", list(out.values())[-1]))
        else:
            completed = out

    return completed.squeeze(0).cpu().numpy().astype(np.float32)  # (M, 3)


def run_numcc(completed_pts: np.ndarray, weights_dir: Path) -> "trimesh.Trimesh":
    """Reconstruct mesh from completed point cloud with NU-MCC."""
    import torch
    import trimesh
    import sys as _sys
    _sys.path.insert(0, "/opt/numcc")

    # Adjust import path per docs/numcc-api-notes.md if available
    from models.numcc import NUMCC
    model = NUMCC()

    ckpt = torch.load(str(weights_dir / "numcc_checkpoint.pth"), map_location="cuda")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval().cuda()

    with torch.no_grad():
        pts_t = torch.from_numpy(completed_pts).float().unsqueeze(0).cuda()
        result = model.extract_mesh(pts_t)

        if isinstance(result, trimesh.Trimesh):
            return result
        vertices, faces = result
        return trimesh.Trimesh(
            vertices=vertices.cpu().numpy() if hasattr(vertices, "cpu") else vertices,
            faces=faces.cpu().numpy() if hasattr(faces, "cpu") else faces,
        )


def main():
    parser = argparse.ArgumentParser(description="P2C + NU-MCC asset generation pipeline")
    parser.add_argument("--depth",      required=True, type=Path,
                        help="Depth file (.npy float32 meters or .png uint16 mm)")
    parser.add_argument("--color",      default=None,  type=Path,
                        help="Color file (.npy or .png), optional")
    parser.add_argument("--intrinsics", default=None,  type=Path,
                        help="intrinsics.json with fx/fy/cx/cy")
    parser.add_argument("--fx",         default=None,  type=float)
    parser.add_argument("--fy",         default=None,  type=float)
    parser.add_argument("--cx",         default=None,  type=float)
    parser.add_argument("--cy",         default=None,  type=float)
    parser.add_argument("--name",       required=True, type=str)
    parser.add_argument("--output",     required=True, type=Path)
    parser.add_argument("--n-input",    default=2048,  type=int,
                        help="Points sampled from depth map for P2C input")
    parser.add_argument("--n-output",   default=16384, type=int,
                        help="Target points in P2C completion output")
    args = parser.parse_args()

    if args.intrinsics is None and any(v is None for v in [args.fx, args.fy, args.cx, args.cy]):
        parser.error("Provide either --intrinsics or all of --fx --fy --cx --cy")

    asset_dir = args.output / args.name
    parts_dir = asset_dir / f"{args.name}_parts"
    asset_dir.mkdir(parents=True, exist_ok=True)
    models_root = Path("/opt/models")

    t_start = time.time()

    print(f"[1/6] Loading depth: {args.depth}")
    depth = load_depth(args.depth)

    print("[2/6] Back-projecting depth -> partial point cloud...")
    if args.intrinsics:
        fx, fy, cx, cy = load_intrinsics(args.intrinsics)
    else:
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    all_pts = depth_to_pointcloud(depth, fx=fx, fy=fy, cx=cx, cy=cy)
    partial_pts = subsample_pointcloud(all_pts, n=args.n_input)
    print(f"      {len(all_pts)} valid pixels -> {len(partial_pts)} sampled points")

    print(f"[3/6] P2C: completing point cloud ({args.n_input} -> ~{args.n_output} pts)...")
    completed_pts = run_p2c(partial_pts, n_output=args.n_output, weights_dir=models_root / "p2c")
    pc_path = asset_dir / f"{args.name}_pointcloud.npy"
    np.save(str(pc_path), completed_pts)
    print(f"      {len(completed_pts)} points -> saved {pc_path}")

    print("[4/6] NU-MCC: reconstructing mesh from point cloud...")
    raw_mesh = run_numcc(completed_pts, weights_dir=models_root / "numcc")
    print(f"      Raw mesh: {len(raw_mesh.faces)} faces")

    print("[5/6] Normalizing mesh (longest axis -> 20 cm)...")
    mesh = normalize_mesh(raw_mesh)
    obj_path = asset_dir / f"{args.name}.obj"
    mesh.export(str(obj_path))
    print(f"      Saved: {obj_path}  ({len(mesh.faces)} faces)")

    print("[5b/6] Convex decomposition...")
    try:
        parts = decompose_convex(mesh, parts_dir, args.name)
    except Exception as e:
        print(f"       coacd failed ({e}), falling back to convex hull")
        import trimesh
        hull = trimesh.convex.convex_hull(mesh)
        parts_dir.mkdir(parents=True, exist_ok=True)
        fallback = parts_dir / "convex_piece_000.obj"
        hull.export(str(fallback))
        parts = [fallback]
    print(f"      {len(parts)} convex part(s)")

    print("[6/6] Generating SDF...")
    relative_parts = [Path(f"{args.name}_parts") / p.name for p in parts]
    sdf_path = asset_dir / f"{args.name}.sdf"
    sdf_path.write_text(generate_sdf(args.name, mesh, relative_parts))
    print(f"      Saved: {sdf_path}")

    print(f"\nDone -> {asset_dir}/  ({time.time() - t_start:.1f}s total)")
    print(f"  {args.name}.obj")
    print(f"  {args.name}.sdf")
    print(f"  {args.name}_pointcloud.npy")
    print(f"  {args.name}_parts/")


if __name__ == "__main__":
    main()
