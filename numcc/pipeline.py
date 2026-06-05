"""
Runs inside the Docker container.
Usage:
  python pipeline.py --depth /input/depth.npy --intrinsics /input/intrinsics.json \
                     --output /output --name mug
  python pipeline.py --depth /input/depth.png --color /input/color.npy \
                     --fx 615 --fy 615 --cx 320 --cy 240 --output /output --name mug

Note: --color is required for NU-MCC (RGB image provides texture cues for surface reconstruction).
      If not provided, a synthetic gray image is generated from the depth map.
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


def _load_color(color_path: Path | None, depth: np.ndarray) -> np.ndarray:
    """Load color image (H, W, 3) uint8. If not provided, synthesize gray from depth."""
    if color_path is None:
        h, w = depth.shape
        gray = np.clip(depth / depth.max() * 255, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=2)
    path = Path(color_path)
    if path.suffix == ".npy":
        return np.load(str(path))
    else:
        import imageio.v2 as imageio
        return imageio.imread(str(path))


def run_p2c(partial_pts: np.ndarray, weights_dir: Path) -> np.ndarray:
    """Complete partial point cloud with P2C. Returns (M, 3) float32 numpy array.

    P2C API (github.com/CuiRuikai/Partial2Complete):
      - Model class: P2C in models/P2C.py
      - Input: tensor (B, N, 3) float32
      - Output: tensor (B, M, 3) float32 — direct, no dict wrapper
      - Config must match training (default values used here).
    """
    import torch
    import sys as _sys
    _sys.path.insert(0, "/opt/p2c")

    from easydict import EasyDict
    from models.P2C import P2C

    # Default config matching P2C paper / standard training setup.
    # If your checkpoint was trained with different values, adjust here.
    config = EasyDict({
        "num_group": 128,
        "group_size": 32,
        "mask_ratio": [48, 48, 32],   # must sum to num_group=128
        "feat_dim": 1024,
        "n_points": 2048,             # output point count
        "nbr_ratio": 1,
        # loss params — required by __init__ even at inference time
        "support": 8,
        "neighborhood_size": 32,
        "shape_matching_weight": 10.0,
        "shape_recon_weight": 1.0,
        "latent_weight": 1.0,
        "manifold_weight": 0.1,
    })

    model = P2C(config)
    ckpt = torch.load(str(weights_dir / "p2c_checkpoint.pth"), map_location="cuda")
    # P2C checkpoints may be saved as {'model': state_dict} or directly as state_dict
    state = ckpt.get("model", ckpt)
    if isinstance(state, dict) and "base_model" in state:
        state = state["base_model"]
    model.load_state_dict(state)
    model.eval().cuda()

    with torch.no_grad():
        pts_t = torch.from_numpy(partial_pts).float().unsqueeze(0).cuda()  # (1, N, 3)
        completed = model(pts_t)  # (1, M, 3) — direct tensor output

    return completed.squeeze(0).cpu().numpy().astype(np.float32)  # (M, 3)


def run_numcc(
    color: np.ndarray,
    seen_xyz: np.ndarray,
    weights_dir: Path,
    udf_threshold: float = 0.03,
    n_query: int = 200_000,
    batch_size: int = 40_000,
) -> np.ndarray:
    """Reconstruct surface point cloud with NU-MCC. Returns (N, 3) float32 numpy array.

    NU-MCC API (github.com/sail-sg/numcc):
      - Model class: NUMCC in src/model/nu_mcc.py
      - Input: seen_images (B,3,H,W), seen_xyz (B,N,3), query_xyz (B,Q,3), valid_seen_xyz (B,N)
      - Output: UDF predictions at query_xyz — find surface where UDF < threshold
      - Checkpoint loaded via misc.load_model or direct state_dict.

    The P2C completed point cloud is used as seen_xyz to give NU-MCC full object coverage.
    A 3D query grid is generated around the point cloud bounding box.
    Surface points are extracted where predicted UDF < udf_threshold.
    """
    import torch
    import torch.nn.functional as F
    import sys as _sys
    _sys.path.insert(0, "/opt/numcc")

    from src.model.nu_mcc import NUMCC
    from src.fns import shrink_points_beyond_threshold, preprocess_img

    # Args namespace matching NU-MCC's expected parameters.
    # Adjust nneigh/shrink_threshold/xyz_size to match your checkpoint's training config.
    import argparse as _ap
    numcc_args = _ap.Namespace(
        nneigh=45,
        shrink_threshold=10.0,
        xyz_size=112,
        xyz_size_hr=224,
        hr=0,
        device="cuda",
        n_query_udf=batch_size,
        udf_threshold=udf_threshold,
        udf_n_iter=3,
    )

    model = NUMCC(args=numcc_args)
    ckpt = torch.load(str(weights_dir / "numcc_checkpoint.pth"), map_location="cuda")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval().cuda()

    # Prepare seen_images: (1, 3, H, W) float32 in [0, 1]
    if color.dtype == np.uint8:
        color_f = color.astype(np.float32) / 255.0
    else:
        color_f = color.astype(np.float32)
    seen_images = torch.from_numpy(color_f.transpose(2, 0, 1)).float().unsqueeze(0).cuda()

    # Prepare seen_xyz: (1, N, 3), valid mask: (1, N)
    seen_xyz_t = torch.from_numpy(seen_xyz).float().unsqueeze(0).cuda()
    valid_seen = (seen_xyz_t.abs().sum(-1) > 0)  # (1, N) — all provided points are valid

    # Generate 3D query grid around the bounding box of seen points
    padding = 0.05  # 5 cm beyond the object extent
    mins = seen_xyz.min(axis=0) - padding
    maxs = seen_xyz.max(axis=0) + padding
    n_side = int(np.cbrt(n_query))
    xs = np.linspace(mins[0], maxs[0], n_side)
    ys = np.linspace(mins[1], maxs[1], n_side)
    zs = np.linspace(mins[2], maxs[2], n_side)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    query_xyz_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).cuda()  # (1, Q, 3)

    # Encode once, decode in batches (NU-MCC query batching pattern from demo)
    with torch.no_grad():
        seen_images_proc = preprocess_img(seen_images.clone())
        seen_xyz_shrunk = shrink_points_beyond_threshold(seen_xyz_t, numcc_args.shrink_threshold)
        query_xyz_shrunk = shrink_points_beyond_threshold(query_xyz_t, numcc_args.shrink_threshold)

        latent, up_grid_fea = model.encoder(seen_images_proc, seen_xyz_shrunk, valid_seen)
        fea = model.decoderl1(latent)

    total_q = query_xyz_shrunk.shape[1]
    surface_pts = []
    for start in range(0, total_q, batch_size):
        end = min(start + batch_size, total_q)
        q_batch = query_xyz_shrunk[:, start:end]
        with torch.no_grad():
            pred = model.decoderl2(q_batch, seen_xyz_shrunk, valid_seen, fea, up_grid_fea)
            pred = model.fc_out(pred)
        udf = F.relu(pred[:, :, :1]).squeeze(-1)  # (1, Q_batch)
        mask = udf[0] < udf_threshold
        pts = q_batch[0][mask].cpu().numpy()
        if len(pts) > 0:
            surface_pts.append(pts)

    if not surface_pts:
        # No surface found — return original seen_xyz as fallback
        return seen_xyz
    return np.concatenate(surface_pts, axis=0)


def _points_to_mesh(surface_pts: np.ndarray) -> "trimesh.Trimesh":
    """Convert surface point cloud to mesh via Poisson reconstruction (open3d)."""
    import trimesh
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        # Remove low-density vertices (artifacts at mesh boundary)
        threshold = np.quantile(np.asarray(densities), 0.05)
        mesh_o3d.remove_vertices_by_mask(np.asarray(densities) < threshold)
        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception:
        # open3d unavailable — fall back to convex hull of surface points
        return trimesh.PointCloud(surface_pts).convex_hull


def main():
    parser = argparse.ArgumentParser(description="P2C + NU-MCC asset generation pipeline")
    parser.add_argument("--depth",      required=True, type=Path,
                        help="Depth file (.npy float32 meters or .png uint16 mm)")
    parser.add_argument("--color",      default=None,  type=Path,
                        help="Color file (.npy uint8 or .png). Required for best NU-MCC quality.")
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
                        help="Target points in P2C completion output (unused — P2C outputs n_points from config)")
    parser.add_argument("--udf-threshold", default=0.03, type=float,
                        help="NU-MCC UDF threshold for surface extraction (default: 0.03)")
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
    color = _load_color(args.color, depth)

    print("[2/6] Back-projecting depth -> partial point cloud...")
    if args.intrinsics:
        fx, fy, cx, cy = load_intrinsics(args.intrinsics)
    else:
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    all_pts = depth_to_pointcloud(depth, fx=fx, fy=fy, cx=cx, cy=cy)
    partial_pts = subsample_pointcloud(all_pts, n=args.n_input)
    print(f"      {len(all_pts)} valid pixels -> {len(partial_pts)} sampled points")

    print(f"[3/6] P2C: completing point cloud (~{args.n_input} -> 2048 pts)...")
    completed_pts = run_p2c(partial_pts, weights_dir=models_root / "p2c")
    pc_path = asset_dir / f"{args.name}_pointcloud.npy"
    np.save(str(pc_path), completed_pts)
    print(f"      {len(completed_pts)} points -> saved {pc_path}")

    print("[4/6] NU-MCC: reconstructing surface from RGB + point cloud...")
    surface_pts = run_numcc(color, completed_pts, weights_dir=models_root / "numcc",
                            udf_threshold=args.udf_threshold)
    print(f"      {len(surface_pts)} surface points")

    print("[4b/6] Meshing surface points (Poisson reconstruction)...")
    raw_mesh = _points_to_mesh(surface_pts)
    print(f"       Raw mesh: {len(raw_mesh.faces)} faces")

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
