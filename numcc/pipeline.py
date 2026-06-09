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
    ckpt_path = weights_dir / "p2c_checkpoint.pth"
    ckpt = torch.load(str(ckpt_path), map_location="cuda")
    # P2C checkpoints may be saved as {'model': state_dict} or directly as state_dict
    state = ckpt.get("model", ckpt)
    if isinstance(state, dict) and "base_model" in state:
        state = state["base_model"]
    # Strip DataParallel 'module.' prefix if present
    if isinstance(state, dict) and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"      WARNING: P2C checkpoint architecture mismatch ({e}). "
              "Returning input points unchanged — NU-MCC will use raw depth cloud.")
        return partial_pts
    model.eval().cuda()

    with torch.no_grad():
        pts_t = torch.from_numpy(partial_pts).float().unsqueeze(0).cuda()  # (1, N, 3)
        completed = model(pts_t)  # (1, M, 3) — direct tensor output

    return completed.squeeze(0).cpu().numpy().astype(np.float32)  # (M, 3)


def run_numcc(
    color: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    weights_dir: Path,
    query_pts: np.ndarray | None = None,
    seen_mask: np.ndarray | None = None,
    udf_threshold: float = 0.05,
    n_query: int = 200_000,
    batch_size: int = 40_000,
) -> np.ndarray:
    """Reconstruct surface point cloud with NU-MCC. Returns (N, 3) float32 numpy array.

    NU-MCC expects:
      seen_images:    (B, 3, 800, 800) — RGB image (preprocess_img downscales to 224)
      seen_xyz:       (B, H, W, 3)     — per-pixel XYZ map at xyz_size resolution (112×112)
      valid_seen_xyz: (B, H, W)        — True where depth > 0 AND in SAM2 object mask
      query_xyz:      (B, Q, 3)        — 3D positions to evaluate UDF

    seen_mask: optional (H, W) uint8 array at ORIGINAL depth resolution.
        If provided, valid_seen_xyz is further restricted to object pixels only,
        so the encoder sees only the object surface — not the background scene.
        This matches the CO3D-V2 training distribution better.
    """
    import torch
    import torch.nn.functional as F
    import sys as _sys
    _sys.path.insert(0, "/opt/numcc")

    from src.model.nu_mcc import NUMCC
    from src.fns import shrink_points_beyond_threshold, preprocess_img

    XYZ_SIZE = 112  # must match checkpoint training resolution

    import argparse as _ap
    numcc_args = _ap.Namespace(
        nneigh=45,
        shrink_threshold=10.0,
        xyz_size=XYZ_SIZE,
        xyz_size_hr=224,
        hr=0,
        device="cuda",
        drop_path=0,
        n_groups=200,
        nn_seen=3,
        no_fine=0,
        regress_color=0,
        n_query_udf=batch_size,
        udf_threshold=udf_threshold,
        udf_n_iter=3,
    )

    model = NUMCC(args=numcc_args)
    ckpt = torch.load(str(weights_dir / "numcc_checkpoint.pth"), map_location="cuda")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval().cuda()

    # ── seen_images: (1, 3, 800, 800) ────────────────────────────────────────
    # Drop alpha channel if RGBA — model expects exactly 3 channels.
    color_rgb = color[:, :, :3]
    if color_rgb.dtype == np.uint8:
        color_f = color_rgb.astype(np.float32) / 255.0
    else:
        color_f = color_rgb.astype(np.float32).clip(0.0, 1.0)
    seen_images = torch.from_numpy(color_f.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    # preprocess_img (called later) asserts input is 800×800 before downscaling to 224.
    if seen_images.shape[2] != 800 or seen_images.shape[3] != 800:
        seen_images = F.interpolate(seen_images, size=(800, 800), mode="bilinear", align_corners=False)

    # ── xyz_map at 112×112: back-project downsampled depth ───────────────────
    H_orig, W_orig = depth.shape
    depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    # Nearest-neighbour preserves valid/invalid pixel boundaries exactly.
    depth_small = F.interpolate(depth_t, size=(XYZ_SIZE, XYZ_SIZE), mode="nearest")
    depth_small = depth_small.squeeze().numpy()  # (112, 112)

    # Scale intrinsics proportionally to the new resolution.
    scale_x = XYZ_SIZE / W_orig
    scale_y = XYZ_SIZE / H_orig
    fx_s, fy_s = fx * scale_x, fy * scale_y
    cx_s, cy_s = cx * scale_x, cy * scale_y

    v_idx, u_idx = np.mgrid[0:XYZ_SIZE, 0:XYZ_SIZE]
    Z = depth_small
    X = (u_idx - cx_s) * Z / fx_s
    Y = (v_idx - cy_s) * Z / fy_s
    xyz_map = np.stack([X, Y, Z], axis=-1).astype(np.float32)  # (112, 112, 3)
    xyz_map[Z == 0] = np.nan  # no depth → invalid

    # ── apply SAM2 mask at numpy level (before normalization) ────────────────
    # The demo (demo_iphone.py) masks *before* normalize(), so stats are computed
    # from object pixels only — not the background scene.
    if seen_mask is not None:
        from PIL import Image as PILImage
        m_img = PILImage.fromarray(seen_mask.astype(np.uint8) * 255).resize(
            (XYZ_SIZE, XYZ_SIZE), PILImage.NEAREST
        )
        mask_small_np = np.asarray(m_img) > 0  # (112, 112) bool
        xyz_map[~mask_small_np] = np.nan  # background pixels → invalid

    # ── normalize seen_xyz — matches demo_iphone.py normalize() ─────────────
    # CO3D-V2 training uses point clouds normalized to zero-mean, unit std.
    # Without this the XYZPosEmbed linear layer receives out-of-distribution
    # metric-scale coords (e.g. Z≈0.5 m) and produces garbage UDF predictions.
    valid_mask = np.isfinite(xyz_map).all(-1)   # (112, 112)
    valid_pts  = xyz_map[valid_mask]             # (K, 3)
    norm_center = np.zeros(3, dtype=np.float32)
    norm_scale  = 1.0
    if len(valid_pts) >= 3:
        per_axis_std = (valid_pts.var(axis=0) ** 0.5)   # (3,) — std per axis
        _scale = float(per_axis_std.mean())
        if _scale > 1e-6:
            norm_center = valid_pts.mean(axis=0)         # (3,)
            norm_scale  = _scale
            xyz_map[valid_mask] = (xyz_map[valid_mask] - norm_center) / norm_scale
            print(f"      normalize: center={norm_center.round(3)}  scale={norm_scale:.4f}")
        else:
            print("      WARNING: near-zero variance in seen_xyz — normalization skipped")
    else:
        print(f"      WARNING: only {len(valid_pts)} valid xyz pixels — normalization skipped")

    # Apply the same linear transform to query_pts so both are in the same space.
    if query_pts is None:
        query_pts = xyz_map[valid_mask]  # already normalized
    else:
        query_pts = (query_pts - norm_center) / norm_scale

    # ── convert to tensor ─────────────────────────────────────────────────────
    seen_xyz_t = torch.from_numpy(xyz_map).unsqueeze(0).cuda()  # (1, 112, 112, 3)
    valid_seen = torch.isfinite(seen_xyz_t.sum(-1))              # (1, 112, 112)
    # float('inf') as sentinel for invalid: shrink_points_beyond_threshold skips
    # non-finite values, and XYZPosEmbed overwrites them with invalid_xyz_token.
    seen_xyz_t[~valid_seen] = float('inf')
    print(f"      seen_xyz valid: {valid_seen.sum().item()} / {XYZ_SIZE**2} px"
          + (" (depth+mask)" if seen_mask is not None else " (depth only)"))

    # ── query grid in normalized coordinate space ─────────────────────────────
    # Use 0.3 padding in normalized units (matches the ±0.3 offset the demo uses
    # around anchor-predicted centers). Camera-frame query_pts were normalized above.
    padding = 0.3
    mins = query_pts.min(axis=0) - padding
    maxs = query_pts.max(axis=0) + padding
    n_side = int(np.cbrt(n_query))
    xs = np.linspace(mins[0], maxs[0], n_side)
    ys = np.linspace(mins[1], maxs[1], n_side)
    zs = np.linspace(mins[2], maxs[2], n_side)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    query_xyz_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).cuda()  # (1, Q, 3)

    # Encode once, decode in batches
    with torch.no_grad():
        seen_images_proc = preprocess_img(seen_images.clone())
        seen_xyz_shrunk = shrink_points_beyond_threshold(seen_xyz_t, numcc_args.shrink_threshold)
        query_xyz_shrunk = shrink_points_beyond_threshold(query_xyz_t, numcc_args.shrink_threshold)

        latent, up_grid_fea = model.encoder(seen_images_proc, seen_xyz_shrunk, valid_seen)
        fea = model.decoderl1(latent)

    total_q = query_xyz_shrunk.shape[1]
    surface_pts = []
    all_udf_vals = []
    for start in range(0, total_q, batch_size):
        end = min(start + batch_size, total_q)
        q_batch = query_xyz_shrunk[:, start:end]
        with torch.no_grad():
            pred = model.decoderl2(q_batch, seen_xyz_shrunk, valid_seen, fea, up_grid_fea)
            pred = model.fc_out(pred)
        udf = F.relu(pred[:, :, :1]).squeeze(-1)  # (1, Q_batch)
        all_udf_vals.append(udf[0].cpu())
        mask = udf[0] < udf_threshold
        pts = q_batch[0][mask].cpu().numpy()
        if len(pts) > 0:
            surface_pts.append(pts)

    all_udf = torch.cat(all_udf_vals).numpy()
    print(f"      UDF stats: min={all_udf.min():.4f}  p5={np.percentile(all_udf,5):.4f}"
          f"  p25={np.percentile(all_udf,25):.4f}  median={np.median(all_udf):.4f}"
          f"  p75={np.percentile(all_udf,75):.4f}  max={all_udf.max():.4f}")
    for t in [0.01, 0.03, 0.05, 0.10, 0.20]:
        print(f"      UDF < {t:.2f}: {(all_udf < t).sum()} pts")

    if not surface_pts:
        # Fallback: return normalized seen_xyz points — mesh_utils.normalize_mesh
        # rescales by longest axis anyway, so normalized coords are acceptable.
        valid = np.isfinite(xyz_map).all(-1)
        return xyz_map[valid]
    # Surface points are in normalized space; normalize_mesh handles rescaling.
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


def _save_ply(path: Path, pts: np.ndarray, colors: np.ndarray | None = None):
    """Write binary little-endian PLY. pts: (N,3) float32, colors: (N,3) uint8."""
    N = len(pts)
    has_color = colors is not None and len(colors) == N
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if has_color:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.zeros(N, dtype=dt)
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if has_color:
        data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    prop_lines = "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        prop_lines += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        f"{prop_lines}end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def _apply_mask_to_pointcloud(
    all_pts: np.ndarray, depth: np.ndarray, mask_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter back-projected points to only those inside the SAM2 segmask.

    Returns (filtered_pts, v_indices, u_indices) so callers can sample colors
    from the color image at the corresponding pixel locations.
    """
    mask = np.load(str(mask_path)).astype(bool)
    H, W = depth.shape
    if mask.shape != (H, W):
        from PIL import Image as PILImage
        m_img = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
            (W, H), PILImage.NEAREST
        )
        mask = np.asarray(m_img) > 0

    v_valid, u_valid = np.where(depth > 0)
    in_object = mask[v_valid, u_valid]
    filtered_pts = all_pts[in_object]
    v_obj = v_valid[in_object]
    u_obj = u_valid[in_object]
    print(f"      Mask filter: {len(all_pts)} → {len(filtered_pts)} object pts")
    return filtered_pts, v_obj, u_obj


def main():
    parser = argparse.ArgumentParser(description="NU-MCC asset generation pipeline")
    parser.add_argument("--depth",      required=True, type=Path,
                        help="Full (unmasked) depth file (.npy float32 meters or .png uint16 mm). "
                             "Used as-is for seen_xyz fed to NU-MCC.")
    parser.add_argument("--color",      default=None,  type=Path,
                        help="Color file (.npy uint8 or .png). Required for best NU-MCC quality.")
    parser.add_argument("--mask",       default=None,  type=Path,
                        help="SAM2 binary mask .npy (H×W uint8, 1=object). "
                             "Filters the back-projected point cloud AND restricts seen_xyz "
                             "to object pixels only (background set to invalid). "
                             "Normalization stats computed from object pixels only.")
    parser.add_argument("--intrinsics", default=None,  type=Path,
                        help="intrinsics.json with fx/fy/cx/cy")
    parser.add_argument("--fx",         default=None,  type=float)
    parser.add_argument("--fy",         default=None,  type=float)
    parser.add_argument("--cx",         default=None,  type=float)
    parser.add_argument("--cy",         default=None,  type=float)
    parser.add_argument("--name",       required=True, type=str)
    parser.add_argument("--output",     required=True, type=Path)
    parser.add_argument("--n-input",    default=2048,  type=int,
                        help="Object points sampled for P2C / query-grid bounding box")
    parser.add_argument("--no-p2c",    action="store_true",
                        help="Skip P2C completion — use raw (mask-filtered) depth cloud directly. "
                             "Recommended when P2C checkpoint coordinate system differs from camera frame.")
    parser.add_argument("--udf-threshold", default=0.05, type=float,
                        help="NU-MCC UDF threshold for surface extraction (default: 0.05)")
    args = parser.parse_args()

    if args.intrinsics is None and any(v is None for v in [args.fx, args.fy, args.cx, args.cy]):
        parser.error("Provide either --intrinsics or all of --fx --fy --cx --cy")

    asset_dir = args.output / args.name
    parts_dir = asset_dir / f"{args.name}_parts"
    asset_dir.mkdir(parents=True, exist_ok=True)
    models_root = Path("/opt/models")

    t_start = time.time()

    # ── [1] Load full (unmasked) depth + color ────────────────────────────────
    print(f"[1/6] Loading depth: {args.depth}")
    depth = load_depth(args.depth)          # full scene, no masking applied here
    color = _load_color(args.color, depth)
    print(f"      depth shape={depth.shape}  valid_px={int((depth > 0).sum())}  "
          f"range=[{depth[depth>0].min():.3f}, {depth[depth>0].max():.3f}] m")

    # ── [2] Back-project full depth → 3D cloud, then filter by SAM2 mask ─────
    print("[2/6] Back-projecting depth -> point cloud (full scene, then mask-filter)...")
    if args.intrinsics:
        fx, fy, cx, cy = load_intrinsics(args.intrinsics)
    else:
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    print(f"      intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    all_pts = depth_to_pointcloud(depth, fx=fx, fy=fy, cx=cx, cy=cy)
    print(f"      full scene: {len(all_pts)} pts  "
          f"X=[{all_pts[:,0].min():.3f},{all_pts[:,0].max():.3f}]  "
          f"Y=[{all_pts[:,1].min():.3f},{all_pts[:,1].max():.3f}]  "
          f"Z=[{all_pts[:,2].min():.3f},{all_pts[:,2].max():.3f}] m")

    if args.mask is not None:
        object_pts, v_obj, u_obj = _apply_mask_to_pointcloud(all_pts, depth, args.mask)
    else:
        object_pts = all_pts
        v_obj, u_obj = np.where(depth > 0)  # all valid pixels
        print("      (no mask — using full point cloud)")

    if len(object_pts) == 0:
        raise RuntimeError("No object points after mask filter. Check mask alignment.")

    # Sample RGB colors from the color image at the object pixel locations.
    # color may differ in resolution from depth — scale indices accordingly.
    color_H, color_W = color.shape[:2]
    depth_H, depth_W = depth.shape
    cv = (v_obj * (color_H / depth_H)).astype(int).clip(0, color_H - 1)
    cu = (u_obj * (color_W / depth_W)).astype(int).clip(0, color_W - 1)
    object_colors = color[cv, cu, :3].astype(np.uint8)  # (N, 3)

    # Save PLY of the SAM2-filtered object point cloud (all points, before subsampling)
    ply_path = asset_dir / f"{args.name}_object_cloud.ply"
    _save_ply(ply_path, object_pts, object_colors)
    print(f"      PLY saved: {ply_path}  ({len(object_pts)} pts, RGB)")

    partial_pts = subsample_pointcloud(object_pts, n=args.n_input)
    print(f"      object cloud: {len(object_pts)} pts -> {len(partial_pts)} subsampled  "
          f"X=[{partial_pts[:,0].min():.3f},{partial_pts[:,0].max():.3f}]  "
          f"Y=[{partial_pts[:,1].min():.3f},{partial_pts[:,1].max():.3f}]  "
          f"Z=[{partial_pts[:,2].min():.3f},{partial_pts[:,2].max():.3f}] m")

    # ── [3] Optional P2C completion ───────────────────────────────────────────
    if args.no_p2c:
        print("[3/6] P2C: SKIPPED (--no-p2c). Using mask-filtered depth cloud.")
        query_pts = partial_pts   # camera frame — correct for NU-MCC query grid
    else:
        print(f"[3/6] P2C: completing point cloud (~{args.n_input} -> 2048 pts)...")
        completed_pts = run_p2c(partial_pts, weights_dir=models_root / "p2c")
        print(f"      {len(completed_pts)} pts  "
              f"Z=[{completed_pts[:,2].min():.3f},{completed_pts[:,2].max():.3f}] m  "
              f"(negative Z → P2C re-centered; will use partial_pts for query grid)")
        # P2C re-centers the cloud to origin — incompatible coordinate system with seen_xyz.
        # Always use camera-frame partial_pts for the query grid bounding box.
        query_pts = partial_pts
        completed_pts_path = asset_dir / f"{args.name}_pointcloud.npy"
        np.save(str(completed_pts_path), completed_pts)
        print(f"      Saved P2C output: {completed_pts_path}")

    # ── [4] NU-MCC ────────────────────────────────────────────────────────────
    # Load the SAM2 mask (uint8, depth resolution) to restrict valid_seen in NU-MCC.
    seen_mask = None
    if args.mask is not None:
        seen_mask = np.load(str(args.mask)).astype(np.uint8)
        H_d, W_d = depth.shape
        if seen_mask.shape != (H_d, W_d):
            from PIL import Image as PILImage
            m_img = PILImage.fromarray(seen_mask * 255).resize((W_d, H_d), PILImage.NEAREST)
            seen_mask = (np.asarray(m_img) > 0).astype(np.uint8)

    print("[4/6] NU-MCC: reconstructing surface from RGB + depth XYZ map...")
    print(f"      seen_xyz source: full depth ({depth.shape}), {int((depth>0).sum())} valid px")
    print(f"      seen_mask: {'yes (object pixels only)' if seen_mask is not None else 'no (full scene)'}")
    print(f"      query grid anchor: {len(query_pts)} object pts (camera frame)")
    surface_pts = run_numcc(color, depth, fx, fy, cx, cy,
                            weights_dir=models_root / "numcc",
                            query_pts=query_pts,
                            seen_mask=seen_mask,
                            udf_threshold=args.udf_threshold,
                            n_query=50_000,
                            batch_size=4_000)
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
