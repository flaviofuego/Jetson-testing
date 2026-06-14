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

from pointcloud_utils import (
    load_depth, depth_to_pointcloud, subsample_pointcloud, load_intrinsics,
    to_y_up, CAM_TO_YUP,
)
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


def _pad_to_square(arr: np.ndarray, value: float) -> np.ndarray:
    """Pad an (H, W, C) array to a square (side = max(H, W)) by extending the
    bottom or right edge with `value`.

    Matches pad_image() in demo_iphone.py: a one-sided pad (not centered), so the
    object stays anchored at the top-left exactly like the reference before the
    final resize to the encoder resolution.
    """
    h, w = arr.shape[:2]
    if h == w:
        return arr
    if h > w:
        pad = np.full((h, h - w, arr.shape[2]), value, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)
    pad = np.full((w - h, w, arr.shape[2]), value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _save_img(path: Path, arr: np.ndarray):
    """Save an (H,W) or (H,W,3) array as PNG (min-max stretched if float)."""
    from PIL import Image as PILImage
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = np.where(np.isfinite(a), a, lo)
        a = ((a - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    PILImage.fromarray(a).save(str(path))


def _dump_numcc_io(
    debug_dir: Path,
    seen_images, seen_images_proc, seen_xyz_t, valid_seen,
    norm_center, norm_scale, grid_min, grid_max,
    all_udf, candidates, surface_pts, udf_threshold,
    anchors=None,
):
    """Persist a 'serie de copias' of NU-MCC's actual inputs and outputs so the
    reconstruction can be inspected for the taza (or any) input. Points are saved
    BOTH in normalized space (what NU-MCC sees) and de-normalized metric Y-up space
    (comparable to the depth cloud). Helps answer: is NU-MCC completing the object,
    or only re-tracing the seen surface?"""
    import json
    debug_dir.mkdir(parents=True, exist_ok=True)

    def to_metric_yup(pts_norm):
        m = np.asarray(pts_norm, np.float32) * norm_scale + norm_center
        return to_y_up(m)

    # ── INPUTS ───────────────────────────────────────────────────────────────
    si = seen_images[0].detach().cpu().numpy().transpose(1, 2, 0)         # 800×800×3
    _save_img(debug_dir / "01_input_seen_image.png", np.clip(si, 0, 1))
    sip = seen_images_proc[0].detach().cpu().numpy().transpose(1, 2, 0)   # 224×224×3 (norm)
    _save_img(debug_dir / "02_input_seen_image_proc.png", sip)

    vs = valid_seen[0].detach().cpu().numpy()                            # 112×112 bool
    _save_img(debug_dir / "03_input_seen_xyz_valid.png", vs.astype(np.uint8) * 255)
    sxyz = seen_xyz_t[0].detach().cpu().numpy()                          # 112×112×3
    z = sxyz[:, :, 2].copy(); z[~np.isfinite(z)] = np.nan
    _save_img(debug_dir / "04_input_seen_xyz_Z.png", z)

    seen_pts_norm = sxyz[vs]                                              # (K,3) normalized
    _save_ply(debug_dir / "05_input_seen_xyz.ply", to_metric_yup(seen_pts_norm))

    # ── OUTPUTS ──────────────────────────────────────────────────────────────
    if candidates is not None and len(candidates):
        _save_ply(debug_dir / "06_output_candidates.ply", to_metric_yup(candidates))
    _save_ply(debug_dir / "07_output_surface.ply", to_metric_yup(surface_pts))
    np.save(debug_dir / "08_output_udf_values.npy", all_udf)
    if anchors is not None:
        # The 200 anchor centers NU-MCC predicts (its internal shape hypothesis).
        _save_ply(debug_dir / "10_model_anchors.ply", to_metric_yup(anchors))

    # ── completion metric: does the surface go BEHIND the seen shell? ─────────
    seen_m = to_metric_yup(seen_pts_norm)
    surf_m = to_metric_yup(surface_pts)
    seen_zmin = float(seen_m[:, 2].min())
    behind = int((surf_m[:, 2] < seen_zmin - 0.005).sum())

    summary = {
        "norm_center": [float(x) for x in np.asarray(norm_center).ravel()],
        "norm_scale": float(norm_scale),
        "udf_threshold": float(udf_threshold),
        "seen_xyz_valid_px": int(vs.sum()),
        "candidates": int(len(candidates)) if candidates is not None else 0,
        "surface_pts": int(len(surface_pts)),
        "udf_min": float(all_udf.min()), "udf_median": float(np.median(all_udf)),
        "udf_max": float(all_udf.max()),
        "grid_min": [float(x) for x in np.asarray(grid_min).ravel()],
        "grid_max": [float(x) for x in np.asarray(grid_max).ravel()],
        "seen_Z_metric_min": seen_zmin,
        "surface_Z_metric_min": float(surf_m[:, 2].min()),
        "surface_pts_behind_seen": behind,
        "pct_completion": round(100.0 * behind / max(len(surf_m), 1), 2),
    }
    (debug_dir / "09_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"      [debug-dump] NU-MCC I/O → {debug_dir}  "
          f"(completion={summary['pct_completion']}% behind seen shell)")


def run_numcc(
    color: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    weights_dir: Path,
    query_pts: np.ndarray | None = None,
    seen_mask: np.ndarray | None = None,
    udf_threshold: float = 0.23,
    n_query: int = 200_000,
    batch_size: int = 40_000,
    n_iter: int = 10,
    debug_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reconstruct surface point cloud with NU-MCC.

    Returns (surface_pts, norm_center, norm_scale) where surface_pts is (N, 3) float32
    in normalized space, and norm_center/norm_scale are the normalization params used
    (needed by the caller to convert metric floor depths to normalized space).

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
    from src.fns import shrink_points_beyond_threshold, preprocess_img, move_points

    XYZ_SIZE = 112  # must match checkpoint training resolution

    import argparse as _ap
    numcc_args = _ap.Namespace(
        nneigh=4,              # training default (was 45 — wrong, changes attention distribution)
        shrink_threshold=10.0,
        xyz_size=XYZ_SIZE,
        xyz_size_hr=224,
        hr=0,
        device="cuda",
        drop_path=0,
        n_groups=200,
        nn_seen=4,             # training default (was 3 — lowered to avoid OOM from -1)
        no_fine=0,
        regress_color=0,
        n_query_udf=batch_size,
        udf_threshold=udf_threshold,
        udf_n_iter=n_iter,
        repulsive=1,           # enable repulsive force in move_points (training default)
        distributed=False,
    )

    model = NUMCC(args=numcc_args)
    ckpt = torch.load(str(weights_dir / "numcc_checkpoint.pth"), map_location="cuda")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval().cuda()

    # ── back-project FULL-resolution depth → per-pixel metric XYZ map ─────────
    # Build the XYZ map at full resolution FIRST. This is what lets the later crop
    # zoom into the object and still fill the 112×112 grid — downsampling the whole
    # frame to 112 up front (the previous approach) left only a tiny patch of valid
    # object pixels in a corner, which is out-of-distribution for NU-MCC.
    H_orig, W_orig = depth.shape
    v_idx, u_idx = np.mgrid[0:H_orig, 0:W_orig]
    Zf = depth.astype(np.float32)
    Xf = (u_idx - cx) * Zf / fx
    Yf = (v_idx - cy) * Zf / fy
    xyz_full = np.stack([Xf, Yf, Zf], axis=-1).astype(np.float32)  # (H, W, 3)
    xyz_full[Zf == 0] = np.nan  # no depth → invalid

    # ── restrict to object pixels (SAM2 mask) at full resolution ─────────────
    # demo_iphone.py masks *before* normalize(), so stats come from object pixels.
    if seen_mask is not None:
        mask_full = seen_mask.astype(bool)
        if mask_full.shape != (H_orig, W_orig):
            from PIL import Image as PILImage
            m_img = PILImage.fromarray(mask_full.astype(np.uint8) * 255).resize(
                (W_orig, H_orig), PILImage.NEAREST
            )
            mask_full = np.asarray(m_img) > 0
        xyz_full[~mask_full] = np.nan  # background pixels → invalid

    # ── normalize seen_xyz — matches demo_iphone.py normalize() ──────────────
    # CO3D-V2 training uses point clouds normalized to zero-mean, unit-std (single
    # scalar scale = mean of per-axis std). Without this the XYZPosEmbed linear
    # layer receives out-of-distribution metric coords and produces garbage UDF.
    valid_full  = np.isfinite(xyz_full).all(-1)   # (H, W)
    valid_pts   = xyz_full[valid_full]            # (K, 3)
    norm_center = np.zeros(3, dtype=np.float32)
    norm_scale  = 1.0
    if len(valid_pts) >= 3:
        _scale = float((valid_pts.var(axis=0) ** 0.5).mean())
        if _scale > 1e-6:
            norm_center = valid_pts.mean(axis=0)         # (3,)
            norm_scale  = _scale
            xyz_full[valid_full] = (xyz_full[valid_full] - norm_center) / norm_scale
            print(f"      normalize: center={norm_center.round(3)}  scale={norm_scale:.4f}")
        else:
            print("      WARNING: near-zero variance in seen_xyz — normalization skipped")
    else:
        print(f"      WARNING: only {len(valid_pts)} valid xyz pixels — normalization skipped")

    # ── crop to object bbox (+margin), pad to square, resize → 112 ───────────
    # This is the key step from demo_iphone.py main_demo(): the object is cropped
    # and zoomed so it FILLS the 112×112 encoder grid instead of occupying a tiny
    # patch of the full frame. Both seen_xyz and seen_images use the SAME bbox so
    # the image and geometry stay spatially aligned for the encoder fusion.
    ys, xs = np.where(valid_full)
    if len(ys) == 0:
        raise RuntimeError("No valid object pixels for seen_xyz — check mask/depth alignment.")
    MARGIN = 40  # pixels, matches demo_iphone.py
    top    = max(int(ys.min()) - MARGIN, 0)
    bottom = min(int(ys.max()) + MARGIN, H_orig - 1)
    left   = max(int(xs.min()) - MARGIN, 0)
    right  = min(int(xs.max()) + MARGIN, W_orig - 1)
    xyz_crop = xyz_full[top:bottom + 1, left:right + 1]   # (h, w, 3)
    print(f"      object bbox: rows[{top}:{bottom}] cols[{left}:{right}]  "
          f"crop={xyz_crop.shape[0]}×{xyz_crop.shape[1]} of {H_orig}×{W_orig}")

    xyz_sq = _pad_to_square(xyz_crop, np.nan)             # square, nan-padded
    # Nearest keeps valid/invalid boundaries crisp (no nan bleeding into neighbours).
    xyz_map = F.interpolate(
        torch.from_numpy(xyz_sq).permute(2, 0, 1).unsqueeze(0),
        size=(XYZ_SIZE, XYZ_SIZE), mode="nearest",
    ).squeeze(0).permute(1, 2, 0).numpy()                 # (112, 112, 3)

    # ── seen_images: crop the SAME bbox (in color resolution), pad, resize → 800 ─
    # Drop alpha channel if RGBA — model expects exactly 3 channels.
    color_rgb = color[:, :, :3]
    if color_rgb.dtype == np.uint8:
        color_f = color_rgb.astype(np.float32) / 255.0
    else:
        color_f = color_rgb.astype(np.float32).clip(0.0, 1.0)
    cH, cW = color_f.shape[:2]
    sy, sx = cH / H_orig, cW / W_orig
    ct = max(int(round(top * sy)), 0)
    cb = min(int(round((bottom + 1) * sy)), cH)
    cl = max(int(round(left * sx)), 0)
    cr = min(int(round((right + 1) * sx)), cW)
    color_sq = _pad_to_square(color_f[ct:cb, cl:cr], 0.0)  # black pad (matches reference)
    seen_images = torch.from_numpy(color_sq.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    # preprocess_img (called later) asserts input is 800×800 before downscaling to 224.
    seen_images = F.interpolate(seen_images, size=(800, 800), mode="bilinear", align_corners=False)

    # Apply the same linear transform to query_pts so both are in the same space.
    if query_pts is None:
        query_pts = xyz_full[valid_full]  # already normalized
    else:
        query_pts = (query_pts - norm_center) / norm_scale

    # ── convert to tensor ─────────────────────────────────────────────────────
    seen_xyz_t = torch.from_numpy(np.ascontiguousarray(xyz_map)).unsqueeze(0).cuda()  # (1,112,112,3)
    valid_seen = torch.isfinite(seen_xyz_t.sum(-1))              # (1, 112, 112)
    # float('inf') as sentinel for invalid: shrink_points_beyond_threshold skips
    # non-finite values, and XYZPosEmbed overwrites them with invalid_xyz_token.
    seen_xyz_t[~valid_seen] = float('inf')
    print(f"      seen_xyz valid: {valid_seen.sum().item()} / {XYZ_SIZE**2} px"
          + (" (cropped+zoomed object)" if seen_mask is not None else " (full frame)"))

    # ── encode once ───────────────────────────────────────────────────────────
    with torch.no_grad():
        seen_images_proc = preprocess_img(seen_images.clone())
        seen_xyz_shrunk = shrink_points_beyond_threshold(seen_xyz_t, numcc_args.shrink_threshold)

        latent, up_grid_fea = model.encoder(seen_images_proc, seen_xyz_shrunk, valid_seen)
        fea = model.decoderl1(latent)

    # ── query grid: anchored to model-predicted centers (demo_iphone.py approach) ──
    # fea['anchors_xyz'] are the 200 anchor points the decoder predicts after l1.
    # Using them (± offset) focuses the query grid where the model says the surface is,
    # rather than relying on the depth bounding box which can be noisy.
    centers_xyz = fea["anchors_xyz"]  # (1, 200, 3) in normalized space
    offset = 0.3
    c_min = centers_xyz[0].min(dim=0).values - offset  # (3,)
    c_max = centers_xyz[0].max(dim=0).values + offset  # (3,)

    # Widen to also cover the depth back-projection bounding box so we don't
    # miss regions the depth sees but the anchors haven't centered on yet.
    bb_min = torch.tensor(query_pts.min(axis=0) - offset, device="cuda")
    bb_max = torch.tensor(query_pts.max(axis=0) + offset, device="cuda")
    grid_min = torch.min(c_min, bb_min).cpu().numpy()
    grid_max = torch.max(c_max, bb_max).cpu().numpy()

    n_side = int(np.cbrt(n_query))
    xs = np.linspace(grid_min[0], grid_max[0], n_side)
    ys = np.linspace(grid_min[1], grid_max[1], n_side)
    zs = np.linspace(grid_min[2], grid_max[2], n_side)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    query_xyz_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).cuda()  # (1, Q, 3)
    query_xyz_shrunk = shrink_points_beyond_threshold(query_xyz_t, numcc_args.shrink_threshold)
    print(f"      query grid: {len(grid):,} pts  "
          f"X=[{grid_min[0]:.2f},{grid_max[0]:.2f}]  "
          f"Y=[{grid_min[1]:.2f},{grid_max[1]:.2f}]  "
          f"Z=[{grid_min[2]:.2f},{grid_max[2]:.2f}]  (normalized)")

    # ── pass 1: evaluate UDF on query grid, collect candidate points ─────────
    total_q = query_xyz_shrunk.shape[1]
    candidate_batches = []
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
        if mask.sum() > 0:
            candidate_batches.append(q_batch[0][mask])  # keep as tensor on GPU

    all_udf = torch.cat(all_udf_vals).numpy()
    print(f"      UDF stats: min={all_udf.min():.4f}  p5={np.percentile(all_udf,5):.4f}"
          f"  p25={np.percentile(all_udf,25):.4f}  median={np.median(all_udf):.4f}"
          f"  p75={np.percentile(all_udf,75):.4f}  max={all_udf.max():.4f}")
    for t in [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]:
        print(f"      UDF < {t:.2f}: {(all_udf < t).sum()} pts")

    if not candidate_batches:
        # Fallback: no query point fell below the UDF threshold. Return the
        # normalized seen_xyz points so the caller still gets a (pts, center, scale)
        # tuple instead of crashing on tuple-unpacking.
        valid = np.isfinite(xyz_map).all(-1)
        print("      WARNING: no UDF candidates below threshold — returning seen_xyz cloud")
        return xyz_map[valid], norm_center, norm_scale

    # ── pass 2: move_points — gradient descent to snap candidates to surface ──
    # Each point moves toward the zero-level set: x ← x - ∇UDF * UDF(x)
    # Equivalent to what demo_iphone.py does per-batch before returning results.
    refined = []
    for cand in candidate_batches:
        pts = cand.unsqueeze(0)  # (1, N, 3)
        pts = move_points(model, pts, seen_xyz_shrunk, valid_seen, fea, up_grid_fea,
                          numcc_args, n_iter=numcc_args.udf_n_iter)
        refined.append(pts.detach().squeeze(0).cpu().numpy().astype(np.float32))

    surface_pts = np.concatenate(refined, axis=0)

    if debug_dir is not None:
        cand_all = torch.cat([c for c in candidate_batches], dim=0).cpu().numpy()
        _dump_numcc_io(
            debug_dir, seen_images, seen_images_proc, seen_xyz_t, valid_seen,
            norm_center, norm_scale, grid_min, grid_max,
            all_udf, cand_all, surface_pts, udf_threshold,
            anchors=centers_xyz[0].detach().cpu().numpy(),
        )

    return surface_pts, norm_center, norm_scale


def _slice_and_cap_at_floor(mesh: "trimesh.Trimesh", z_floor_norm: float) -> "trimesh.Trimesh":
    """Cut the mesh at the floor plane and add a shape-following cap.

    Uses trimesh.intersections.slice_mesh_plane with cap=True, which computes the
    cross-section of the mesh at z=z_floor_norm and closes it with a Delaunay-
    triangulated polygon that follows the object's contour — NOT a flat rectangular
    plane, but the actual silhouette shape of the object at that height.
    """
    import trimesh

    # Camera looks down: small Z = top of object (near camera), large Z = bin (far).
    # We want to KEEP the near side (z < z_floor_norm) and cap the cut at the bin level.
    plane_normal = np.array([0.0, 0.0, -1.0])  # keep vertices with z < z_floor_norm
    plane_origin = np.array([0.0, 0.0, z_floor_norm])
    try:
        # Step 1: slice without cap
        mesh_cut = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal, plane_origin, cap=False
        )
        n_removed = len(mesh.faces) - len(mesh_cut.faces)

        # Step 2: find boundary edges (appear only once = open boundary at cut)
        edges_sorted = np.sort(mesh_cut.edges, axis=1)
        unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        boundary = unique_edges[counts == 1]
        if len(boundary) == 0:
            print(f"      floor cap: cut at Z_norm={z_floor_norm:.3f}  "
                  f"removed {n_removed} faces  no boundary — mesh already closed")
            return mesh_cut

        # Step 3: trace all boundary loops using proper adjacency traversal
        adj = {}
        for a, b in boundary.tolist():
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        visited_edges: set = set()
        loops: list = []
        for start in list(adj.keys()):
            # skip if all edges from this vertex already consumed
            if all((min(start, n), max(start, n)) in visited_edges
                   for n in adj[start]):
                continue
            loop = [start]
            prev, cur = None, start
            while True:
                nexts = [n for n in adj[cur]
                         if (min(cur, n), max(cur, n)) not in visited_edges
                         and n != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                visited_edges.add((min(cur, nxt), max(cur, nxt)))
                if nxt == start:
                    break
                loop.append(nxt)
                prev, cur = cur, nxt
            if len(loop) >= 3:
                loops.append(loop)

        if not loops:
            print(f"      floor cap: cut at Z_norm={z_floor_norm:.3f}  "
                  f"removed {n_removed} faces  no valid loops found")
            return mesh_cut

        # Step 4: triangulate loops using shapely + interpolate Z from actual boundary.
        #
        # The boundary has varying Z (the object is sloped). Instead of a flat cap at
        # z_floor_norm, we:
        #   1. Triangulate the XY footprint with shapely (handles noise, multiple loops)
        #   2. For every triangulated vertex, interpolate its Z from the actual boundary
        #      vertices using scipy linear interpolation — so the cap follows the real
        #      3D terrain of the boundary instead of a flat plane.
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
        from scipy.interpolate import griddata

        # Collect boundary control points (XY → Z) from all significant loops.
        # Used for Z interpolation of interior cap vertices.
        raw_polys  = []
        ctrl_xy_list: list = []
        ctrl_z_list:  list = []

        for loop in loops:
            pts = mesh_cut.vertices[loop]           # (N, 3) real 3D positions
            poly = ShapelyPolygon(pts[:, :2])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area > 1e-8:
                raw_polys.append(poly)
                ctrl_xy_list.append(pts[:, :2])
                ctrl_z_list.append(pts[:, 2])

        if not raw_polys:
            print(f"      floor cap: cut at Z_norm={z_floor_norm:.3f}  "
                  f"removed {n_removed} faces  no valid polygons")
            return mesh_cut

        # Z interpolation grid from all boundary vertices
        ctrl_xy = np.vstack(ctrl_xy_list)
        ctrl_z  = np.concatenate(ctrl_z_list)

        max_area = max(p.area for p in raw_polys)
        significant = [p for p in raw_polys if p.area >= 0.01 * max_area]
        buf = max_area ** 0.5 * 0.05
        merged = unary_union([p.buffer(buf) for p in significant]).buffer(-buf * 0.5).buffer(0)

        # Tight polygon for penetration guard: small buffer (0.75% radius) for numeric
        # tolerance only — does NOT bridge concave gaps like the 5% buf above does.
        from shapely.geometry import Point as _ShapelyPoint
        original_for_guard = unary_union(significant).buffer(buf * 0.15).buffer(0)

        geoms = list(getattr(merged, "geoms", [merged]))
        cap_verts_list: list = []
        cap_faces_list: list = []
        n_base = len(mesh_cut.vertices)
        offset = 0

        for poly in geoms:
            if poly.is_empty or poly.area < 1e-8:
                continue
            try:
                v2d, f2d = trimesh.creation.triangulate_polygon(poly, engine="earcut")
            except Exception:
                continue
            if v2d is None or len(f2d) == 0:
                continue
            # Interpolate Z for each cap vertex from the real boundary Z values.
            # 'linear' = smooth surface; fall back to 'nearest' for points outside hull.
            z_lin = griddata(ctrl_xy, ctrl_z, v2d, method="linear")
            z_nn  = griddata(ctrl_xy, ctrl_z, v2d, method="nearest")
            z_interp = np.where(np.isnan(z_lin), z_nn, z_lin)
            v3d = np.column_stack([v2d, z_interp])
            cap_verts_list.append(v3d)
            cap_faces_list.append(f2d + n_base + offset)
            offset += len(v3d)

        if not cap_faces_list:
            print(f"      floor cap: cut at Z_norm={z_floor_norm:.3f}  "
                  f"removed {n_removed} faces  triangulation produced no faces")
            return mesh_cut

        all_verts = np.vstack([mesh_cut.vertices] + cap_verts_list)
        all_cap_faces = np.vstack(cap_faces_list)

        # Penetration guard: remove cap triangles whose centroid falls outside the
        # original (un-buffered) mesh footprint. The 5%-buffered 'merged' polygon
        # bridges concave gaps (e.g. headphone arch, space between ear cups), producing
        # triangles that intersect the mesh walls. The tight 'original_for_guard'
        # (0.75% buffer) rejects those without affecting genuine cap fill.
        centroids_2d = all_verts[all_cap_faces].mean(axis=1)[:, :2]
        guard_mask = np.array(
            [original_for_guard.contains(_ShapelyPoint(float(c[0]), float(c[1])))
             for c in centroids_2d]
        )
        n_guard_removed = int(np.sum(~guard_mask))
        if n_guard_removed > 0:
            print(f"      floor cap: penetration guard removed {n_guard_removed} face(s)")
            all_cap_faces = all_cap_faces[guard_mask]
        if len(all_cap_faces) == 0:
            print(f"      floor cap: all faces removed by guard — returning cut mesh")
            return mesh_cut

        # Ensure cap normals point outward (+Z toward bin)
        for i, f in enumerate(all_cap_faces):
            v = all_verts[f]
            nz = (v[1,0]-v[0,0])*(v[2,1]-v[0,1]) - (v[1,1]-v[0,1])*(v[2,0]-v[0,0])
            if nz < 0:
                all_cap_faces[i] = f[::-1]

        all_faces = np.vstack([mesh_cut.faces, all_cap_faces])
        mesh_final = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)

        n_cap = len(all_cap_faces)
        print(f"      floor cap: cut at Z_norm={z_floor_norm:.3f}  "
              f"removed {n_removed} faces  {len(loops)} loop(s)  cap +{n_cap} faces  "
              f"watertight={mesh_final.is_watertight}")
        return mesh_final
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"      floor cap: failed ({e}) — returning original mesh")
        return mesh


def _points_to_mesh_noksr(surface_pts: np.ndarray) -> "trimesh.Trimesh":
    """Surface reconstruction with nksr (Neural Kernel Surface Reconstruction).

    nksr learns a kernel-based implicit field — handles noisy/sparse clouds better
    than Poisson because it incorporates a learned prior over surface shapes.
    No pretrained checkpoint needed; uses a GPU kernel solver at runtime.

    Fallback (when nksr not installed): Ball-Pivoting Algorithm (BPA) via Open3D.
    BPA is also different from Poisson — it rolls a virtual sphere over the point
    cloud and creates faces where the sphere touches 3 points simultaneously, so
    it stays closer to the actual input points rather than fitting a smooth implicit.
    """
    import trimesh
    import open3d as o3d

    # ── prepare normals (needed by both nksr and BPA) ────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_pts)
    extent = surface_pts.max(axis=0) - surface_pts.min(axis=0)
    density = len(surface_pts) / max(float(np.prod(extent)), 1e-6)
    radius_n = float((1.0 / density) ** (1 / 3)) * 3.0
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_n, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    # ── attempt nksr ──────────────────────────────────────────────────────────
    try:
        import torch
        import nksr
        # Verify this is the real nksr (stub from PyPI has no Reconstructor)
        if not hasattr(nksr, "Reconstructor"):
            raise ImportError("nksr stub installed (PyPI placeholder) — real wheel unavailable")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pts_t     = torch.from_numpy(surface_pts).float().to(device)
        normals   = np.asarray(pcd.normals).astype(np.float32)
        normals_t = torch.from_numpy(normals).float().to(device)

        reconstructor = nksr.Reconstructor(device)
        # detail_level: higher → finer octree (1.0 default, 2.0 = ~4× more cells)
        field     = reconstructor.reconstruct(pts_t, normal=normals_t, detail_level=1.0)
        mesh_nksr = field.extract_dual_mesh(mise_iter=1)

        verts = mesh_nksr.v.cpu().numpy()
        faces = mesh_nksr.f.cpu().numpy()
        print(f"       nksr: {len(verts)} verts, {len(faces)} faces")
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    except ImportError as e:
        print(f"      nksr unavailable ({e}) — using BPA (Ball-Pivoting Algorithm)")
    except Exception as e:
        print(f"      nksr failed ({e}) — using BPA (Ball-Pivoting Algorithm)")

    # ── fallback: Ball-Pivoting Algorithm (open3d) ───────────────────────────
    # BPA radius: ~2× average spacing so the ball bridges neighbouring points.
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_dist  = float(np.mean(distances))
    radii     = [avg_dist * 2.0, avg_dist * 4.0]   # two passes: fine + coarse
    mesh_bpa  = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh_bpa.remove_degenerate_triangles()
    mesh_bpa.remove_duplicated_vertices()
    verts = np.asarray(mesh_bpa.vertices)
    faces = np.asarray(mesh_bpa.triangles)
    print(f"       BPA: {len(verts)} verts, {len(faces)} faces  (avg spacing={avg_dist:.4f})")
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _points_to_mesh(surface_pts: np.ndarray, poisson_depth: int = 10) -> "trimesh.Trimesh":
    """Convert surface point cloud to mesh via Poisson reconstruction (open3d).

    poisson_depth controls octree depth — higher = finer mesh but slower:
      9  → coarse  (~30K faces typical)
      10 → medium  (~100K faces typical)  [default]
      11 → fine    (~300K faces typical)
    """
    import trimesh
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_pts)

        # Adaptive normal estimation: radius = 5× median nearest-neighbour distance.
        # Avoids under/over-smoothing when points are not uniformly distributed.
        nn_dists = np.sort(
            np.linalg.norm(surface_pts[None] - surface_pts[:, None], axis=-1), axis=1
        )[:, 1] if len(surface_pts) < 5000 else None  # skip brute-force for large clouds

        if nn_dists is not None:
            radius = float(np.median(nn_dists)) * 5.0
        else:
            # Estimate from point cloud extent
            extent = surface_pts.max(axis=0) - surface_pts.min(axis=0)
            density = len(surface_pts) / max(np.prod(extent), 1e-6)
            radius = float((1.0 / density) ** (1 / 3)) * 3.0

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
        )
        pcd.orient_normals_consistent_tangent_plane(30)

        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        # Remove low-density boundary artifacts (bottom 2% — conservative to keep surface)
        threshold = np.quantile(np.asarray(densities), 0.02)
        mesh_o3d.remove_vertices_by_mask(np.asarray(densities) < threshold)
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_vertices()
        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception:
        return trimesh.PointCloud(surface_pts).convex_hull


def _clip_by_mask_silhouette(
    mesh: "trimesh.Trimesh",
    norm_center: np.ndarray,
    norm_scale: float,
    fx: float, fy: float, cx: float, cy: float,
    mask: np.ndarray,
    dilation_px: int = 15,
) -> "trimesh.Trimesh":
    """Remove faces that project outside the SAM2 mask when viewed from the camera.

    Projects each vertex from NU-MCC normalized space → metric camera space → pixel.
    Keeps a face if AT LEAST ONE of its vertices projects inside the dilated mask.
    Dilation avoids over-clipping at the object silhouette boundary.
    """
    import trimesh
    from scipy.ndimage import binary_dilation

    mask_bin = mask.astype(bool)
    if dilation_px > 0:
        struct = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), dtype=bool)
        mask_bin = binary_dilation(mask_bin, structure=struct)
    H, W = mask_bin.shape

    # Normalized → metric camera space
    verts_m = mesh.vertices * norm_scale + norm_center  # (N, 3)
    X, Y, Z = verts_m[:, 0], verts_m[:, 1], verts_m[:, 2]

    valid = Z > 0
    u = np.where(valid, (fx * X / np.where(valid, Z, 1)) + cx, -1.0)
    v = np.where(valid, (fy * Y / np.where(valid, Z, 1)) + cy, -1.0)

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    in_bounds = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    in_mask = np.zeros(len(verts_m), dtype=bool)
    in_mask[in_bounds] = mask_bin[vi[in_bounds], ui[in_bounds]]

    # Keep face if any vertex is inside mask
    keep = in_mask[mesh.faces].any(axis=1)
    n_removed = int((~keep).sum())
    if n_removed == 0:
        return mesh

    mesh_clipped = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces[keep], process=True
    )
    print(f"      silhouette clip: removed {n_removed} faces  "
          f"kept {len(mesh_clipped.faces)}  (dilation={dilation_px}px)")
    return mesh_clipped


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
    parser.add_argument("--udf-threshold", default=0.23, type=float,
                        help="NU-MCC UDF threshold for surface extraction (default: 0.23, "
                             "matches CO3D-V2 training). Candidates below this distance are "
                             "then refined with move_points gradient descent.")
    parser.add_argument("--udf-n-iter", default=10, type=int,
                        help="move_points gradient-descent iterations to snap candidates onto the "
                             "zero-level set (default: 10, matches demo_iphone.py; more→better "
                             "surface alignment)")
    parser.add_argument("--n-query", default=200_000, type=int,
                        help="Query grid points for UDF evaluation (default: 200000, "
                             "n_side=cbrt(n_query) per axis). More→denser surface coverage.")
    parser.add_argument("--poisson-depth", default=10, type=int,
                        help="Poisson reconstruction octree depth (default: 10; 9=coarse, 11=fine)")
    parser.add_argument("--mesh-method", default="poisson",
                        choices=["poisson", "noksr", "both"],
                        help="Mesh reconstruction method: poisson (Open3D Poisson, default), "
                             "noksr (nksr Neural Kernel Surface Reconstruction), "
                             "both (run both and save {name}_poisson.obj + {name}_noksr.obj "
                             "for quality comparison; {name}.obj uses noksr)")
    parser.add_argument("--no-sdf", action="store_true",
                        help="Skip CoACD convex decomposition and SDF generation.")
    parser.add_argument("--no-floor-cap", action="store_true",
                        help="Disable the automatic floor cap. By default, when a mask is "
                             "provided the mesh is cut at the support plane (1st pct of "
                             "object Z) and capped with a contour-following Delaunay face.")
    parser.add_argument("--debug-dump", default=None, type=Path,
                        help="If set, write a 'serie de copias' of NU-MCC's inputs "
                             "(seen image, seen_xyz map, valid mask, points) and outputs "
                             "(candidates, surface, UDF values, completion summary) to this "
                             "directory for inspection.")
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

    # Save PLY of the SAM2-filtered object point cloud (all points, before subsampling).
    # Persisted in the Y-up frame for viewers/Drake; object_pts stays camera-frame
    # for the floor-cap percentile and query-grid bbox computed below.
    ply_path = asset_dir / f"{args.name}_object_cloud.ply"
    _save_ply(ply_path, to_y_up(object_pts), object_colors)
    print(f"      PLY saved: {ply_path}  ({len(object_pts)} pts, RGB, Y-up)")

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
        np.save(str(completed_pts_path), to_y_up(completed_pts))
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

    # Floor cap: the bin surface is at the MAX depth (camera overhead → large Z = far = bin).
    # Use 95th percentile of object Z: cuts slightly inside the mesh (not at the very
    # edge) so slice_mesh_plane finds a proper interior cross-section to cap.
    # 99th pct is right at the Poisson boundary where faces are degenerate.
    z_floor = None
    if args.mask is not None and not args.no_floor_cap:
        z_floor = float(np.percentile(object_pts[:, 2], 95))
        print(f"      floor cap: will cut at z_floor={z_floor:.4f} m  "
              f"(95th pct of {len(object_pts)} object pts = bin surface)")

    print("[4/6] NU-MCC: reconstructing surface from RGB + depth XYZ map...")
    print(f"      seen_xyz source: full depth ({depth.shape}), {int((depth>0).sum())} valid px")
    print(f"      seen_mask: {'yes (object pixels only)' if seen_mask is not None else 'no (full scene)'}")
    print(f"      query grid anchor: {len(query_pts)} object pts (camera frame)")
    surface_pts, norm_center, norm_scale = run_numcc(
        color, depth, fx, fy, cx, cy,
        weights_dir=models_root / "numcc",
        query_pts=query_pts,
        seen_mask=seen_mask,
        udf_threshold=args.udf_threshold,
        n_query=args.n_query,
        batch_size=6_000,
        n_iter=args.udf_n_iter,
        debug_dir=args.debug_dump,
    )

    # Convert floor depth to normalized space (same coord system as surface_pts)
    z_floor_norm = None
    if z_floor is not None and norm_scale > 1e-6:
        z_floor_norm = (z_floor - float(norm_center[2])) / norm_scale
    print(f"      {len(surface_pts)} surface points")

    # Save PLY of raw NU-MCC surface output (normalized space, before meshing).
    # Persisted Y-up; surface_pts stays camera-frame for the in-process meshing,
    # silhouette clip and floor cap (all of which depend on the camera convention).
    numcc_ply = asset_dir / f"{args.name}_numcc_surface.ply"
    _save_ply(numcc_ply, to_y_up(surface_pts))
    print(f"      PLY saved: {numcc_ply}  (Y-up)")

    use_noksr  = args.mesh_method in ("noksr", "both")
    use_poisson = args.mesh_method in ("poisson", "both")

    raw_mesh_poisson = None
    raw_mesh_noksr   = None

    if use_poisson:
        print("[4b/6] Meshing surface points (Poisson reconstruction)...")
        raw_mesh_poisson = _points_to_mesh(surface_pts, poisson_depth=args.poisson_depth)
        print(f"       Poisson raw mesh: {len(raw_mesh_poisson.faces)} faces")

    if use_noksr:
        print("[4b/6] Meshing surface points (nksr — Neural Kernel Surface Reconstruction)...")
        raw_mesh_noksr = _points_to_mesh_noksr(surface_pts)
        print(f"       nksr raw mesh: {len(raw_mesh_noksr.faces)} faces")

    # ── silhouette clip first: remove lateral excess before adding floor cap ───
    # Must run BEFORE floor cap so the cap vertices (which project below the mask
    # footprint) are not clipped away.
    if seen_mask is not None:
        print("[4c/6] Clipping lateral excess via SAM2 mask silhouette...")
        if raw_mesh_poisson is not None:
            raw_mesh_poisson = _clip_by_mask_silhouette(
                raw_mesh_poisson, norm_center, norm_scale, fx, fy, cx, cy, seen_mask,
                dilation_px=0)
        if raw_mesh_noksr is not None:
            raw_mesh_noksr = _clip_by_mask_silhouette(
                raw_mesh_noksr, norm_center, norm_scale, fx, fy, cx, cy, seen_mask,
                dilation_px=0)

    # ── floor cap: cut at support plane + add contour-following cap ──────────
    if z_floor_norm is not None:
        print("[4d/6] Applying floor cap (slice at support plane + contour cap)...")
        if raw_mesh_poisson is not None:
            raw_mesh_poisson = _slice_and_cap_at_floor(raw_mesh_poisson, z_floor_norm)
        if raw_mesh_noksr is not None:
            raw_mesh_noksr = _slice_and_cap_at_floor(raw_mesh_noksr, z_floor_norm)

    # ── convert finished mesh(es) to Y-up frame ──────────────────────────────
    # Everything above (NU-MCC, silhouette clip, floor cap) runs in the OpenCV
    # camera frame (Y-down, Z-forward). Rotate the completed mesh 180° about X so
    # the persisted OBJ/SDF (and the comparison exports) are Y-up like the clouds.
    print("[4e/6] Converting mesh(es) to Y-up frame (180° about X)...")
    T_yup = np.eye(4)
    T_yup[:3, :3] = CAM_TO_YUP
    if raw_mesh_poisson is not None:
        raw_mesh_poisson.apply_transform(T_yup)
    if raw_mesh_noksr is not None:
        raw_mesh_noksr.apply_transform(T_yup)

    # Primary mesh: noksr when available, else poisson
    raw_mesh = raw_mesh_noksr if raw_mesh_noksr is not None else raw_mesh_poisson

    print("[5/6] Normalizing mesh(es) (longest axis -> 20 cm)...")
    mesh = normalize_mesh(raw_mesh)
    obj_path = asset_dir / f"{args.name}.obj"
    mesh.export(str(obj_path))
    method_label = "noksr" if raw_mesh_noksr is not None else "poisson"
    print(f"      Saved: {obj_path}  ({len(mesh.faces)} faces)  [{method_label}]")

    if args.mesh_method == "both":
        # Save individual comparison files before normalization scaling is lost
        for label, rm in [("poisson", raw_mesh_poisson), ("noksr", raw_mesh_noksr)]:
            if rm is None:
                continue
            m_norm = normalize_mesh(rm)
            cmp_path = asset_dir / f"{args.name}_{label}.obj"
            m_norm.export(str(cmp_path))
            print(f"      Saved comparison: {cmp_path}  ({len(m_norm.faces)} faces)")

    if not args.no_sdf:
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
    else:
        print("[5b/6] Skipping CoACD + SDF (--no-sdf)")

    print(f"\nDone -> {asset_dir}/  ({time.time() - t_start:.1f}s total)")
    print(f"  {args.name}.obj  [{method_label}]")
    if args.mesh_method == "both":
        print(f"  {args.name}_poisson.obj")
        print(f"  {args.name}_noksr.obj")
    if not args.no_sdf:
        print(f"  {args.name}.sdf")
        print(f"  {args.name}_parts/")
    print(f"  {args.name}_numcc_surface.ply  ← raw NU-MCC output (normalized space)")
    print(f"  {args.name}_object_cloud.ply   ← depth back-projection (metric space)")
    print(f"  {args.name}_pointcloud.npy")


if __name__ == "__main__":
    main()
