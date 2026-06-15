#!/usr/bin/env python3
"""
Multiview pipeline: directory of N images → OBJ mesh.

Stages:
  1. DA3 multiview  → per-view depth + intrinsics + extrinsics
  2. SAM2 AMG       → per-view segmask, best view by centerness
  3. SAM2 bbox      → fallback refinement for poor masks
  4. Merge clouds   → back-project + transform to world frame + voxel downsample
  5. Remesh         → NKSR or Poisson via numcc:x86 Docker
  6. Smooth         → PyMeshLab smooth_preserve.py

Usage:
  .venv/bin/python3 tools/pipeline_scripts/pipeline_multiview.py \\
      --views-dir data/images/headphones_views/ \\
      --name headphones
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent.parent.resolve()
DA3_CLI       = Path("/home/worker-node-4/Documents/GitHub/UniWhere/.venv/bin/da3")
MODELS_NUMCC  = Path.home() / "models" / "numcc"
MODELS_SAM2   = Path.home() / "models" / "sam2"
NKSR_CACHE    = Path.home() / "models" / "nksr_cache"
DOCKER_NUMCC  = "numcc:x86"
DOCKER_SAM2   = "sam2:x86"

CAM_TO_YUP = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float32)


# ─── Utility functions ────────────────────────────────────────────────────────

def _centerness_distance(mask: np.ndarray, H: int, W: int) -> float:
    """Euclidean distance from mask centroid to image centre. Returns inf if mask empty."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return float("inf")
    cy_mask, cx_mask = float(ys.mean()), float(xs.mean())
    return math.sqrt((cx_mask - W / 2) ** 2 + (cy_mask - H / 2) ** 2)


def _area_ratio_ok(mask_i: np.ndarray, mask_best: np.ndarray,
                   lo: float = 0.25, hi: float = 4.0) -> bool:
    """Return True if mask_i area is within [lo, hi]x the best-view mask area."""
    best_area = int(mask_best.sum())
    if best_area == 0:
        return False
    ratio = int(mask_i.sum()) / best_area
    return lo <= ratio <= hi


def _transform_to_world(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply camera-to-world transform. pts_cam: (N,3), R: (3,3), t: (3,) → (N,3)."""
    return (R @ pts_cam.T).T + t


def _project_to_bbox(
    pts_world: np.ndarray,
    R_i: np.ndarray, t_i: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    margin: int = 20,
) -> list[int]:
    """Project world points into camera i and return 2D bounding box [x1,y1,x2,y2]."""
    R_inv = R_i.T
    pts_cam = (R_inv @ (pts_world - t_i).T).T    # world → cam_i
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return [0, 0, W, H]
    u = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(int)
    v = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy).astype(int)
    x1 = max(int(u.min()) - margin, 0)
    y1 = max(int(v.min()) - margin, 0)
    x2 = min(int(u.max()) + margin, W)
    y2 = min(int(v.max()) + margin, H)
    return [x1, y1, x2, y2]


def _voxel_downsample(
    pts: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-hash voxel downsample. Keeps first point per voxel. pts: (N,3), colors: (N,3)."""
    keys = (pts / voxel_size).astype(np.int32)
    seen: dict[tuple, int] = {}
    for i, k in enumerate(map(tuple, keys)):
        if k not in seen:
            seen[k] = i
    idx = list(seen.values())
    return pts[idx], colors[idx]


def _save_ply(path: Path, pts: np.ndarray, colors: np.ndarray | None = None):
    """Write binary little-endian PLY. pts: (N,3) float32, colors: (N,3) uint8 optional."""
    N = len(pts)
    has_color = colors is not None and len(colors) == N
    dt: list = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if has_color:
        dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.zeros(N, dtype=dt)
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if has_color:
        data["red"]   = colors[:, 0]
        data["green"] = colors[:, 1]
        data["blue"]  = colors[:, 2]
    prop = "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        prop += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {N}\n{prop}end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# ─── VRAM Monitor ─────────────────────────────────────────────────────────────

class VRAMMonitor:
    def __init__(self, poll_ms: int = 200):
        self._poll_ms = poll_ms
        self._data: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while self._running:
            mb = self._query()
            if mb is not None:
                with self._lock:
                    self._data.append((time.time(), mb))
            time.sleep(self._poll_ms / 1000.0)

    def _query(self) -> int | None:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                return int(r.stdout.strip())
        except Exception:
            pass
        return None

    def current_mb(self) -> int:
        v = self._query()
        return v if v is not None else 0

    def peak_mb(self, t0: float, t1: float) -> int:
        with self._lock:
            vals = [mb for t, mb in self._data if t0 <= t <= t1]
        return max(vals) if vals else self.current_mb()

    def save_csv(self, path: Path):
        with self._lock:
            rows = list(self._data)
        with open(path, "w") as f:
            f.write("timestamp_s,used_mb\n")
            for t, mb in rows:
                f.write(f"{t:.3f},{mb}\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


def run(cmd: list) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True)


def _save_depth_vis_2panel(depth: np.ndarray, original_image: Path, out_path: Path):
    """2-panel depth viz: original | depth colourised."""
    try:
        from PIL import Image as PILImage
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        H, W = depth.shape
        d_min, d_max = float(depth.min()), float(depth.max())
        d_norm = (depth - d_min) / max(d_max - d_min, 1e-6)
        depth_rgb = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)
        orig = PILImage.open(str(original_image)).convert("RGB").resize((W, H), PILImage.LANCZOS)
        panel = PILImage.new("RGB", (W * 2, H))
        panel.paste(orig, (0, 0))
        panel.paste(PILImage.fromarray(depth_rgb), (W, 0))
        panel.save(str(out_path))
    except Exception as e:
        print(f"      [depth_vis] skipped ({e})")


def _save_depth_vis_3panel(depth: np.ndarray, mask_bool: np.ndarray,
                            original_image: Path, out_path: Path):
    """3-panel depth viz: original | depth | depth x mask."""
    try:
        from PIL import Image as PILImage
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        H, W = depth.shape
        d_min, d_max = float(depth.min()), float(depth.max())
        d_norm = (depth - d_min) / max(d_max - d_min, 1e-6)
        depth_rgb = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)
        obj_rgb = depth_rgb.copy()
        if mask_bool.shape == (H, W):
            obj_rgb[~mask_bool] = 20
        orig = PILImage.open(str(original_image)).convert("RGB").resize((W, H), PILImage.LANCZOS)
        panel = PILImage.new("RGB", (W * 3, H))
        panel.paste(orig, (0, 0))
        panel.paste(PILImage.fromarray(depth_rgb), (W, 0))
        panel.paste(PILImage.fromarray(obj_rgb), (W * 2, 0))
        panel.save(str(out_path))
    except Exception as e:
        print(f"      [depth_vis_3] skipped ({e})")


# ─── Stage 1: DA3 multiview ───────────────────────────────────────────────────

def stage_da3(images: list[Path], out_dir: Path, monitor: VRAMMonitor) -> dict:
    """
    Run DA3 on N images simultaneously.
    Returns: {depths: list[Path], intrinsics: list[Path], extrinsics_npy: Path, t0, t1, peak_mb}
    """
    header("STAGE 1 — DA3 multiview depth estimation")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DA3_CLI.exists():
        sys.exit(f"DA3 CLI not found: {DA3_CLI}\n"
                 "Activate UniWhere venv: source /home/worker-node-4/Documents/GitHub/UniWhere/.venv/bin/activate")

    # DA3 'images' subcommand takes a directory; all images must be in the same dir
    images_dir = images[0].parent
    t0 = time.time()
    run([str(DA3_CLI), "images", str(images_dir),
         "--export-format", "npz",
         "--export-dir", str(out_dir),
         "--auto-cleanup"])
    t1 = time.time()

    npz_files = sorted(out_dir.rglob("*.npz"))
    if not npz_files:
        sys.exit(f"DA3 produced no .npz in {out_dir}")
    npz_path = npz_files[0]

    data = np.load(str(npz_path), allow_pickle=True)
    print(f"  NPZ keys: {list(data.keys())}")

    # depth: (N, H, W) or (H, W) for single image
    raw_depth = data["depth"].astype(np.float32)
    if raw_depth.ndim == 2:
        raw_depth = raw_depth[None]   # (1, H, W)
    N = raw_depth.shape[0]
    print(f"  Views: {N}  depth shape: {raw_depth.shape}")

    # intrinsics: (N, 3, 3)
    raw_intri = data["intrinsics"] if "intrinsics" in data else None
    if raw_intri is not None and raw_intri.ndim == 2:
        raw_intri = raw_intri[None]

    # extrinsics: (N, 3, 4) — camera-to-world [R|t]
    raw_extri = data["extrinsics"] if "extrinsics" in data else None
    if raw_extri is not None and raw_extri.ndim == 2:
        raw_extri = raw_extri[None]

    depth_paths, intri_paths = [], []
    for i in range(N):
        depth_i = np.clip(raw_depth[i], 0.05, 20.0)
        d_path = out_dir / f"depth_{i}.npy"
        np.save(str(d_path), depth_i)
        depth_paths.append(d_path)
        print(f"  depth_{i}: shape={depth_i.shape}  [{depth_i.min():.3f},{depth_i.max():.3f}] m")

        H_i, W_i = depth_i.shape
        if raw_intri is not None and i < len(raw_intri):
            K = raw_intri[i]
            intri = {"fx": float(K[0,0]), "fy": float(K[1,1]),
                     "cx": float(K[0,2]), "cy": float(K[1,2])}
        else:
            fx = W_i / (2 * np.tan(np.radians(60) / 2))
            intri = {"fx": fx, "fy": fx, "cx": float(W_i/2), "cy": float(H_i/2)}
            print(f"    fallback intrinsics (60 deg HFOV)")
        k_path = out_dir / f"intrinsics_{i}.json"
        k_path.write_text(json.dumps(intri, indent=2))
        intri_paths.append(k_path)
        print(f"  intrinsics_{i}: fx={intri['fx']:.1f} fy={intri['fy']:.1f} "
              f"cx={intri['cx']:.1f} cy={intri['cy']:.1f}")

        _save_depth_vis_2panel(depth_i, images[i], out_dir / f"depth_vis_{i}.png")

    # Save extrinsics as (N, 3, 4)
    if raw_extri is not None:
        extri_arr = raw_extri[:N].astype(np.float32)
    else:
        # identity for all views (single-image fallback)
        extri_arr = np.zeros((N, 3, 4), dtype=np.float32)
        extri_arr[:, :3, :3] = np.eye(3)
    extri_path = out_dir / "extrinsics.npy"
    np.save(str(extri_path), extri_arr)
    print(f"  extrinsics: shape={extri_arr.shape}")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")

    return {
        "depths": depth_paths,
        "intrinsics": intri_paths,
        "extrinsics_npy": extri_path,
        "extrinsics": extri_arr,
        "t0": t0, "t1": t1, "peak_mb": peak,
    }


def stage_sam2_all(
    images: list[Path],
    name: str,
    out_dir: Path,
    monitor: VRAMMonitor,
    depths: list[Path],
    sam2_model: str = "small",
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> dict:
    """
    Run SAM2 AMG on all N views. Returns best_idx (most centred mask).
    """
    header("STAGE 2 — SAM2 AMG (all views) + best view selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    segmask_paths: list[Path] = []
    centerness_scores: list[float] = []

    for i, img in enumerate(images):
        print(f"\n  --- View {i}: {img.name} ---")
        view_out = out_dir / f"view_{i}"
        view_out.mkdir(exist_ok=True)

        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{img.resolve().parent}:/input:ro",
            "-v", f"{view_out.resolve()}:/output",
            "-v", f"{MODELS_SAM2}:/opt/sam2/checkpoints:ro",
            "-v", f"{PROJECT_ROOT / 'submodules' / 'sam2' / 'pipeline.py'}:/opt/sam2/pipeline.py:ro",
            "--entrypoint", "python3",
            DOCKER_SAM2,
            "/opt/sam2/pipeline.py",
            "--input",  f"/input/{img.name}",
            "--output", "/output",
            "--name",   f"{name}_{i}",
            "--model",  sam2_model,
            "--pred-iou-thresh",        str(pred_iou_thresh),
            "--stability-score-thresh", str(stability_score_thresh),
        ]
        t_i0 = time.time()
        run(cmd)
        t_i1 = time.time()
        print(f"    SAM2 view {i}: {t_i1-t_i0:.1f}s")

        # Move outputs to 02_sam2/ flat structure
        segmask_src = view_out / f"{name}_{i}_segmask.npy"
        viz_src     = view_out / f"{name}_{i}_viz.png"
        segmask_dst = out_dir / f"{name}_segmask_{i}.npy"
        viz_dst     = out_dir / f"{name}_viz_{i}.png"
        if not segmask_src.exists():
            print(f"  WARNING: SAM2 view {i} produced no segmask — skipping view")
            segmask_paths.append(None)
            centerness_scores.append(float("inf"))
            continue
        shutil.move(str(segmask_src), str(segmask_dst))
        if viz_src.exists():
            shutil.move(str(viz_src), str(viz_dst))
        shutil.rmtree(str(view_out), ignore_errors=True)
        segmask_paths.append(segmask_dst)

        # Centerness score
        mask = np.load(str(segmask_dst)).astype(np.uint8)
        from PIL import Image as PILImage
        img_np = np.array(PILImage.open(str(img)).convert('RGB'))
        H, W = img_np.shape[:2]
        score = _centerness_distance(mask, H, W)
        centerness_scores.append(score)
        print(f"    centerness_distance={score:.1f}  mask_sum={mask.sum()}")

    # best view = smallest centerness distance (closest to centre)
    valid_scores = [(s, i) for i, s in enumerate(centerness_scores)
                    if s < float("inf") and segmask_paths[i] is not None]
    if not valid_scores:
        sys.exit("SAM2 failed on all views — no valid segmask produced")
    best_idx = min(valid_scores, key=lambda x: x[0])[1]
    print(f"\n  Best view: {best_idx}  (centerness={centerness_scores[best_idx]:.1f})")

    (out_dir / "best_view_idx.txt").write_text(str(best_idx))

    # Generate depth_vis_masked for each view (3-panel) using saved masks
    for i, img in enumerate(images):
        if segmask_paths[i] is None:
            continue
        depth = np.load(str(depths[i]))
        mask = np.load(str(segmask_paths[i])).astype(bool)
        # Resize mask to depth resolution if needed
        if mask.shape != depth.shape:
            from PIL import Image as PILImage
            m_img = PILImage.fromarray(mask.astype(np.uint8)*255).resize(
                (depth.shape[1], depth.shape[0]), PILImage.NEAREST)
            mask = np.asarray(m_img) > 0
        _save_depth_vis_3panel(depth, mask, img, out_dir / f"depth_vis_masked_{i}.png")

    peak = monitor.peak_mb(0, time.time())  # cumulative
    return {
        "segmask_paths": segmask_paths,
        "centerness_scores": centerness_scores,
        "best_idx": best_idx,
        "t0": 0, "t1": 0, "peak_mb": peak,  # individual timing tracked per-view above
    }


def stage_sam2_bbox_fallback(
    images: list[Path],
    name: str,
    sam2_dir: Path,
    segmask_paths: list[Path | None],
    best_idx: int,
    depths: list[Path],
    intrinsics: list[Path],
    extrinsics: np.ndarray,
    monitor: VRAMMonitor,
    sam2_model: str = "small",
) -> list[Path | None]:
    """
    For views where AMG produced a poor mask (area_ratio outside [0.25, 4.0] vs best view),
    re-run SAM2 with a bbox prompt derived from the best-view 3D cloud.
    Returns updated segmask_paths list.
    """
    header("STAGE 3 — SAM2 bbox fallback for poor masks")

    best_mask_path = segmask_paths[best_idx]
    if best_mask_path is None:
        print("  Best view has no mask — skipping bbox fallback")
        return segmask_paths

    mask_best = np.load(str(best_mask_path)).astype(np.uint8)
    depth_best = np.load(str(depths[best_idx])).astype(np.float32)
    intri_best = json.loads(Path(intrinsics[best_idx]).read_text())
    fx_b, fy_b = intri_best["fx"], intri_best["fy"]
    cx_b, cy_b = intri_best["cx"], intri_best["cy"]
    R_best = extrinsics[best_idx, :, :3]
    t_best = extrinsics[best_idx, :, 3]

    # Back-project best view depth filtered by mask → world frame
    H_b, W_b = depth_best.shape
    m_best_resized = mask_best
    if mask_best.shape != (H_b, W_b):
        from PIL import Image as PILImage
        m_img = PILImage.fromarray(mask_best*255).resize((W_b, H_b), PILImage.NEAREST)
        m_best_resized = (np.asarray(m_img) > 0).astype(np.uint8)

    # Simple inline back-projection
    v_idx, u_idx = np.where((depth_best > 0) & (m_best_resized > 0))
    Z = depth_best[v_idx, u_idx]
    X = (u_idx - cx_b) * Z / fx_b
    Y = (v_idx - cy_b) * Z / fy_b
    pts_cam_best = np.stack([X, Y, Z], axis=1).astype(np.float32)
    pts_world = _transform_to_world(pts_cam_best, R_best, t_best)

    updated = list(segmask_paths)
    any_fallback = False

    for i, img in enumerate(images):
        if i == best_idx or segmask_paths[i] is None:
            continue
        mask_i = np.load(str(segmask_paths[i])).astype(np.uint8)
        if _area_ratio_ok(mask_i, mask_best):
            print(f"  View {i}: mask OK (area_ratio in range)")
            continue

        print(f"  View {i}: mask poor → running SAM2 bbox prompt")
        any_fallback = True

        intri_i = json.loads(Path(intrinsics[i]).read_text())
        fx_i, fy_i = intri_i["fx"], intri_i["fy"]
        cx_i, cy_i = intri_i["cx"], intri_i["cy"]
        R_i = extrinsics[i, :, :3]
        t_i = extrinsics[i, :, 3]
        depth_i = np.load(str(depths[i]))
        H_i, W_i = depth_i.shape

        bbox = _project_to_bbox(pts_world, R_i, t_i, fx_i, fy_i, cx_i, cy_i, H_i, W_i)
        print(f"    projected bbox: {bbox}")

        view_out = sam2_dir / f"view_{i}_bbox"
        view_out.mkdir(exist_ok=True)
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{img.resolve().parent}:/input:ro",
            "-v", f"{view_out.resolve()}:/output",
            "-v", f"{MODELS_SAM2}:/opt/sam2/checkpoints:ro",
            "-v", f"{PROJECT_ROOT / 'submodules' / 'sam2' / 'pipeline.py'}:/opt/sam2/pipeline.py:ro",
            "--entrypoint", "python3",
            DOCKER_SAM2,
            "/opt/sam2/pipeline.py",
            "--input",  f"/input/{img.name}",
            "--output", "/output",
            "--name",   f"{name}_{i}",
            "--model",  sam2_model,
            "--bbox",   str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
        ]
        run(cmd)

        new_mask = view_out / f"{name}_{i}_segmask.npy"
        if new_mask.exists():
            dst = sam2_dir / f"{name}_segmask_{i}.npy"
            shutil.move(str(new_mask), str(dst))
            updated[i] = dst
            m_new = np.load(str(dst))
            print(f"    bbox mask: sum={m_new.sum()}")
        shutil.rmtree(str(view_out), ignore_errors=True)

    if not any_fallback:
        print("  All masks passed quality check — no fallback needed")

    return updated


def stage_merge_clouds(
    images: list[Path],
    depths: list[Path],
    intrinsics: list[Path],
    extrinsics: np.ndarray,
    segmask_paths: list[Path | None],
    name: str,
    out_dir: Path,
    voxel_size: float,
    monitor: VRAMMonitor,
) -> dict:
    """
    Back-project each masked depth, transform to world frame, merge, voxel-downsample.
    """
    header("STAGE 4 — Merge point clouds")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for i, img in enumerate(images):
        if segmask_paths[i] is None:
            print(f"  View {i}: no mask — skipping")
            continue

        depth_i = np.load(str(depths[i])).astype(np.float32)
        H_i, W_i = depth_i.shape
        intri_i  = json.loads(Path(intrinsics[i]).read_text())
        fx, fy   = intri_i["fx"], intri_i["fy"]
        cx, cy   = intri_i["cx"], intri_i["cy"]
        R_i = extrinsics[i, :, :3].astype(np.float32)
        t_i = extrinsics[i, :, 3].astype(np.float32)

        mask_i = np.load(str(segmask_paths[i])).astype(bool)
        if mask_i.shape != (H_i, W_i):
            from PIL import Image as PILImage
            m_img = PILImage.fromarray(mask_i.astype(np.uint8)*255).resize(
                (W_i, H_i), PILImage.NEAREST)
            mask_i = np.asarray(m_img) > 0

        # Back-project masked pixels
        v_idx, u_idx = np.where((depth_i > 0) & mask_i)
        if len(v_idx) == 0:
            print(f"  View {i}: 0 object pts after mask filter — skipping")
            continue
        Z = depth_i[v_idx, u_idx]
        X = (u_idx - cx) * Z / fx
        Y = (v_idx - cy) * Z / fy
        pts_cam = np.stack([X, Y, Z], axis=1).astype(np.float32)

        # Transform cam → world frame
        pts_world = _transform_to_world(pts_cam, R_i, t_i)

        # Sample colors from original image at object pixel locations
        from PIL import Image as PILImage
        img_np = np.array(PILImage.open(str(img)).convert("RGB"))
        cH, cW = img_np.shape[:2]
        cv = (v_idx * (cH / H_i)).astype(int).clip(0, cH - 1)
        cu = (u_idx * (cW / W_i)).astype(int).clip(0, cW - 1)
        colors_i = img_np[cv, cu, :3].astype(np.uint8)

        # Save per-view PLY (Y-up)
        ply_i = out_dir / f"{name}_cloud_{i}.ply"
        _save_ply(ply_i, (CAM_TO_YUP @ pts_world.T).T, colors_i)
        print(f"  View {i}: {len(pts_world):,} pts  → {ply_i.name}")

        all_pts.append(pts_world)
        all_colors.append(colors_i)

    if not all_pts:
        sys.exit("No object points from any view — check masks and depth alignment")

    merged_pts    = np.concatenate(all_pts, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    print(f"\n  Before downsample: {len(merged_pts):,} pts")

    # Voxel downsample (numpy fallback — Open3D not available in venv)
    t0 = time.time()
    down_pts, down_colors = _voxel_downsample(merged_pts, merged_colors, voxel_size)
    print(f"  After downsample ({voxel_size}m voxels): {len(down_pts):,} pts  "
          f"({time.time()-t0:.2f}s)")

    if len(down_pts) < 100:
        sys.exit(f"Merged cloud has only {len(down_pts)} pts — insufficient coverage")

    # Apply Y-up rotation for viewer/Drake convention
    down_pts_yup = (CAM_TO_YUP @ down_pts.T).T

    merged_ply = out_dir / f"{name}_merged_cloud.ply"
    _save_ply(merged_ply, down_pts_yup, down_colors)
    print(f"  Saved: {merged_ply}")

    t1 = time.time()
    peak = monitor.peak_mb(t0, t1)
    return {
        "merged_ply": merged_ply,
        "n_pts_before": len(merged_pts),
        "n_pts_after":  len(down_pts),
        "t0": t0, "t1": t1, "peak_mb": peak,
    }


def stage_remesh(
    merged_ply: Path,
    name: str,
    mesh_dir: Path,
    mesh_method: str,
    monitor: VRAMMonitor,
) -> dict:
    """Run numcc:x86 remesh.py on merged_cloud.ply → OBJ."""
    header("STAGE 5 — Remesh  (numcc:x86 remesh.py)")
    mesh_dir.mkdir(parents=True, exist_ok=True)
    NKSR_CACHE.mkdir(parents=True, exist_ok=True)

    input_dir = merged_ply.parent.resolve()
    tmp_root = Path(tempfile.mkdtemp(prefix=f"numcc_{name}_", dir=mesh_dir.parent))

    numcc_remesh = PROJECT_ROOT / "models" / "numcc" / "remesh.py"
    if not numcc_remesh.exists():
        sys.exit(f"models/numcc/remesh.py not found: {numcc_remesh}")

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "TORCH_HOME=/nksr_cache",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{tmp_root.resolve()}:/output",
        "-v", f"{NKSR_CACHE}:/nksr_cache",
        "-v", f"{numcc_remesh}:/app/remesh.py:ro",
        "--entrypoint", "python3",
        DOCKER_NUMCC,
        "/app/remesh.py",
        "--cloud",       f"/input/{merged_ply.name}",
        "--name",        name,
        "--output",      "/output",
        "--mesh-method", mesh_method,
    ]

    t0 = time.time()
    run(cmd)
    t1 = time.time()

    asset_tmp = tmp_root / name
    if not asset_tmp.exists():
        raise RuntimeError(f"remesh.py produced no output in {asset_tmp}")

    obj_path = None
    for src in sorted(asset_tmp.iterdir()):
        dst = mesh_dir / src.name
        shutil.move(str(src), str(dst))
        if dst.suffix == ".obj" and dst.stem == name:
            obj_path = dst
        print(f"  -> 04_mesh/{src.name}")
    shutil.rmtree(str(tmp_root), ignore_errors=True)

    if obj_path is None or not obj_path.exists():
        raise RuntimeError(f"OBJ not found after remesh: {mesh_dir}/{name}.obj")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  OBJ: {obj_path}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")
    return {"obj_path": obj_path, "t0": t0, "t1": t1, "peak_mb": peak}


def stage_smooth(obj_path: Path, mesh_dir: Path, name: str, monitor: VRAMMonitor) -> dict:
    """Run smooth_preserve.py → <name>_smoothed.obj."""
    header("STAGE 6 — Smooth  (Clustering -> Taubin -> Two Steps -> Normal Smooth)")
    smooth_script = PROJECT_ROOT / "tools" / "smooth_preserve.py"
    out_obj = mesh_dir / f"{name}_smoothed.obj"
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        sys.exit(f"venv python not found: {venv_python}")
    cmd = [str(venv_python), str(smooth_script), str(obj_path), str(out_obj)]
    t0 = time.time()
    run(cmd)
    t1 = time.time()
    if not out_obj.exists():
        raise RuntimeError(f"smooth_preserve.py did not produce {out_obj}")
    peak = monitor.peak_mb(t0, t1)
    print(f"\n  Smoothed: {out_obj}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")
    return {"smoothed_obj": out_obj, "t0": t0, "t1": t1, "peak_mb": peak}


def main():
    ap = argparse.ArgumentParser(
        description="Multiview pipeline: image directory → OBJ mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--views-dir",   required=True, type=Path,
                    help="Directory containing N images (*.jpg *.png), sorted alphabetically")
    ap.add_argument("--name",        required=True,
                    help="Asset name (determines output subdirectory)")
    ap.add_argument("--output-dir",  type=Path, default=Path("output"))
    ap.add_argument("--mesh-method", default="noksr", choices=["noksr", "poisson"])
    ap.add_argument("--voxel-size",  type=float, default=0.005)
    ap.add_argument("--sam2-model",  default="small", choices=["tiny", "small", "base_plus"])
    ap.add_argument("--sam2-pred-iou-thresh",   type=float, default=0.88)
    ap.add_argument("--sam2-stability-thresh",  type=float, default=0.95)
    ap.add_argument("--skip-da3",    action="store_true")
    ap.add_argument("--skip-sam2",   action="store_true")
    ap.add_argument("--skip-merge",  action="store_true")
    ap.add_argument("--skip-smooth", action="store_true")
    ap.add_argument("--debug",       action="store_true")
    args = ap.parse_args()

    # Collect and validate images
    views_dir = args.views_dir.resolve()
    if not views_dir.is_dir():
        sys.exit(f"--views-dir not found: {views_dir}")
    images = sorted(views_dir.glob("*.jpg")) + sorted(views_dir.glob("*.JPG")) + \
             sorted(views_dir.glob("*.png")) + sorted(views_dir.glob("*.PNG"))
    images = sorted(set(images))
    if len(images) < 2:
        sys.exit(f"Need at least 2 images in {views_dir}, found {len(images)}")
    print(f"\n  Views ({len(images)}): {[img.name for img in images]}")

    # Output dirs
    out_root   = args.output_dir.resolve() / args.name
    dir_da3    = out_root / "01_da3"
    dir_sam2   = out_root / "02_sam2"
    dir_clouds = out_root / "03_clouds"
    dir_mesh   = out_root / "04_mesh"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_root}")

    monitor = VRAMMonitor()
    monitor.start()
    baseline_mb = monitor.current_mb()
    print(f"  Baseline VRAM: {baseline_mb} MB")

    stages: dict[str, dict] = {}

    try:
        # [1] DA3
        if args.skip_da3:
            depth_paths = sorted(dir_da3.glob("depth_*.npy"))
            if not depth_paths:
                sys.exit(f"--skip-da3: no depth_*.npy in {dir_da3}")
            extri_npy = dir_da3 / "extrinsics.npy"
            if not extri_npy.exists():
                sys.exit(f"--skip-da3: extrinsics.npy not found in {dir_da3}")
            extrinsics = np.load(str(extri_npy))
            intri_paths = sorted(dir_da3.glob("intrinsics_*.json"))
            da3_result = {"depths": depth_paths, "intrinsics": intri_paths,
                          "extrinsics_npy": extri_npy, "extrinsics": extrinsics,
                          "t0": 0, "t1": 0, "peak_mb": 0}
            print(f"\n[--skip-da3] Using {len(depth_paths)} depths from {dir_da3}")
        else:
            da3_result = stage_da3(images, dir_da3, monitor)
            stages["DA3"] = da3_result

        # [2] SAM2 AMG
        if args.skip_sam2:
            segmask_paths = sorted(dir_sam2.glob(f"{args.name}_segmask_*.npy"))
            if not segmask_paths:
                sys.exit(f"--skip-sam2: no segmasks in {dir_sam2}")
            best_view_file = dir_sam2 / "best_view_idx.txt"
            best_idx = int(best_view_file.read_text()) if best_view_file.exists() else 0
            centerness_scores = [float("inf")] * len(images)
            sam2_result = {"segmask_paths": segmask_paths, "best_idx": best_idx,
                           "centerness_scores": centerness_scores,
                           "t0": 0, "t1": 0, "peak_mb": 0}
            print(f"\n[--skip-sam2] Using {len(segmask_paths)} masks, best_idx={best_idx}")
        else:
            t0_sam2 = time.time()
            sam2_result = stage_sam2_all(
                images, args.name, dir_sam2, monitor,
                depths=da3_result["depths"],
                sam2_model=args.sam2_model,
                pred_iou_thresh=args.sam2_pred_iou_thresh,
                stability_score_thresh=args.sam2_stability_thresh,
            )
            sam2_result["t0"] = t0_sam2
            sam2_result["t1"] = time.time()
            stages["SAM2"] = sam2_result

        # [3] bbox fallback (runs if neither --skip-sam2 stage lock nor --skip-merge requested)
        if not args.skip_merge:
            segmask_paths = stage_sam2_bbox_fallback(
                images, args.name, dir_sam2,
                sam2_result["segmask_paths"],
                sam2_result["best_idx"],
                da3_result["depths"],
                da3_result["intrinsics"],
                da3_result["extrinsics"],
                monitor,
                sam2_model=args.sam2_model,
            )
        else:
            segmask_paths = sam2_result["segmask_paths"]
        best_idx = sam2_result["best_idx"]

        # [4] Merge
        if args.skip_merge:
            merged_ply = dir_clouds / f"{args.name}_merged_cloud.ply"
            if not merged_ply.exists():
                sys.exit(f"--skip-merge: {merged_ply} not found")
            print(f"\n[--skip-merge] Using {merged_ply}")
            merge_result = {"merged_ply": merged_ply, "t0": 0, "t1": 0, "peak_mb": 0}
        else:
            merge_result = stage_merge_clouds(
                images, da3_result["depths"], da3_result["intrinsics"],
                da3_result["extrinsics"], segmask_paths,
                args.name, dir_clouds, args.voxel_size, monitor,
            )
            stages["merge"] = merge_result
        merged_ply = merge_result["merged_ply"]

        # [5] Remesh
        result = stage_remesh(merged_ply, args.name, dir_mesh,
                              args.mesh_method, monitor)
        stages["remesh"] = result
        obj_path = result["obj_path"]

        # [6] Smooth
        if not args.skip_smooth:
            result = stage_smooth(obj_path, dir_mesh, args.name, monitor)
            stages["smooth"] = result

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed (exit {e.returncode})", file=sys.stderr)
        monitor.stop()
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        monitor.stop()
        sys.exit(1)
    finally:
        monitor.stop()

    # Report
    real = {k: v for k, v in stages.items() if v.get("t0", 0) > 0 or v.get("t1", 0) > 0}
    total_s = None
    if real:
        t_all_start = min(v["t0"] for v in real.values())
        t_all_end   = max(v["t1"] for v in real.values())
        total_s = round(t_all_end - t_all_start, 2)

    report = {
        "name": args.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_views": len(images),
        "views": [str(img) for img in images],
        "params": {
            "mesh_method":  args.mesh_method,
            "voxel_size":   args.voxel_size,
            "sam2_model":   args.sam2_model,
        },
        "stages": {k: {"duration_s": round(v["t1"] - v["t0"], 2),
                       "peak_vram_mb": v.get("peak_mb", 0)}
                   for k, v in stages.items()},
        "total_duration_s": total_s,
        "output_root": str(out_root),
    }
    (out_root / "pipeline_report.json").write_text(json.dumps(report, indent=2))
    monitor.save_csv(out_root / "vram_profile.csv")

    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE")
    print("=" * 62)
    for sname, info in stages.items():
        print(f"  {sname:<14}  {info['t1']-info['t0']:>6.1f}s  "
              f"VRAM peak {info.get('peak_mb', 0):>5} MB")
    if total_s:
        print(f"\n  Total: {total_s}s")
    print(f"  Output: {out_root}")
    print("=" * 62)


if __name__ == "__main__":
    main()
