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
    """Return True if mask_i area is within [lo, hi]× the best-view mask area."""
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


if __name__ == "__main__":
    print("pipeline_multiview.py — stub (stages not yet implemented)")
