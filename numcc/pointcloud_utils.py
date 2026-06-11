import json
from pathlib import Path
import numpy as np


def load_depth(path: Path) -> np.ndarray:
    """Load depth map as float32 array in meters. Accepts .npy or .png (uint16 mm)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    elif path.suffix == ".png":
        import imageio.v2 as imageio
        raw = imageio.imread(str(path))
        return raw.astype(np.float32) / 1000.0
    else:
        raise ValueError(f"Unsupported depth format: {path.suffix} (expected .npy or .png)")


def depth_to_pointcloud(depth: np.ndarray, fx: float, fy: float,
                        cx: float, cy: float) -> np.ndarray:
    """Back-project depth image to 3D point cloud using pinhole model. Returns (N, 3) float32."""
    v_idx, u_idx = np.where(depth > 0)
    Z = depth[v_idx, u_idx]
    X = (u_idx - cx) * Z / fx
    Y = (v_idx - cy) * Z / fy
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


# Coordinate convention: depth_to_pointcloud() emits points in the OpenCV camera
# frame (+X right, +Y down, +Z forward/into scene). Viewers and Drake expect a
# Y-up frame. Convert with a 180° rotation about the X axis: it flips Y (down→up)
# and Z (forward→back), fixing the XZ-plane orientation, while keeping det=+1 so
# handedness is preserved (no mirroring). This is the standard OpenCV→OpenGL
# camera-frame conversion, applied as a post-processing step on the reconstruction
# (NOT inside the depth estimator).
CAM_TO_YUP = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float32)


def to_y_up(pts: np.ndarray) -> np.ndarray:
    """Rotate (N, 3) points from the OpenCV camera frame (Y-down, Z-forward) to a
    Y-up frame: (x, y, z) → (x, -y, -z). Returns a new (N, 3) float32 array."""
    return (np.asarray(pts, dtype=np.float32) @ CAM_TO_YUP.T).astype(np.float32)


def subsample_pointcloud(pts: np.ndarray, n: int) -> np.ndarray:
    """Random subsample to n points; repeat-pad if fewer than n available."""
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
    else:
        idx = np.random.choice(len(pts), n, replace=True)
    return pts[idx]


def load_intrinsics(path: Path) -> tuple[float, float, float, float]:
    """Load camera intrinsics from JSON. Returns (fx, fy, cx, cy)."""
    data = json.loads(Path(path).read_text())
    return float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])
