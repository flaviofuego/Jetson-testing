"""Unit test for the camera-frame → Y-up coordinate transform.

The numcc pipeline back-projects depth in the OpenCV camera convention
(+X right, +Y down, +Z forward). Reconstructions persisted for viewers / Drake
must be Y-up. `to_y_up` performs a 180° rotation about X — flips Y and Z,
preserves X and right-handedness (no mirroring).

Run:  python3 numcc/test_y_up.py
"""
import numpy as np

from pointcloud_utils import to_y_up, CAM_TO_YUP


def test_flips_y_and_z_keeps_x():
    pts = np.array([[1.0, 2.0, 3.0],
                    [-0.5, 0.25, -4.0]], dtype=np.float32)
    out = to_y_up(pts)
    expected = np.array([[1.0, -2.0, -3.0],
                         [-0.5, -0.25, 4.0]], dtype=np.float32)
    assert np.allclose(out, expected), f"got {out}"


def test_preserves_handedness():
    # det must be +1 (proper rotation, no mirror) — chirality preserved.
    assert np.isclose(np.linalg.det(CAM_TO_YUP), 1.0), np.linalg.det(CAM_TO_YUP)


def test_y_down_object_becomes_y_up():
    # Cup opening at top of image → smaller (more negative) camera-Y.
    # After transform the opening must point to +Y (up).
    opening_cam = np.array([[0.0, -0.05, 0.30]], dtype=np.float32)  # top of cup
    base_cam    = np.array([[0.0,  0.05, 0.30]], dtype=np.float32)  # bottom of cup
    opening_up = to_y_up(opening_cam)[0]
    base_up    = to_y_up(base_cam)[0]
    assert opening_up[1] > base_up[1], (
        f"opening Y={opening_up[1]} should be above base Y={base_up[1]}"
    )


def test_returns_float32_same_shape():
    pts = np.random.randn(100, 3).astype(np.float64)
    out = to_y_up(pts)
    assert out.shape == (100, 3)
    assert out.dtype == np.float32


def test_involution():
    # Applying the rotation twice returns the original (180° + 180° = 360°).
    pts = np.random.randn(50, 3).astype(np.float32)
    assert np.allclose(to_y_up(to_y_up(pts)), pts)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("All tests passed.")
