import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "numcc"))


def test_load_depth_npy(tmp_path):
    from pointcloud_utils import load_depth
    depth = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    p = tmp_path / "depth.npy"
    np.save(str(p), depth)
    result = load_depth(p)
    np.testing.assert_array_equal(result, depth)
    assert result.dtype == np.float32


def test_load_depth_png_converts_mm_to_m(tmp_path):
    from pointcloud_utils import load_depth
    import imageio
    depth_mm = np.array([[1000, 2000], [0, 3000]], dtype=np.uint16)
    p = tmp_path / "depth.png"
    imageio.imwrite(str(p), depth_mm)
    result = load_depth(p)
    np.testing.assert_allclose(result, np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32), rtol=1e-5)
    assert result.dtype == np.float32


def test_load_depth_unsupported_extension_raises(tmp_path):
    from pointcloud_utils import load_depth
    p = tmp_path / "depth.xyz"
    p.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported depth format"):
        load_depth(p)


def test_depth_to_pointcloud_basic():
    from pointcloud_utils import depth_to_pointcloud
    depth = np.array([[2.0]], dtype=np.float32)
    pts = depth_to_pointcloud(depth, fx=500.0, fy=500.0, cx=0.0, cy=0.0)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 2.0], atol=1e-6)


def test_depth_to_pointcloud_filters_zero_depth():
    from pointcloud_utils import depth_to_pointcloud
    depth = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    pts = depth_to_pointcloud(depth, fx=500.0, fy=500.0, cx=0.5, cy=0.5)
    assert pts.shape[0] == 2


def test_depth_to_pointcloud_known_values():
    from pointcloud_utils import depth_to_pointcloud
    # pixel at (u=2, v=1), Z=1.0m, fx=fy=1.0, cx=cy=0 -> X=2.0, Y=1.0, Z=1.0
    depth = np.zeros((3, 4), dtype=np.float32)
    depth[1, 2] = 1.0
    pts = depth_to_pointcloud(depth, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [2.0, 1.0, 1.0], atol=1e-6)


def test_subsample_pointcloud_exact_count():
    from pointcloud_utils import subsample_pointcloud
    pts = np.random.rand(10000, 3).astype(np.float32)
    result = subsample_pointcloud(pts, n=2048)
    assert result.shape == (2048, 3)


def test_subsample_pointcloud_fewer_than_n_pads():
    from pointcloud_utils import subsample_pointcloud
    pts = np.random.rand(100, 3).astype(np.float32)
    result = subsample_pointcloud(pts, n=2048)
    assert result.shape == (2048, 3)


def test_load_intrinsics_from_json(tmp_path):
    from pointcloud_utils import load_intrinsics
    import json
    data = {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0}
    p = tmp_path / "intrinsics.json"
    p.write_text(json.dumps(data))
    fx, fy, cx, cy = load_intrinsics(p)
    assert fx == 615.0 and fy == 615.0 and cx == 320.0 and cy == 240.0
