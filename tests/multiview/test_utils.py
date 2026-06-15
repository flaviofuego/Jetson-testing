import sys
from pathlib import Path
import numpy as np
import pytest

# Add pipeline_scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "pipeline_scripts"))


# ── _centerness_distance ──────────────────────────────────────────────────────

def test_centerness_distance_centered_mask():
    from pipeline_multiview import _centerness_distance
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 1  # centroid near (50, 50) = image center
    dist = _centerness_distance(mask, H=100, W=100)
    assert dist < 5.0  # centroid within 5px of center

def test_centerness_distance_corner_mask():
    from pipeline_multiview import _centerness_distance
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # centroid ~(5, 5), far from (50, 50)
    dist = _centerness_distance(mask, H=100, W=100)
    assert dist > 50.0

def test_centerness_distance_empty_mask_returns_inf():
    from pipeline_multiview import _centerness_distance
    mask = np.zeros((100, 100), dtype=np.uint8)
    dist = _centerness_distance(mask, H=100, W=100)
    assert dist == float("inf")


# ── _area_ratio_ok ────────────────────────────────────────────────────────────

def test_area_ratio_ok_similar_masks():
    from pipeline_multiview import _area_ratio_ok
    best = np.zeros((100, 100), dtype=np.uint8); best[20:80, 20:80] = 1  # 3600 px
    other = np.zeros((100, 100), dtype=np.uint8); other[25:75, 25:75] = 1  # 2500 px
    assert _area_ratio_ok(other, best) is True

def test_area_ratio_ok_too_small():
    from pipeline_multiview import _area_ratio_ok
    best = np.zeros((100, 100), dtype=np.uint8); best[10:90, 10:90] = 1  # 6400 px
    tiny = np.zeros((100, 100), dtype=np.uint8); tiny[0:5, 0:5] = 1      # 25 px → ratio 0.004
    assert _area_ratio_ok(tiny, best) is False

def test_area_ratio_ok_too_large():
    from pipeline_multiview import _area_ratio_ok
    best = np.zeros((100, 100), dtype=np.uint8); best[40:60, 40:60] = 1  # 400 px
    big = np.zeros((100, 100), dtype=np.uint8); big[:, :] = 1            # 10000 px → ratio 25
    assert _area_ratio_ok(big, best) is False

def test_area_ratio_ok_zero_best_returns_false():
    from pipeline_multiview import _area_ratio_ok
    best = np.zeros((100, 100), dtype=np.uint8)
    other = np.zeros((100, 100), dtype=np.uint8); other[20:80, 20:80] = 1
    assert _area_ratio_ok(other, best) is False


# ── _transform_to_world ───────────────────────────────────────────────────────

def test_transform_to_world_identity():
    from pipeline_multiview import _transform_to_world
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    result = _transform_to_world(pts, R, t)
    np.testing.assert_allclose(result, pts, atol=1e-6)

def test_transform_to_world_translation():
    from pipeline_multiview import _transform_to_world
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _transform_to_world(pts, R, t)
    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]], atol=1e-6)

def test_transform_to_world_rotation_90_deg_y():
    from pipeline_multiview import _transform_to_world
    # 90° rotation around Y: X→-Z, Z→X
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    result = _transform_to_world(pts, R, t)
    np.testing.assert_allclose(result, [[0.0, 0.0, -1.0]], atol=1e-6)


# ── _project_to_bbox ─────────────────────────────────────────────────────────

def test_project_to_bbox_identity_cam():
    from pipeline_multiview import _project_to_bbox
    # Points at known locations, identity extrinsics (world = cam)
    pts_world = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)  # on Z axis at Z=1
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    fx, fy, cx, cy = 100.0, 100.0, 50.0, 50.0
    H, W = 100, 100
    bbox = _project_to_bbox(pts_world, R, t, fx, fy, cx, cy, H, W, margin=0)
    # u = 0*100/1 + 50 = 50, v = 0*100/1 + 50 = 50 → bbox [50,50,50,50]
    assert len(bbox) == 4
    assert bbox[0] <= bbox[2] and bbox[1] <= bbox[3]

def test_project_to_bbox_respects_margin():
    from pipeline_multiview import _project_to_bbox
    pts_world = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    bbox_no_margin = _project_to_bbox(pts_world, R, t, 100, 100, 50, 50, 100, 100, margin=0)
    bbox_with_margin = _project_to_bbox(pts_world, R, t, 100, 100, 50, 50, 100, 100, margin=10)
    assert bbox_with_margin[0] <= bbox_no_margin[0]
    assert bbox_with_margin[1] <= bbox_no_margin[1]
    assert bbox_with_margin[2] >= bbox_no_margin[2]
    assert bbox_with_margin[3] >= bbox_no_margin[3]

def test_project_to_bbox_clips_to_image_bounds():
    from pipeline_multiview import _project_to_bbox
    pts_world = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    bbox = _project_to_bbox(pts_world, R, t, 100, 100, 50, 50, 100, 100, margin=100)
    assert bbox[0] >= 0 and bbox[1] >= 0
    assert bbox[2] <= 100 and bbox[3] <= 100


# ── _voxel_downsample ─────────────────────────────────────────────────────────

def test_voxel_downsample_reduces_duplicate_points():
    from pipeline_multiview import _voxel_downsample
    # 4 points all in the same voxel (within 0.01m)
    pts = np.array([[0.001, 0.001, 0.001],
                    [0.002, 0.001, 0.001],
                    [0.001, 0.002, 0.001],
                    [0.001, 0.001, 0.002]], dtype=np.float32)
    colors = np.ones((4, 3), dtype=np.uint8) * 128
    pts_down, colors_down = _voxel_downsample(pts, colors, voxel_size=0.01)
    assert len(pts_down) == 1
    assert len(colors_down) == 1

def test_voxel_downsample_keeps_separate_points():
    from pipeline_multiview import _voxel_downsample
    pts = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    pts_down, colors_down = _voxel_downsample(pts, colors, voxel_size=0.01)
    assert len(pts_down) == 3

def test_voxel_downsample_preserves_colors():
    from pipeline_multiview import _voxel_downsample
    pts = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    pts_down, colors_down = _voxel_downsample(pts, colors, voxel_size=0.01)
    assert colors_down.shape == (2, 3)
    assert colors_down.dtype == np.uint8


# ── _save_ply ─────────────────────────────────────────────────────────────────

def test_save_ply_creates_file(tmp_path):
    from pipeline_multiview import _save_ply
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    out = tmp_path / "test.ply"
    _save_ply(out, pts, colors)
    assert out.exists()
    content = out.read_bytes()
    assert b"ply" in content
    assert b"element vertex 2" in content
    assert b"property uchar red" in content

def test_save_ply_no_colors(tmp_path):
    from pipeline_multiview import _save_ply
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = tmp_path / "test_nocolor.ply"
    _save_ply(out, pts)
    assert out.exists()
    content = out.read_bytes()
    assert b"property uchar red" not in content
