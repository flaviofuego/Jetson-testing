import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "pipeline_scripts"))


# ── _fpfh_register ────────────────────────────────────────────────────────────

def test_fpfh_register_known_rotation():
    """FPFH should recover a known 90° Z-axis rotation on a synthetic L-shaped cloud."""
    from pipeline_multiview import _fpfh_register
    rng = np.random.default_rng(42)
    tgt = np.vstack([
        rng.uniform([0.0, 0.0, 0.0], [0.2, 0.05, 0.05], (200, 3)),
        rng.uniform([0.0, 0.0, 0.0], [0.05, 0.2, 0.05], (200, 3)),
    ]).astype(np.float32)
    theta = np.pi / 2
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0, 0, 1]], dtype=np.float32)
    src = (Rz @ tgt.T).T

    T, fitness = _fpfh_register(src, tgt, voxel_size=0.01)

    assert fitness > 0.3, f"FPFH fitness too low: {fitness}"
    R = T[:3, :3]
    t = T[:3, 3]
    src_aligned = (R @ src.T).T + t
    dists = np.linalg.norm(src_aligned[:, None] - tgt[None], axis=-1).min(axis=1)
    assert dists.mean() < 0.02, f"Mean dist after FPFH: {dists.mean():.4f}"


def test_fpfh_register_returns_4x4():
    from pipeline_multiview import _fpfh_register
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 0.1, (100, 3)).astype(np.float32)
    T, fitness = _fpfh_register(pts, pts, voxel_size=0.01)
    assert T.shape == (4, 4)
    assert isinstance(fitness, float)


# ── _icp_refine ───────────────────────────────────────────────────────────────

def test_icp_refine_identity_init_small_offset():
    """ICP should converge when source is slightly offset from target."""
    from pipeline_multiview import _icp_refine
    rng = np.random.default_rng(1)
    tgt = rng.uniform(0, 0.1, (300, 3)).astype(np.float32)
    src = tgt + np.array([0.005, 0.003, -0.002], dtype=np.float32)
    init_T = np.eye(4, dtype=np.float64)

    T, rmse = _icp_refine(src, tgt, init_T, threshold=0.02)

    assert T.shape == (4, 4)
    assert rmse < 0.005, f"ICP RMSE too high: {rmse:.4f}"
    t = T[:3, 3]
    assert abs(t[0] + 0.005) < 0.002 and abs(t[1] + 0.003) < 0.002


def test_icp_refine_returns_coarse_when_rmse_high():
    """_icp_refine just returns the result; caller decides if rmse is bad."""
    from pipeline_multiview import _icp_refine
    rng = np.random.default_rng(2)
    src = rng.uniform(0, 0.1, (100, 3)).astype(np.float32)
    tgt = rng.uniform(1.0, 1.1, (100, 3)).astype(np.float32)
    init_T = np.eye(4, dtype=np.float64)
    T, rmse = _icp_refine(src, tgt, init_T, threshold=0.02)
    assert T.shape == (4, 4)
    assert isinstance(rmse, float)


# ── _project_bbox_from_icp ────────────────────────────────────────────────────

def test_project_bbox_from_icp_identity_transform():
    """With identity transform and zero centroids, projection = standard pinhole."""
    from pipeline_multiview import _project_bbox_from_icp
    pts = np.array([[0.0, 0.0, 1.0],
                    [0.1, 0.0, 1.0],
                    [0.0, 0.1, 1.0]], dtype=np.float32)
    T = np.eye(4, dtype=np.float64)
    centroid_i = np.zeros(3, dtype=np.float32)
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0

    bbox = _project_bbox_from_icp(pts, T, centroid_i, fx, fy, cx, cy, H_i=480, W_i=640)

    assert len(bbox) == 4
    x1, y1, x2, y2 = bbox
    assert x1 < x2 and y1 < y2
    assert 290 <= x1 <= 310
    assert 210 <= y1 <= 230


def test_project_bbox_from_icp_clips_to_image():
    """Bbox should be clipped to image bounds."""
    from pipeline_multiview import _project_bbox_from_icp
    pts = np.array([[-10.0, -10.0, 1.0], [10.0, 10.0, 1.0]], dtype=np.float32)
    T = np.eye(4, dtype=np.float64)
    centroid_i = np.zeros(3, dtype=np.float32)
    bbox = _project_bbox_from_icp(pts, T, centroid_i, 500., 500., 320., 240.,
                                   H_i=480, W_i=640)
    x1, y1, x2, y2 = bbox
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 640 and y2 <= 480


# ── stage_merge_v2 integration ────────────────────────────────────────────────

def test_stage_merge_v2_single_view_identity():
    """With one non-ref view identical to ref (both flat depth), merged cloud is non-empty."""
    import tempfile, json
    from pathlib import Path as _Path
    from pipeline_multiview import stage_merge_v2, VRAMMonitor

    rng = np.random.default_rng(7)
    H, W = 20, 20
    depth = np.full((H, W), 1.0, dtype=np.float32)
    mask = np.ones((H, W), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = _Path(tmp)
        depths, masks, intris = [], [], []
        for i in range(2):
            d = tmp / f"depth_{i}.npy"; np.save(d, depth); depths.append(d)
            m = tmp / f"mask_{i}.npy";  np.save(m, mask);  masks.append(m)
            k = tmp / f"intri_{i}.json"
            k.write_text(json.dumps({"fx": 20., "fy": 20., "cx": 10., "cy": 10.}))
            intris.append(k)

        out_dir = tmp / "clouds"
        out_dir.mkdir()

        monitor = VRAMMonitor()
        result = stage_merge_v2(
            images=[tmp / "img_0.jpg", tmp / "img_1.jpg"],
            name="test",
            sam2_dir=tmp,
            segmask_paths=list(masks),
            best_idx=0,
            depths=depths,
            intrinsics=intris,
            out_dir=out_dir,
            voxel_size=0.05,
            monitor=monitor,
            sam2_model="small",
            registration="icp",
            icp_threshold=0.02,
            skip_sam2_bbox=True,
        )
        merged_ply = result["merged_ply"]
        assert merged_ply.exists()
        assert result["n_pts_after"] > 0
