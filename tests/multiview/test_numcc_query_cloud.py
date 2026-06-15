"""Unit tests for --query-cloud argument in numcc/pipeline.py.

Full E2E testing requires numcc:x86 Docker image + GPU.
These tests cover argument parsing only.
"""
import sys
import types
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models" / "numcc"))


def _stub_heavy_imports():
    """Stub torch, trimesh, etc. so we can import the module header."""
    stubs = ["torch", "torch.nn.functional", "trimesh",
             "src.models.NU_MCC", "src.fns",
             "src.models", "src",
             "open3d", "imageio", "imageio.v2",
             "PIL", "PIL.Image"]
    for mod in stubs:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # torch needs a few attrs
    torch_mod = sys.modules["torch"]
    if not hasattr(torch_mod, "no_grad"):
        torch_mod.no_grad = lambda: (lambda f: f)

    # trimesh needs Trimesh class for type annotations in mesh_utils
    trimesh_mod = sys.modules["trimesh"]
    if not hasattr(trimesh_mod, "Trimesh"):
        trimesh_mod.Trimesh = type("Trimesh", (), {})

    # stub mesh_utils and sdf_generator so pipeline.py top-level imports succeed
    _numcc_dir = str(Path(__file__).parent.parent.parent / "models" / "numcc")
    for fake_mod, attrs in [
        ("mesh_utils", {"normalize_mesh": lambda m: m, "decompose_convex": lambda m, **kw: []}),
        ("sdf_generator", {"generate_sdf": lambda *a, **kw: None}),
        ("pointcloud_utils", {
            "load_depth": lambda p: None,
            "depth_to_pointcloud": lambda d, **kw: None,
            "subsample_pointcloud": lambda pts, n: pts,
            "load_intrinsics": lambda p: (None, None, None, None),
            "to_y_up": lambda pts: pts,
            "CAM_TO_YUP": [[1,0,0],[0,-1,0],[0,0,-1]],
        }),
    ]:
        if fake_mod not in sys.modules:
            m = types.ModuleType(fake_mod)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[fake_mod] = m
    return


def test_argparse_query_cloud_flag():
    """--query-cloud argument is accepted and stored as a Path."""
    _stub_heavy_imports()
    # Re-import fresh
    if "pipeline" in sys.modules:
        del sys.modules["pipeline"]
    import pipeline as numcc_pipeline
    parser = numcc_pipeline._build_parser()
    args = parser.parse_args([
        "--depth", "/tmp/d.npy",
        "--name", "obj",
        "--output", "/tmp/out",
        "--query-cloud", "/tmp/cloud.ply",
    ])
    assert args.query_cloud == Path("/tmp/cloud.ply")


def test_argparse_no_query_cloud_defaults_to_none():
    """Without --query-cloud, attribute is None."""
    _stub_heavy_imports()
    if "pipeline" in sys.modules:
        del sys.modules["pipeline"]
    import pipeline as numcc_pipeline
    parser = numcc_pipeline._build_parser()
    args = parser.parse_args([
        "--depth", "/tmp/d.npy", "--name", "obj", "--output", "/tmp/out",
    ])
    assert args.query_cloud is None
