"""Unit tests for --query-cloud argument in numcc/pipeline.py.

Full E2E testing requires numcc:x86 Docker image + GPU.
These tests cover argument parsing only.
"""
import sys
import types
import importlib.util
from pathlib import Path
import pytest

_NUMCC_PIPELINE_PATH = Path(__file__).parent.parent.parent / "models" / "numcc" / "pipeline.py"


def _stub_heavy_imports():
    """Stub only the 3 modules imported at top-level of models/numcc/pipeline.py.

    torch, trimesh, imageio, PIL, src.* are only used inside function bodies —
    stubbing them at module level would poison sys.modules for other tests that
    use the real libraries (e.g. open3d → scipy → checks for real torch.Tensor).
    """
    for fake_mod, attrs in [
        ("pointcloud_utils", {
            "load_depth": lambda p: None,
            "depth_to_pointcloud": lambda d, **kw: None,
            "subsample_pointcloud": lambda pts, n: pts,
            "load_intrinsics": lambda p: (None, None, None, None),
            "to_y_up": lambda pts: pts,
            "CAM_TO_YUP": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        }),
        ("mesh_utils", {"normalize_mesh": lambda m: m, "decompose_convex": lambda m, **kw: []}),
        ("sdf_generator", {"generate_sdf": lambda *a, **kw: None}),
    ]:
        if fake_mod not in sys.modules:
            m = types.ModuleType(fake_mod)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[fake_mod] = m


def _import_numcc_pipeline():
    """Import models/numcc/pipeline.py by file path to avoid sys.path ambiguity.

    tools/pipeline_scripts/pipeline.py also exists and would be found first
    when sys.path has tools/pipeline_scripts/ at index 0 (inserted by test_registration.py
    module-level code during pytest collection). Using spec_from_file_location bypasses
    sys.path entirely.
    """
    _stub_heavy_imports()
    spec = importlib.util.spec_from_file_location("_numcc_pipeline_mod", _NUMCC_PIPELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_numcc_pipeline_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_argparse_query_cloud_flag():
    """--query-cloud argument is accepted and stored as a Path."""
    numcc_pipeline = _import_numcc_pipeline()
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
    numcc_pipeline = _import_numcc_pipeline()
    parser = numcc_pipeline._build_parser()
    args = parser.parse_args([
        "--depth", "/tmp/d.npy", "--name", "obj", "--output", "/tmp/out",
    ])
    assert args.query_cloud is None
