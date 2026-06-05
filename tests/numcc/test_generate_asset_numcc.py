import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))


def test_build_docker_command_contains_required_parts(tmp_path):
    from generate_asset_numcc import build_docker_command
    depth = tmp_path / "depth.npy"
    depth.write_bytes(b"")
    cmd = build_docker_command(
        depth_path=depth,
        color_path=None,
        intrinsics_path=None,
        fx=615.0, fy=615.0, cx=320.0, cy=240.0,
        name="mug",
        project_root=tmp_path,
        x86=False,
        extra_args=[],
    )
    assert "docker" in cmd
    assert "--runtime=nvidia" in cmd
    assert "numcc" in cmd
    assert "--depth" in cmd
    assert "/input/depth.npy" in cmd
    assert "--name" in cmd
    assert "mug" in cmd


def test_build_docker_command_x86_uses_correct_image(tmp_path):
    from generate_asset_numcc import build_docker_command
    depth = tmp_path / "depth.npy"
    depth.write_bytes(b"")
    cmd = build_docker_command(
        depth_path=depth,
        color_path=None,
        intrinsics_path=None,
        fx=615.0, fy=615.0, cx=320.0, cy=240.0,
        name="mug",
        project_root=tmp_path,
        x86=True,
        extra_args=[],
    )
    assert "numcc:x86" in cmd
    assert "--runtime=nvidia" not in cmd
    assert "--gpus" in cmd


def test_build_docker_command_passes_intrinsics_json(tmp_path):
    from generate_asset_numcc import build_docker_command
    depth = tmp_path / "depth.npy"
    depth.write_bytes(b"")
    intr = tmp_path / "intrinsics.json"
    intr.write_text(json.dumps({"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}))
    cmd = build_docker_command(
        depth_path=depth,
        color_path=None,
        intrinsics_path=intr,
        fx=None, fy=None, cx=None, cy=None,
        name="mug",
        project_root=tmp_path,
        x86=False,
        extra_args=[],
    )
    assert "--intrinsics" in cmd
    assert "/input/intrinsics.json" in cmd


def test_resolve_asset_name_uses_depth_stem():
    from generate_asset_numcc import resolve_asset_name
    assert resolve_asset_name(Path("scans/mug_depth.npy"), None) == "mug_depth"


def test_resolve_asset_name_uses_explicit_name():
    from generate_asset_numcc import resolve_asset_name
    assert resolve_asset_name(Path("scans/mug_depth.npy"), "mug") == "mug"
