import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))


def test_build_docker_command_default_flags(tmp_path):
    from generate_asset_trellis import build_docker_command
    cmd = build_docker_command(
        image_path=tmp_path / "mug.png",
        project_root=tmp_path,
        name="mug",
        steps=4,
        texture_size=512,
    )
    assert "docker" in cmd
    assert "--gpus" in cmd
    assert "--steps" in cmd
    assert "4" in cmd
    assert "--texture-size" in cmd
    assert "512" in cmd
    assert "/input/mug.png" in cmd


def test_build_docker_command_custom_quality(tmp_path):
    from generate_asset_trellis import build_docker_command
    cmd = build_docker_command(
        image_path=tmp_path / "mug.png",
        project_root=tmp_path,
        name="mug",
        steps=12,
        texture_size=1024,
    )
    assert "12" in cmd
    assert "1024" in cmd


def test_resolve_asset_name_uses_stem_when_no_name():
    from generate_asset_trellis import resolve_asset_name
    assert resolve_asset_name(Path("path/to/mug.png"), None) == "mug"


def test_resolve_asset_name_uses_explicit_name():
    from generate_asset_trellis import resolve_asset_name
    assert resolve_asset_name(Path("path/to/mug.png"), "cup") == "cup"


def test_pipeline_argparse_accepts_all_flags():
    import subprocess
    pipeline = Path(__file__).parent.parent.parent / "trellis" / "pipeline.py"
    result = subprocess.run(
        [sys.executable, str(pipeline), "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--steps" in result.stdout
    assert "--texture-size" in result.stdout
