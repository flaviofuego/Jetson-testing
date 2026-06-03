import sys
from pathlib import Path
import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "trellis"))


def test_generate_sdf_contains_model_name():
    from sdf_generator import generate_sdf
    mesh = trimesh.creation.icosphere(subdivisions=2)
    parts = [Path("mug_parts/convex_piece_000.obj")]
    xml = generate_sdf("mug", mesh, parts)
    assert 'name="mug"' in xml


def test_generate_sdf_contains_collision_block():
    from sdf_generator import generate_sdf
    mesh = trimesh.creation.icosphere(subdivisions=2)
    parts = [Path("mug_parts/convex_piece_000.obj")]
    xml = generate_sdf("mug", mesh, parts)
    assert "<collision" in xml
    assert "drake:declare_convex" in xml


def test_generate_sdf_visual_points_to_obj():
    from sdf_generator import generate_sdf
    mesh = trimesh.creation.icosphere(subdivisions=2)
    parts = [Path("mug_parts/convex_piece_000.obj")]
    xml = generate_sdf("mug", mesh, parts)
    assert "mug.obj" in xml


def test_generate_sdf_multiple_parts():
    from sdf_generator import generate_sdf
    mesh = trimesh.creation.icosphere(subdivisions=2)
    parts = [
        Path("mug_parts/convex_piece_000.obj"),
        Path("mug_parts/convex_piece_001.obj"),
    ]
    xml = generate_sdf("mug", mesh, parts)
    assert xml.count("<collision") == 2
