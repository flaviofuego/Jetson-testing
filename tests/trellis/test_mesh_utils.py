import sys
from pathlib import Path
import pytest
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "trellis"))


def test_normalize_mesh_longest_axis_is_20cm():
    from mesh_utils import normalize_mesh
    box = trimesh.creation.box(extents=[1.0, 0.5, 0.3])
    result = normalize_mesh(box)
    assert abs(result.extents.max() - 0.20) < 1e-6


def test_normalize_mesh_centered_at_origin():
    from mesh_utils import normalize_mesh
    box = trimesh.creation.box(extents=[1.0, 0.5, 0.3])
    box.apply_translation([5.0, 5.0, 5.0])
    result = normalize_mesh(box)
    np.testing.assert_allclose(result.centroid, [0.0, 0.0, 0.0], atol=1e-6)


def test_glb_to_trimesh_returns_single_mesh(tmp_path):
    from mesh_utils import glb_to_trimesh
    sphere = trimesh.creation.icosphere(subdivisions=1)
    glb_path = tmp_path / "sphere.glb"
    sphere.export(str(glb_path))
    result = glb_to_trimesh(glb_path)
    assert isinstance(result, trimesh.Trimesh)
    assert len(result.faces) > 0


def test_glb_to_trimesh_scene_is_concatenated(tmp_path):
    from mesh_utils import glb_to_trimesh
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box())
    scene.add_geometry(trimesh.creation.icosphere(subdivisions=1))
    glb_path = tmp_path / "scene.glb"
    scene.export(str(glb_path))
    result = glb_to_trimesh(glb_path)
    assert isinstance(result, trimesh.Trimesh)
