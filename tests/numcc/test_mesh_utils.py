import sys
from pathlib import Path
import pytest
import trimesh
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "numcc"))


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
