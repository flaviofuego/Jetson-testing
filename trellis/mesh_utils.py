from pathlib import Path
import trimesh
import numpy as np


def glb_to_trimesh(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(glb_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        return loaded.dump(concatenate=True)
    return loaded


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_translation(-result.centroid)
    scale = 0.20 / result.extents.max()
    result.apply_scale(scale)
    return result


def decompose_convex(mesh: trimesh.Trimesh, output_dir: Path, name: str) -> list[Path]:
    import coacd
    output_dir.mkdir(parents=True, exist_ok=True)
    m = coacd.Mesh(
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(m)
    written = []
    for i, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        part_path = output_dir / f"convex_piece_{i:03d}.obj"
        part_mesh.export(str(part_path))
        written.append(part_path)
    return written
