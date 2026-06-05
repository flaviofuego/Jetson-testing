"""
Converts a Gaussian Splat PLY (from dvlt) to OBJ mesh via Poisson reconstruction.

Usage:
    python tools/ply_to_obj.py path/to/scene.ply
    python tools/ply_to_obj.py path/to/scene.ply --output path/to/output.obj
    python tools/ply_to_obj.py path/to/scene.ply --depth 10 --density 0.01
"""
import argparse
from pathlib import Path


def ply_to_obj(ply_path: Path, obj_path: Path, depth: int, density_threshold: float):
    try:
        import open3d as o3d
    except ImportError:
        import subprocess, sys
        print("Instalando open3d...")
        subprocess.run([sys.executable, "-m", "pip", "install", "open3d"], check=True)
        import open3d as o3d

    print(f"Leyendo {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  {len(pcd.points):,} puntos cargados")

    print("Estimando normales...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    print(f"Reconstruyendo mesh (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    if density_threshold > 0:
        import numpy as np
        vertices_to_remove = densities < (np.quantile(densities, density_threshold))
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"  Filtrado por densidad (quantile={density_threshold})")

    mesh.compute_vertex_normals()

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    print(f"  {len(mesh.triangles):,} triángulos")
    print(f"Guardado: {obj_path}")


def main():
    parser = argparse.ArgumentParser(description="Convierte PLY gaussiano a OBJ mesh")
    parser.add_argument("ply", type=Path, help="Archivo PLY de entrada")
    parser.add_argument("--output", type=Path, default=None,
                        help="Archivo OBJ de salida (default: mismo directorio que el PLY)")
    parser.add_argument("--depth", type=int, default=9,
                        help="Profundidad de reconstrucción Poisson — mayor = más detalle (default: 9)")
    parser.add_argument("--density", type=float, default=0.01,
                        help="Umbral de densidad para filtrar artefactos 0-1 (default: 0.01)")
    args = parser.parse_args()

    if not args.ply.exists():
        print(f"Error: archivo no encontrado: {args.ply}")
        raise SystemExit(1)

    obj_path = args.output or args.ply.with_suffix(".obj")
    ply_to_obj(args.ply, obj_path, depth=args.depth, density_threshold=args.density)


if __name__ == "__main__":
    main()
