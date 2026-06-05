"""
Segmenta objetos de una nube de puntos PLY y exporta un OBJ por objeto.

Flujo:
    1. Cargar PLY (dvlt / point cloud estándar)
    2. Denoising
    3. RANSAC iterativo → eliminar planos estructurales (mesa, pared, suelo)
    4. DBSCAN → un cluster por objeto
    5. Filtro de tamaño + opción --count
    6. Poisson reconstruction → OBJ por cluster

Usage:
    python tools/segment_objects.py scene.ply --output-dir meshes/
    python tools/segment_objects.py scene.ply --output-dir meshes/ --count 2
    python tools/segment_objects.py scene.ply --output-dir meshes/ --save-clusters
    python tools/segment_objects.py scene.ply --output-dir meshes/ --eps 0.03 --min-points 30
"""
import argparse
from pathlib import Path


def load_and_denoise(ply_path: Path, o3d):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  {len(pcd.points):,} puntos cargados")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  {len(pcd.points):,} puntos tras denoising")
    return pcd


def remove_planes_iterative(pcd, o3d, max_planes: int, plane_ratio: float, distance_threshold: float):
    """
    Elimina planos estructurales (mesa, pared, suelo) con RANSAC iterativo.
    Se detiene cuando ningún plano ocupa más del plane_ratio del cloud original.
    El criterio usa el total original para no eliminar objetos pequeños al final.
    """
    total_original = len(pcd.points)
    for i in range(max_planes):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        ratio = len(inliers) / total_original
        a, b, c, d = plane_model
        print(f"  Plano {i+1}: {len(inliers):,} inliers ({ratio:.1%}) — "
              f"{a:.2f}x+{b:.2f}y+{c:.2f}z+{d:.2f}=0")
        if ratio < plane_ratio:
            print(f"  → plano < {plane_ratio:.0%} del cloud original, parando")
            break
        pcd = pcd.select_by_index(inliers, invert=True)
    print(f"  {len(pcd.points):,} puntos tras remoción de planos")
    return pcd


def cluster_objects(pcd, o3d, eps: float, min_points: int):
    """DBSCAN clustering. Devuelve lista de nubes, una por objeto detectado."""
    import numpy as np
    labels = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    labels = np.asarray(labels)
    n_clusters = int(labels.max()) + 1
    print(f"  {n_clusters} cluster(s) detectado(s) (eps={eps}, min_points={min_points})")
    clusters = []
    for i in range(n_clusters):
        idx = list(np.where(labels == i)[0])
        cluster_pcd = pcd.select_by_index(idx)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        print(f"    Cluster {i}: {len(cluster_pcd.points):,} puntos — "
              f"extent {[f'{v:.3f}' for v in bbox.get_extent()]}")
        clusters.append(cluster_pcd)
    return clusters


def filter_clusters(clusters, min_size: float, max_size: float, count):
    """
    Descarta clusters demasiado pequeños (ruido) o demasiado grandes (plano residual).
    Si count es especificado, devuelve los N clusters más grandes.
    """
    if not clusters:
        return []
    total = sum(len(c.points) for c in clusters)
    filtered = [
        c for c in clusters
        if min_size * total <= len(c.points) <= max_size * total
    ]
    n_discarded = len(clusters) - len(filtered)
    if n_discarded:
        print(f"  {n_discarded} cluster(s) descartado(s) por tamaño")
    if count is not None:
        filtered = sorted(filtered, key=lambda c: len(c.points), reverse=True)[:count]
        print(f"  Seleccionados top {count} cluster(s) por tamaño")
    return filtered


def reconstruct_mesh(pcd, o3d, depth: int, density_threshold: float):
    """Poisson reconstruction con radio de normales adaptado al bounding box del cluster.
    Reduce depth automáticamente para clusters escasos."""
    import numpy as np
    n_points = len(pcd.points)
    # Poisson a depth D necesita ~8^D puntos para ser estable; bajamos si hay pocos
    auto_depth = depth
    if n_points < 5_000:
        auto_depth = min(depth, 7)
    if n_points < 1_000:
        auto_depth = min(depth, 6)
    if auto_depth != depth:
        print(f"    depth auto-reducido {depth}→{auto_depth} ({n_points:,} puntos)")

    bbox = pcd.get_axis_aligned_bounding_box()
    radius = float(bbox.get_max_extent()) * 0.05
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=auto_depth)
    if density_threshold > 0:
        vertices_to_remove = densities < np.quantile(densities, density_threshold)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_obj(mesh, obj_path: Path):
    import numpy as np
    import trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(obj_path))


def _reconstruct_in_subprocess(pcd, o3d, obj_path: Path, ply_path, depth: int, density_threshold: float):
    """
    Ejecuta Poisson en un subprocess aislado para que un segfault de Open3D
    no mate el proceso principal.
    """
    import sys, subprocess, tempfile, json, os

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_ply = tmp.name
    o3d.io.write_point_cloud(tmp_ply, pcd)

    args = json.dumps({
        "ply_in": tmp_ply,
        "obj_out": str(obj_path),
        "ply_save": str(ply_path) if ply_path else "",
        "depth": depth,
        "density_threshold": density_threshold,
    })

    worker = Path(__file__).parent / "_poisson_worker.py"
    result = subprocess.run(
        [sys.executable, str(worker), args],
        capture_output=True, text=True
    )

    for line in result.stdout.splitlines():
        print(f"    {line}")

    if result.returncode != 0:
        # SIGSEGV (-11): reintentar con depth reducido
        if result.returncode == -11 and depth > 6:
            retry_depth = depth - 2
            print(f"    Segfault — reintentando con depth {retry_depth}...")
            args_retry = json.dumps({
                "ply_in": tmp_ply,
                "obj_out": str(obj_path),
                "ply_save": "",
                "depth": retry_depth,
                "density_threshold": density_threshold,
            })
            result2 = subprocess.run(
                [sys.executable, str(worker), args_retry],
                capture_output=True, text=True
            )
            for line in result2.stdout.splitlines():
                print(f"    {line}")
            if result2.returncode != 0:
                print(f"    Error en reintento (código {result2.returncode}) — cluster omitido")
        else:
            print(f"    Error en reconstrucción (código {result.returncode}) — cluster omitido")
            lines = [l for l in result.stderr.splitlines() if l.strip() and "[ERROR]" not in l]
            if lines:
                print(f"    {lines[-1]}")

    try:
        os.unlink(tmp_ply)
    except OSError:
        pass


def segment_objects(
    ply_path: Path,
    output_dir: Path,
    name_prefix: str,
    max_planes: int,
    plane_ratio: float,
    plane_distance: float,
    plane_removal: bool,
    eps: float,
    min_points: int,
    min_size: float,
    max_size: float,
    count,
    poisson_depth: int,
    density_threshold: float,
    save_clusters: bool,
):
    try:
        import open3d as o3d
    except ImportError:
        import subprocess, sys
        print("Instalando open3d...")
        subprocess.run([sys.executable, "-m", "pip", "install", "open3d"], check=True)
        import open3d as o3d

    print(f"\n[1/4] Cargando y denoising: {ply_path}")
    pcd = load_and_denoise(ply_path, o3d)

    if plane_removal:
        print(f"\n[2/4] Eliminando planos estructurales "
              f"(max={max_planes}, ratio={plane_ratio:.0%}, dist={plane_distance})...")
        pcd = remove_planes_iterative(pcd, o3d, max_planes, plane_ratio, plane_distance)
    else:
        print("\n[2/4] Saltando remoción de planos (--no-plane-removal)")

    print(f"\n[3/4] Clustering DBSCAN...")
    clusters = cluster_objects(pcd, o3d, eps=eps, min_points=min_points)
    clusters = filter_clusters(clusters, min_size=min_size, max_size=max_size, count=count)

    if not clusters:
        print("\nNo quedaron clusters tras el filtrado. Ajusta --eps, --min-points o --min-size.")
        return

    print(f"\n[4/4] Reconstruyendo {len(clusters)} objeto(s)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, cluster_pcd in enumerate(clusters):
        label = f"{name_prefix}_{i:02d}"
        print(f"\n  {label}: {len(cluster_pcd.points):,} puntos")

        if save_clusters:
            ply_out_path = output_dir / f"{label}.ply"
        else:
            ply_out_path = None

        obj_path = output_dir / f"{label}.obj"
        _reconstruct_in_subprocess(
            cluster_pcd, o3d, obj_path, ply_out_path, poisson_depth, density_threshold
        )

    print(f"\nListo — {len(clusters)} OBJ(s) en {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Segmenta objetos de una nube PLY y exporta un OBJ por objeto"
    )
    parser.add_argument("ply", type=Path, help="PLY de entrada (dvlt)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directorio de salida (default: <ply_dir>/objects/)")
    parser.add_argument("--name", default="object",
                        help="Prefijo de nombre para los OBJs (default: object)")

    # Remoción de planos
    parser.add_argument("--no-plane-removal", action="store_true",
                        help="Omitir remoción de planos estructurales")
    parser.add_argument("--max-planes", type=int, default=3,
                        help="Máximo de planos a eliminar (default: 3)")
    parser.add_argument("--plane-ratio", type=float, default=0.05,
                        help="Para si el plano < X fracción del cloud original (default: 0.05)")
    parser.add_argument("--plane-distance", type=float, default=0.02,
                        help="Distancia RANSAC al plano (default: 0.02)")

    # Clustering
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Radio DBSCAN — bajar para separar objetos juntos (default: 0.05)")
    parser.add_argument("--min-points", type=int, default=50,
                        help="Densidad mínima DBSCAN (default: 50)")

    # Filtros post-cluster
    parser.add_argument("--min-size", type=float, default=0.01,
                        help="Fracción mínima del total para conservar un cluster (default: 0.01)")
    parser.add_argument("--max-size", type=float, default=0.90,
                        help="Fracción máxima del total para conservar un cluster (default: 0.90)")
    parser.add_argument("--count", type=int, default=None,
                        help="Forzar top-N clusters por tamaño (default: automático)")

    # Reconstrucción
    parser.add_argument("--depth", type=int, default=9,
                        help="Profundidad Poisson (default: 9)")
    parser.add_argument("--density", type=float, default=0.01,
                        help="Umbral densidad Poisson 0-1 (default: 0.01)")

    # Debug
    parser.add_argument("--save-clusters", action="store_true",
                        help="Guardar PLY intermedio por cluster para inspección")

    args = parser.parse_args()

    if not args.ply.exists():
        print(f"Error: {args.ply} no existe")
        raise SystemExit(1)

    output_dir = args.output_dir or args.ply.parent / "objects"

    segment_objects(
        ply_path=args.ply,
        output_dir=output_dir,
        name_prefix=args.name,
        max_planes=args.max_planes,
        plane_ratio=args.plane_ratio,
        plane_distance=args.plane_distance,
        plane_removal=not args.no_plane_removal,
        eps=args.eps,
        min_points=args.min_points,
        min_size=args.min_size,
        max_size=args.max_size,
        count=args.count,
        poisson_depth=args.depth,
        density_threshold=args.density,
        save_clusters=args.save_clusters,
    )


if __name__ == "__main__":
    main()
