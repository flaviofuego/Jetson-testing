"""Worker subprocess para reconstrucción Poisson aislada del proceso principal."""
import json, sys
import numpy as np
import open3d as o3d
from pathlib import Path

args = json.loads(sys.argv[1])
pcd = o3d.io.read_point_cloud(args["ply_in"])

if args["ply_save"]:
    o3d.io.write_point_cloud(args["ply_save"], pcd)
    print(f"PLY guardado: {Path(args['ply_save']).name}")

n = len(pcd.points)
depth = int(args["depth"])
if n < 5000:
    depth = min(depth, 7)
if n < 1000:
    depth = min(depth, 6)
if depth != int(args["depth"]):
    print(f"depth auto-reducido {args['depth']}→{depth} ({n:,} puntos)")

bbox = pcd.get_axis_aligned_bounding_box()
radius = float(bbox.get_max_extent()) * 0.05
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(k=15)

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
dt = float(args["density_threshold"])
if dt > 0:
    mask = densities < np.quantile(densities, dt)
    mesh.remove_vertices_by_mask(mask)
mesh.compute_vertex_normals()

import trimesh
v = np.asarray(mesh.vertices)
f = np.asarray(mesh.triangles)
n_arr = np.asarray(mesh.vertex_normals)
obj_path = Path(args["obj_out"])
obj_path.parent.mkdir(parents=True, exist_ok=True)
trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n_arr).export(str(obj_path))
print(f"{len(mesh.triangles):,} triángulos")
print(f"OBJ guardado: {obj_path.name}")
