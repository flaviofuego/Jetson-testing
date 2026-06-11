#!/usr/bin/env python3
"""Apply MeshLab filters to OBJ files via CLI."""

import argparse
import pymeshlab


def main():
    parser = argparse.ArgumentParser(description="Apply MeshLab filters to a mesh")
    parser.add_argument("input", help="Input mesh file (OBJ, PLY, etc.)")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("--smooth-normals", action="store_true",
                        help="Smooth face normals")
    parser.add_argument("--smooth-iter", type=int, default=1,
                        help="Smooth normals iterations (default: 1, applies filter N times)")
    parser.add_argument("--depth-smooth", action="store_true",
                        help="Depth smooth (coord depth smoothing)")
    parser.add_argument("--depth-smooth-iter", type=int, default=3,
                        help="Depth smooth iterations (default: 3)")
    parser.add_argument("--cluster-decimation", action="store_true",
                        help="Clustering decimation")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Clustering threshold as %% of bbox diagonal (default: 0.3)")
    args = parser.parse_args()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.input)

    m = ms.current_mesh()
    print(f"Loaded: {m.vertex_number()} verts, {m.face_number()} faces")

    if args.cluster_decimation:
        print(f"Clustering decimation (threshold={args.threshold}%)...")
        ms.meshing_decimation_clustering(
            threshold=pymeshlab.PercentageValue(args.threshold)
        )

    if args.smooth_normals:
        print(f"Smooth face normals (iter={args.smooth_iter})...")
        for _ in range(args.smooth_iter):
            ms.apply_normal_smoothing_per_face()

    if args.depth_smooth:
        print(f"Depth smooth (iter={args.depth_smooth_iter})...")
        ms.apply_coord_depth_smoothing(stepsmoothnum=args.depth_smooth_iter)

    m = ms.current_mesh()
    print(f"Result:  {m.vertex_number()} verts, {m.face_number()} faces")

    ms.save_current_mesh(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
