#!/usr/bin/env python3
"""Smooth mesh while preserving geometric detail (removes lumps, keeps edges)."""

import argparse
import pymeshlab


def main():
    parser = argparse.ArgumentParser(description="Smooth mesh preserving detail")
    parser.add_argument("input", help="Input mesh file (OBJ, PLY, etc.)")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("--cluster-threshold", type=float, default=0.3,
                        help="Clustering decimation threshold %% of bbox (default: 0.3)")
    parser.add_argument("--taubin-iter", type=int, default=10,
                        help="Taubin smooth iterations (default: 10, more = smoother)")
    parser.add_argument("--twosteps-iter", type=int, default=3,
                        help="Two steps smooth iterations (default: 3)")
    parser.add_argument("--feature-angle", type=float, default=60.0,
                        help="Feature preservation angle in degrees (default: 60, lower = preserve more edges)")
    args = parser.parse_args()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.input)

    m = ms.current_mesh()
    print(f"Loaded: {m.vertex_number()} verts, {m.face_number()} faces")

    print(f"Clustering decimation (threshold={args.cluster_threshold}%)...")
    ms.meshing_decimation_clustering(
        threshold=pymeshlab.PercentageValue(args.cluster_threshold)
    )

    ms.meshing_remove_unreferenced_vertices()

    print(f"Taubin smooth (iter={args.taubin_iter})...")
    ms.apply_coord_taubin_smoothing(stepsmoothnum=args.taubin_iter)

    print(f"Two steps smooth (iter={args.twosteps_iter}, angle={args.feature_angle}deg)...")
    ms.apply_coord_two_steps_smoothing(
        stepsmoothnum=args.twosteps_iter,
        normalthr=args.feature_angle
    )

    print("Smooth face normals...")
    ms.apply_normal_smoothing_per_face()

    m = ms.current_mesh()
    print(f"Result: {m.vertex_number()} verts, {m.face_number()} faces")

    ms.save_current_mesh(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
