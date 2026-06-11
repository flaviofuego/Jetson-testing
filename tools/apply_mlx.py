#!/usr/bin/env python3
"""Apply a MeshLab filter script (.mlx) to a mesh file."""

import argparse
import pymeshlab


def main():
    parser = argparse.ArgumentParser(description="Apply a .mlx filter script to a mesh")
    parser.add_argument("input", help="Input mesh file")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("script", help="MeshLab filter script (.mlx)")
    args = parser.parse_args()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.input)

    m = ms.current_mesh()
    print(f"Loaded: {m.vertex_number()} verts, {m.face_number()} faces")

    ms.load_filter_script(args.script)
    ms.apply_filter_script()

    m = ms.current_mesh()
    print(f"Result: {m.vertex_number()} verts, {m.face_number()} faces")

    ms.save_current_mesh(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
