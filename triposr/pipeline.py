"""
Runs inside the Docker container.
Usage:
  python pipeline.py --input /input/image.png --output /output --name mug
  python pipeline.py --input /input/image.png --output /output --name mug --device cpu
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import logging
from pathlib import Path

# Silence INFO-level noise from transformers / huggingface_hub
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from image_utils import preprocess_image
from mesh_utils import normalize_mesh, decompose_convex
from sdf_generator import generate_sdf


def run_triposr(image, device: str = "cuda") -> "trimesh.Trimesh":
    import torch
    from tsr.system import TSR

    system = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    ).to(device)
    system.renderer.set_chunk_size(8192)

    with torch.no_grad():
        scene_codes = system([image], device=device)
        meshes = system.extract_mesh(scene_codes, has_vertex_color=False, resolution=256)

    return meshes[0]


def main():
    parser = argparse.ArgumentParser(description="TripoSR asset generation pipeline")
    parser.add_argument("--input",  required=True, type=Path, help="Input image path")
    parser.add_argument("--output", required=True, type=Path, help="Output assets root dir")
    parser.add_argument("--name",   required=True, type=str,  help="Asset name")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Inference device (default: cuda)")
    args = parser.parse_args()

    asset_dir = args.output / args.name
    parts_dir = asset_dir / f"{args.name}_parts"
    asset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Preprocessing image: {args.input}")
    image = preprocess_image(args.input)

    print(f"[2/5] Running TripoSR inference on {args.device}...")
    raw_mesh = run_triposr(image, device=args.device)

    print("[3/5] Normalizing mesh (longest axis → 20 cm)...")
    mesh = normalize_mesh(raw_mesh)
    obj_path = asset_dir / f"{args.name}.obj"
    mesh.export(str(obj_path))
    print(f"      Saved: {obj_path}  ({len(mesh.faces)} faces)")

    print("[4/5] Convex decomposition...")
    try:
        parts = decompose_convex(mesh, parts_dir, args.name)
    except Exception as e:
        print(f"      coacd failed ({e}), falling back to convex hull")
        import trimesh
        hull = trimesh.convex.convex_hull(mesh)
        parts_dir.mkdir(parents=True, exist_ok=True)
        fallback = parts_dir / "convex_piece_000.obj"
        hull.export(str(fallback))
        parts = [fallback]
    print(f"      {len(parts)} convex part(s)")

    print("[5/5] Generating SDF...")
    relative_parts = [Path(f"{args.name}_parts") / p.name for p in parts]
    sdf_path = asset_dir / f"{args.name}.sdf"
    sdf_path.write_text(generate_sdf(args.name, mesh, relative_parts))
    print(f"      Saved: {sdf_path}")

    print(f"\nDone → {asset_dir}/")


if __name__ == "__main__":
    main()
