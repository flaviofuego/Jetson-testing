"""
Runs inside the Docker container.
Usage:
  python pipeline.py --input /input/image.png --output /output --name mug
  python pipeline.py --input /input/image.png --output /output --name mug --steps 12 --texture-size 1024
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import logging
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from image_utils import preprocess_image
from mesh_utils import glb_to_trimesh, normalize_mesh, decompose_convex
from sdf_generator import generate_sdf


def run_trellis(image, steps: int, texture_size: int):
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large"
    )
    pipeline.cuda()

    with torch.no_grad():
        outputs = pipeline.run(
            image,
            seed=42,
            sparse_structure_sampler_params={"steps": steps, "cfg_strength": 3.0},
            slat_sampler_params={"steps": steps, "cfg_strength": 2.0},
        )

    return postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.97,
        texture_size=texture_size,
    )


def main():
    parser = argparse.ArgumentParser(description="TRELLIS asset generation pipeline")
    parser.add_argument("--input",        required=True, type=Path)
    parser.add_argument("--output",       required=True, type=Path)
    parser.add_argument("--name",         required=True, type=str)
    parser.add_argument("--device",       default="cuda", choices=["cuda"])
    parser.add_argument("--steps",        default=4,   type=int,
                        help="Diffusion steps for both samplers (default: 4, max: 12)")
    parser.add_argument("--texture-size", default=512, type=int,
                        help="GLB texture map resolution (default: 512, max: 1024)")
    args = parser.parse_args()

    asset_dir = args.output / args.name
    parts_dir = asset_dir / f"{args.name}_parts"
    asset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Preprocessing image: {args.input}")
    image = preprocess_image(args.input)

    print(f"[2/6] Running TRELLIS inference (steps={args.steps}, texture_size={args.texture_size})...")
    glb_object = run_trellis(image, steps=args.steps, texture_size=args.texture_size)

    glb_path = asset_dir / f"{args.name}.glb"
    glb_object.export(str(glb_path))
    print(f"      Saved: {glb_path}")

    print("[3/6] Extracting mesh from GLB...")
    raw_mesh = glb_to_trimesh(glb_path)

    print("[4/6] Normalizing mesh (longest axis → 20 cm)...")
    mesh = normalize_mesh(raw_mesh)
    obj_path = asset_dir / f"{args.name}.obj"
    mesh.export(str(obj_path))
    print(f"      Saved: {obj_path}  ({len(mesh.faces)} faces)")

    print("[5/6] Convex decomposition...")
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

    print("[6/6] Generating SDF...")
    relative_parts = [Path(f"{args.name}_parts") / p.name for p in parts]
    sdf_path = asset_dir / f"{args.name}.sdf"
    sdf_path.write_text(generate_sdf(args.name, mesh, relative_parts))
    print(f"      Saved: {sdf_path}")

    print(f"\nDone → {asset_dir}/")
    print(f"  {args.name}.glb   (textured)")
    print(f"  {args.name}.obj   (Drake visual)")
    print(f"  {args.name}.sdf   (Drake SDF)")
    print(f"  {args.name}_parts/")


if __name__ == "__main__":
    main()
