"""
Host-side wrapper: builds and calls the triposr Docker container.

Usage:
    python tools/generate_asset.py path/to/image.png
    python tools/generate_asset.py path/to/image.png --name mug
    python tools/generate_asset.py path/to/image.png --cpu        # CPU image
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
IMAGE_NAME = "triposr"
IMAGE_NAME_CPU = "triposr:cpu"


def resolve_asset_name(image_path: Path, name: str | None) -> str:
    if name:
        return name
    return image_path.stem


def build_docker_command(image_path: Path, project_root: Path, name: str, cpu: bool) -> list[str]:
    image_path = image_path.resolve()
    assets_dir = (project_root / "assets").resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path.home() / "models" / "huggingface"

    image = IMAGE_NAME_CPU if cpu else IMAGE_NAME
    gpu_flags = [] if cpu else ["--gpus", "all"]

    return [
        "docker", "run", "--rm",
        *gpu_flags,
        "-v", f"{image_path.parent}:/input:ro",
        "-v", f"{assets_dir}:/output",
        "-v", f"{models_dir}:/root/.cache/huggingface",
        image,
        "--input", f"/input/{image_path.name}",
        "--output", "/output",
        "--name", name,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate an OBJ+SDF asset from an image using TripoSR"
    )
    parser.add_argument("image", type=Path, help="Path to input PNG/JPG image")
    parser.add_argument("--name", default=None, help="Asset name (default: image filename stem)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU image instead of GPU")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    name = resolve_asset_name(args.image, args.name)
    cmd = build_docker_command(args.image, PROJECT_ROOT, name, args.cpu)

    print(f"Generating asset '{name}' from {args.image}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: Docker exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    asset_path = PROJECT_ROOT / "assets" / name
    print(f"\nAsset generated at: {asset_path}")
    print(f"  {name}.obj")
    print(f"  {name}.sdf")
    print(f"  {name}_parts/")


if __name__ == "__main__":
    main()
