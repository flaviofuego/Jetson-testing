"""
Host-side wrapper: builds and calls the trellis Docker container.

Usage:
    python tools/generate_asset_trellis.py path/to/image.png
    python tools/generate_asset_trellis.py path/to/image.png --name mug
    python tools/generate_asset_trellis.py path/to/image.png --steps 12 --texture-size 1024
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
# Default image name; use trellis:x86 for local x86+GPU builds
IMAGE_NAME = "trellis:x86"
# Models are cached here on the host and mounted read-only into the container.
# Pre-download once with: python3 tools/download_models_trellis.py
MODELS_DIR = Path.home() / "models" / "huggingface"


def resolve_asset_name(image_path: Path, name: str | None) -> str:
    return name if name else image_path.stem


def build_docker_command(
    image_path: Path,
    project_root: Path,
    name: str,
    steps: int,
    texture_size: int,
) -> list[str]:
    image_path = image_path.resolve()
    assets_dir = (project_root / "assets").resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    return [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{image_path.parent}:/input:ro",
        "-v", f"{assets_dir}:/output",
        "-v", f"{MODELS_DIR}:/root/.cache/huggingface",
        IMAGE_NAME,
        "--input", f"/input/{image_path.name}",
        "--output", "/output",
        "--name", name,
        "--steps", str(steps),
        "--texture-size", str(texture_size),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate an OBJ+GLB+SDF asset from an image using TRELLIS"
    )
    parser.add_argument("image", type=Path, help="Path to input PNG/JPG image")
    parser.add_argument("--name", default=None,
                        help="Asset name (default: image filename stem)")
    parser.add_argument("--steps", default=4, type=int,
                        help="Diffusion steps for both samplers (default: 4, max: 12)")
    parser.add_argument("--texture-size", default=512, type=int,
                        help="GLB texture map resolution (default: 512, max: 1024)")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    name = resolve_asset_name(args.image, args.name)
    cmd = build_docker_command(
        args.image, PROJECT_ROOT, name, args.steps, args.texture_size
    )

    print(f"Generating asset '{name}' from {args.image}")
    print(f"Settings: steps={args.steps}, texture_size={args.texture_size}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: Docker exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    asset_path = PROJECT_ROOT / "assets" / name
    print(f"\nAsset generated at: {asset_path}")
    print(f"  {name}.glb   (textured)")
    print(f"  {name}.obj   (Drake visual)")
    print(f"  {name}.sdf   (Drake SDF)")
    print(f"  {name}_parts/")


if __name__ == "__main__":
    main()
