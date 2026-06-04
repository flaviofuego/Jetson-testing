"""
Host-side wrapper: builds and calls TripoSR or TRELLIS Docker containers.

Usage:
    python tools/generate_asset.py path/to/image.png                   # TripoSR GPU
    python tools/generate_asset.py path/to/image.png --name mug        # custom name
    python tools/generate_asset.py path/to/image.png --cpu             # TripoSR CPU
    python tools/generate_asset.py path/to/image.png --model trellis   # TRELLIS GPU
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

IMAGES = {
    "triposr":     "triposr",
    "triposr-cpu": "triposr:cpu",
    "trellis":     "trellis",
}


def resolve_asset_name(image_path: Path, name: str | None) -> str:
    return name or image_path.stem


def build_docker_command(image_path: Path, project_root: Path, name: str,
                         model: str, cpu: bool) -> list[str]:
    image_path = image_path.resolve()
    assets_dir = (project_root / "assets").resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path.home() / "models" / "huggingface"
    u2net_dir  = Path.home() / ".u2net"
    u2net_dir.mkdir(parents=True, exist_ok=True)

    if cpu:
        image = IMAGES["triposr-cpu"]
        gpu_flags = []
    else:
        image = IMAGES[model]
        gpu_flags = ["--runtime=nvidia"]

    return [
        "docker", "run", "--rm",
        *gpu_flags,
        "-v", f"{image_path.parent}:/input:ro",
        "-v", f"{assets_dir}:/output",
        "-v", f"{models_dir}:/root/.cache/huggingface",
        "-v", f"{u2net_dir}:/root/.u2net",
        image,
        "--input",  f"/input/{image_path.name}",
        "--output", "/output",
        "--name",   name,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D asset from an image using TripoSR or TRELLIS"
    )
    parser.add_argument("image", type=Path, help="Path to input PNG/JPG image")
    parser.add_argument("--name",  default=None, help="Asset name (default: image stem)")
    parser.add_argument("--model", default="triposr", choices=["triposr", "trellis"],
                        help="Model to use (default: triposr)")
    parser.add_argument("--cpu",   action="store_true", help="TripoSR CPU-only mode")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    name = resolve_asset_name(args.image, args.name)
    cmd  = build_docker_command(args.image, PROJECT_ROOT, name, args.model, args.cpu)

    print(f"Generating asset '{name}' from {args.image} using {args.model.upper()}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: Docker exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    asset_path = PROJECT_ROOT / "assets" / name
    print(f"\nAsset generated at: {asset_path}")


if __name__ == "__main__":
    main()
