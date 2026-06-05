"""
Host-side wrapper for the numcc Docker container.

Usage:
    python tools/generate_asset_numcc.py \
        --depth tmp/mug/depth.npy \
        --color tmp/mug/color.npy \
        --intrinsics tmp/mug/intrinsics.json \
        --name mug

    python tools/generate_asset_numcc.py \
        --depth scans/depth.png \
        --fx 615.0 --fy 615.0 --cx 320.0 --cy 240.0 \
        --name mug

    python tools/generate_asset_numcc.py --depth ... --name mug --x86
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = Path.home() / "models" / "numcc"
IMAGE_JETSON = "numcc"
IMAGE_X86 = "numcc:x86"


def resolve_asset_name(depth_path: Path, name: str | None) -> str:
    return name if name else depth_path.stem


def build_docker_command(
    depth_path: Path,
    color_path: Path | None,
    intrinsics_path: Path | None,
    fx: float | None,
    fy: float | None,
    cx: float | None,
    cy: float | None,
    name: str,
    project_root: Path,
    x86: bool,
    extra_args: list[str],
) -> list[str]:
    depth_path = depth_path.resolve()
    assets_dir = (project_root / "assets").resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    models_dir = MODELS_DIR.resolve()

    # All input files must be in the same directory so a single volume mount works.
    # We use the depth file's directory as the input mount point and verify the
    # other files live there too.
    input_dir = depth_path.parent
    for label, path in [("color", color_path), ("intrinsics", intrinsics_path)]:
        if path is not None and Path(path).resolve().parent != input_dir:
            raise ValueError(
                f"--{label} must be in the same directory as --depth ({input_dir}).\n"
                f"Got: {path}"
            )

    if x86:
        image = IMAGE_X86
        gpu_flags = ["--gpus", "all"]
    else:
        image = IMAGE_JETSON
        gpu_flags = ["--runtime=nvidia"]

    cmd = [
        "docker", "run", "--rm",
        *gpu_flags,
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{assets_dir}:/output",
        "-v", f"{models_dir}:/opt/models:ro",
        image,
        "--depth", f"/input/{depth_path.name}",
        "--output", "/output",
        "--name", name,
    ]

    if color_path is not None:
        cmd += ["--color", f"/input/{Path(color_path).name}"]

    if intrinsics_path is not None:
        cmd += ["--intrinsics", f"/input/{Path(intrinsics_path).name}"]
    elif all(v is not None for v in [fx, fy, cx, cy]):
        cmd += ["--fx", str(fx), "--fy", str(fy), "--cx", str(cx), "--cy", str(cy)]

    cmd += extra_args
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Drake asset from RGB-D depth using P2C + NU-MCC"
    )
    parser.add_argument("--depth",      required=True, type=Path)
    parser.add_argument("--color",      default=None,  type=Path)
    parser.add_argument("--intrinsics", default=None,  type=Path)
    parser.add_argument("--fx",         default=None,  type=float)
    parser.add_argument("--fy",         default=None,  type=float)
    parser.add_argument("--cx",         default=None,  type=float)
    parser.add_argument("--cy",         default=None,  type=float)
    parser.add_argument("--name",       default=None,  type=str)
    parser.add_argument("--x86",        action="store_true")
    parser.add_argument("--n-input",    default=None,  type=int)
    parser.add_argument("--n-output",   default=None,  type=int)
    args = parser.parse_args()

    if not args.depth.exists():
        print(f"Error: depth file not found: {args.depth}", file=sys.stderr)
        sys.exit(1)

    if args.intrinsics is None and any(v is None for v in [args.fx, args.fy, args.cx, args.cy]):
        print("Error: provide either --intrinsics or all of --fx --fy --cx --cy", file=sys.stderr)
        sys.exit(1)

    name = resolve_asset_name(args.depth, args.name)

    extra: list[str] = []
    if args.n_input is not None:
        extra += ["--n-input", str(args.n_input)]
    if args.n_output is not None:
        extra += ["--n-output", str(args.n_output)]

    cmd = build_docker_command(
        depth_path=args.depth,
        color_path=args.color,
        intrinsics_path=args.intrinsics,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        name=name,
        project_root=PROJECT_ROOT,
        x86=args.x86,
        extra_args=extra,
    )

    print(f"Generating asset '{name}' from {args.depth}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: Docker exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nAsset generated at: {PROJECT_ROOT / 'assets' / name}")


if __name__ == "__main__":
    main()
