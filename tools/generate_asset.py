"""
Host-side wrapper: builds and calls TripoSR, TRELLIS or NU-MCC Docker containers.

Usage:
    python tools/generate_asset.py path/to/image.png                   # TripoSR GPU
    python tools/generate_asset.py path/to/image.png --name mug        # custom name
    python tools/generate_asset.py path/to/image.png --cpu             # TripoSR CPU
    python tools/generate_asset.py path/to/image.png --model trellis   # TRELLIS GPU

    # NU-MCC — depth + optional color from an RGBD camera
    python tools/generate_asset.py --model numcc --depth depth.npy --name obj \\
        --intrinsics intrinsics.json
    python tools/generate_asset.py --model numcc --depth depth.npy --color color.npy \\
        --name obj --fx 579.4 --fy 579.4 --cx 319.5 --cy 239.5
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

IMAGES = {
    "triposr":     "triposr:x86",
    "triposr-cpu": "triposr:cpu",
    "trellis":     "trellis:x86",
    "numcc":       "numcc:x86",
    "sam2":        "sam2:x86",
}


def resolve_asset_name(path: Path, name: str | None) -> str:
    return name or path.stem


def build_triposr_trellis_command(image_path: Path, project_root: Path, name: str,
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


def build_sam2_command(image_path: Path, project_root: Path, name: str,
                       all_masks: bool) -> list[str]:
    image_path  = image_path.resolve()
    outputs_dir = (project_root / "data" / "outputs").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ckpts_dir   = Path.home() / "models" / "sam2"

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{image_path.parent}:/input:ro",
        "-v", f"{outputs_dir}:/output",
        "-v", f"{ckpts_dir}:/opt/sam2/checkpoints:ro",
        "-v", f"{project_root / 'submodules' / 'sam2' / 'pipeline.py'}:/opt/sam2/pipeline.py:ro",
        "--entrypoint", "python3",
        IMAGES["sam2"],
        "/opt/sam2/pipeline.py",
        "--input",  f"/input/{image_path.name}",
        "--output", "/output",
        "--name",   name,
    ]
    if all_masks:
        cmd.append("--all-masks")
    return cmd


def build_numcc_command(depth_path: Path, color_path: Path | None,
                        intrinsics_path: Path | None,
                        fx: float | None, fy: float | None,
                        cx: float | None, cy: float | None,
                        project_root: Path, name: str,
                        udf_threshold: float) -> list[str]:
    depth_path = depth_path.resolve()
    assets_dir = (project_root / "assets").resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path.home() / "models" / "numcc"

    # All input files must live in the same directory (single mount point)
    input_dir = depth_path.parent
    if color_path and color_path.resolve().parent != input_dir:
        print("Error: --depth and --color must be in the same directory", file=sys.stderr)
        sys.exit(1)
    if intrinsics_path and intrinsics_path.resolve().parent != input_dir:
        print("Error: --intrinsics must be in the same directory as --depth", file=sys.stderr)
        sys.exit(1)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{assets_dir}:/output",
        "-v", f"{models_dir}:/opt/models:ro",
        IMAGES["numcc"],
        "--depth",  f"/input/{depth_path.name}",
        "--name",   name,
        "--output", "/output",
        "--udf-threshold", str(udf_threshold),
    ]

    if color_path:
        cmd += ["--color", f"/input/{color_path.name}"]
    if intrinsics_path:
        cmd += ["--intrinsics", f"/input/{intrinsics_path.name}"]
    else:
        cmd += ["--fx", str(fx), "--fy", str(fy), "--cx", str(cx), "--cy", str(cy)]

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D asset from sensor data using TripoSR, TRELLIS or NU-MCC"
    )
    parser.add_argument("image", nargs="?", type=Path,
                        help="Input image (TripoSR/TRELLIS). Omit for --model numcc.")
    parser.add_argument("--name",  default=None, help="Asset name (default: input file stem)")
    parser.add_argument("--model", default="triposr",
                        choices=["triposr", "trellis", "numcc", "sam2"],
                        help="Model to use (default: triposr)")
    parser.add_argument("--cpu",       action="store_true", help="TripoSR CPU-only mode")
    parser.add_argument("--all-masks", action="store_true", help="SAM2: save every mask as individual PNG")

    # NU-MCC specific
    numcc = parser.add_argument_group("NU-MCC options (--model numcc)")
    numcc.add_argument("--depth",      type=Path, help="Depth map (.npy float32 m or .png uint16 mm)")
    numcc.add_argument("--color",      type=Path, default=None, help="Color image (.npy uint8 or .png)")
    numcc.add_argument("--intrinsics", type=Path, default=None, help="intrinsics.json with fx/fy/cx/cy")
    numcc.add_argument("--fx",         type=float, default=None)
    numcc.add_argument("--fy",         type=float, default=None)
    numcc.add_argument("--cx",         type=float, default=None)
    numcc.add_argument("--cy",         type=float, default=None)
    numcc.add_argument("--udf-threshold", type=float, default=0.03,
                       help="NU-MCC UDF threshold for surface extraction (default: 0.03)")

    args = parser.parse_args()

    assets_dir = PROJECT_ROOT / "assets"

    if args.model == "sam2":
        if not args.image:
            parser.error("--model sam2 requires a positional image argument")
        if not args.image.exists():
            print(f"Error: image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        name = resolve_asset_name(args.image, args.name)
        cmd  = build_sam2_command(args.image, PROJECT_ROOT, name, args.all_masks)
        print(f"Segmenting '{name}' from {args.image} using SAM2 small")

    elif args.model == "numcc":
        if not args.depth:
            parser.error("--model numcc requires --depth")
        if not args.depth.exists():
            print(f"Error: depth file not found: {args.depth}", file=sys.stderr)
            sys.exit(1)
        if args.intrinsics is None and any(v is None for v in [args.fx, args.fy, args.cx, args.cy]):
            parser.error("--model numcc requires --intrinsics or all of --fx --fy --cx --cy")

        name = resolve_asset_name(args.depth, args.name)
        cmd  = build_numcc_command(
            depth_path=args.depth,
            color_path=args.color,
            intrinsics_path=args.intrinsics,
            fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
            project_root=PROJECT_ROOT,
            name=name,
            udf_threshold=args.udf_threshold,
        )
        print(f"Generating asset '{name}' from {args.depth} using NU-MCC")
    else:
        if not args.image:
            parser.error(f"--model {args.model} requires a positional image argument")
        if not args.image.exists():
            print(f"Error: image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        name = resolve_asset_name(args.image, args.name)
        cmd  = build_triposr_trellis_command(
            args.image, PROJECT_ROOT, name, args.model, args.cpu
        )
        print(f"Generating asset '{name}' from {args.image} using {args.model.upper()}")

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: Docker exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nAsset generated at: {assets_dir / name}")


if __name__ == "__main__":
    main()
