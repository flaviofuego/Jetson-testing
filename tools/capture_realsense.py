"""
Captures one RGB-D frame from an Intel RealSense camera and saves:
  - color.npy   (H, W, 3) uint8 RGB
  - depth.npy   (H, W) float32 meters
  - intrinsics.json  {fx, fy, cx, cy, width, height}

Requires pyrealsense2 on the host:
  pip install pyrealsense2

Usage:
    python tools/capture_realsense.py --output-dir tmp/mug
    python tools/capture_realsense.py --output-dir tmp/mug --name mug
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def capture_frame(output_dir: Path) -> None:
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("Error: pyrealsense2 not installed. Run: pip install pyrealsense2", file=sys.stderr)
        sys.exit(1)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Starting RealSense pipeline...")
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    # Discard first 30 frames to let auto-exposure settle
    print("Warming up camera (30 frames)...")
    for _ in range(30):
        pipeline.wait_for_frames()

    print("Capturing frame...")
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    if not color_frame or not depth_frame:
        print("Error: failed to capture frames", file=sys.stderr)
        pipeline.stop()
        sys.exit(1)

    color_np = np.asanyarray(color_frame.get_data())   # (H, W, 3) uint8 RGB
    depth_raw = np.asanyarray(depth_frame.get_data())  # (H, W) uint16 mm

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_m = depth_raw.astype(np.float32) * depth_scale  # -> meters

    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    intr_dict = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "width": intrinsics.width,
        "height": intrinsics.height,
    }

    pipeline.stop()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "color.npy"), color_np)
    np.save(str(output_dir / "depth.npy"), depth_m)
    (output_dir / "intrinsics.json").write_text(json.dumps(intr_dict, indent=2))

    print(f"Saved to {output_dir}/")
    print(f"  color.npy   shape={color_np.shape} dtype={color_np.dtype}")
    print(f"  depth.npy   shape={depth_m.shape}  dtype={depth_m.dtype}  "
          f"range=[{depth_m[depth_m > 0].min():.3f}, {depth_m.max():.3f}]m")
    print(f"  intrinsics.json  fx={intr_dict['fx']:.1f} fy={intr_dict['fy']:.1f} "
          f"cx={intr_dict['cx']:.1f} cy={intr_dict['cy']:.1f}")
    print(f"\nNext step:")
    print(f"  python tools/generate_asset_numcc.py \\")
    print(f"    --depth {output_dir}/depth.npy \\")
    print(f"    --color {output_dir}/color.npy \\")
    print(f"    --intrinsics {output_dir}/intrinsics.json \\")
    print(f"    --name <asset-name>")


def main():
    parser = argparse.ArgumentParser(
        description="Capture one RGB-D frame from RealSense -> .npy + intrinsics.json"
    )
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to save color.npy, depth.npy, intrinsics.json")
    parser.add_argument("--name", default=None, type=str,
                        help="Optional subfolder name under output-dir")
    args = parser.parse_args()

    output_dir = args.output_dir / args.name if args.name else args.output_dir
    capture_frame(output_dir)


if __name__ == "__main__":
    main()
