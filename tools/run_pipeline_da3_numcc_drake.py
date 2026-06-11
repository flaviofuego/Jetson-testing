#!/usr/bin/env python3
"""
Full pipeline: Depth Anything 3 → SAM2 → numcc → Drake
Measures VRAM at each stage via nvidia-smi polling (200 ms interval).

Usage:
  python3 tools/run_pipeline_da3_numcc_drake.py data/images/taza.jpeg --name taza
  python3 tools/run_pipeline_da3_numcc_drake.py data/images/taza.jpeg --name taza --drake-interactive
  python3 tools/run_pipeline_da3_numcc_drake.py data/images/taza.jpeg --name taza --skip-da3 --skip-sam2

Outputs under data/outputs/pipeline/<name>/:
  da3_raw/          - raw DA3 npz
  numcc_input/      - depth_masked.npy, color, intrinsics.json
  vram_profile.csv  - per-sample timestamp + MB
assets/<name>/      - mesh OBJ, SDF (numcc output)
"""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DA3_CLI      = Path("/home/worker-node-4/Documents/GitHub/UniWhere/.venv/bin/da3")
ASSETS_DIR   = PROJECT_ROOT / "assets"
MODELS_NUMCC = Path.home() / "models" / "numcc"
MODELS_SAM2  = Path.home() / "models" / "sam2"


# ─── VRAM monitor ─────────────────────────────────────────────────────────────

class VRAMMonitor:
    def __init__(self, poll_ms: int = 200):
        self._poll_ms = poll_ms
        self._data: list[tuple[float, int]] = []  # (epoch_s, used_mb)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while self._running:
            mb = self._query()
            if mb is not None:
                with self._lock:
                    self._data.append((time.time(), mb))
            time.sleep(self._poll_ms / 1000.0)

    def _query(self) -> int | None:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                return int(r.stdout.strip())
        except Exception:
            pass
        return None

    def current_mb(self) -> int:
        v = self._query()
        return v if v is not None else 0

    def peak_mb(self, t_start: float, t_end: float) -> int:
        with self._lock:
            vals = [mb for t, mb in self._data if t_start <= t <= t_end]
        return max(vals) if vals else self.current_mb()

    def save_csv(self, path: Path):
        with self._lock:
            rows = list(self._data)
        with open(path, "w") as f:
            f.write("timestamp,used_mb\n")
            for t, mb in rows:
                f.write(f"{t:.3f},{mb}\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


# ─── Stage 1: Depth Anything 3 ────────────────────────────────────────────────

def stage_da3(image_path: Path, out_dir: Path, monitor: VRAMMonitor) -> tuple[Path, float, float]:
    header("STAGE 1 — Depth Anything 3")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run([
        str(DA3_CLI), "image", str(image_path),
        "--export-format", "npz",
        "--export-dir", str(out_dir),
        "--auto-cleanup",
    ])
    t1 = time.time()

    npz_files = sorted(out_dir.rglob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"DA3 produced no .npz in {out_dir}")

    npz = npz_files[0]
    peak = monitor.peak_mb(t0, t1)
    print(f"\n  Output: {npz}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")
    return npz, t0, t1


# ─── Stage 2: SAM2 ────────────────────────────────────────────────────────────

def stage_sam2(image_path: Path, name: str, monitor: VRAMMonitor) -> tuple[Path, Path, float, float]:
    header("STAGE 2 — SAM2 segmentation")
    outputs_dir = (PROJECT_ROOT / "data" / "outputs").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{image_path.resolve().parent}:/input:ro",
        "-v", f"{outputs_dir}:/output",
        "-v", f"{MODELS_SAM2}:/opt/sam2/checkpoints:ro",
        "-v", f"{PROJECT_ROOT / 'sam2' / 'pipeline.py'}:/opt/sam2/pipeline.py:ro",
        "--entrypoint", "python3",
        "sam2:x86",
        "/opt/sam2/pipeline.py",
        "--input",  f"/input/{image_path.name}",
        "--output", "/output",
        "--name",   name,
    ]

    t0 = time.time()
    run(cmd)
    t1 = time.time()

    color_png   = outputs_dir / f"{name}.png"
    segmask_npy = outputs_dir / f"{name}_segmask.npy"
    for p in (color_png, segmask_npy):
        if not p.exists():
            raise RuntimeError(f"SAM2 did not produce {p}")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  Color:  {color_png}")
    print(f"  Mask:   {segmask_npy}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")
    return color_png, segmask_npy, t0, t1


# ─── Stage 3: Convert + mask depth ────────────────────────────────────────────

def stage_convert(
    npz_path: Path,
    segmask_npy: Path,
    original_image: Path,
    work_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Build numcc input directory. Files written:
      depth_full.npy   — full (unmasked) depth, float32 meters, for seen_xyz in NU-MCC
      mask.npy         — SAM2 binary mask resized to depth resolution, uint8
      intrinsics.json  — {fx, fy, cx, cy} from DA3 camera decoder
      <stem>_masked.png — color image with background → white (for numcc --color)

    The mask is passed separately so numcc can:
      1. Back-project the FULL depth to get the scene point cloud
      2. Filter by mask to isolate the object point cloud
      3. Use the FULL depth for seen_xyz (denser scene context for NU-MCC)
    """
    header("STAGE 3 — Prepare numcc inputs (full depth + SAM2 mask)")
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── depth: use DA3 metric output directly, DO NOT renormalize ────────────
    # DA3NESTED-GIANT-LARGE outputs metric depth in meters.  Re-scaling to an
    # arbitrary [0.1, 1.5] range destroys the real Z/XY aspect ratio and makes
    # the point cloud appear stretched along the camera axis when viewed from
    # the side.  Only clip extreme outliers (sensor noise, sky, etc.).
    data = np.load(str(npz_path), allow_pickle=True)
    print(f"  NPZ keys: {list(data.keys())}")

    depth = data["depth"].astype(np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    print(f"  Raw depth (metric): shape={depth.shape}  range=[{depth.min():.4f}, {depth.max():.4f}] m")

    depth_full = np.clip(depth, 0.05, 20.0)
    valid_px = int((depth_full > 0).sum())
    print(f"  Clipped:   [{depth_full.min():.3f}, {depth_full.max():.3f}] m  "
          f"valid_px={valid_px}/{depth_full.size}")

    depth_out = work_dir / "depth_full.npy"
    np.save(str(depth_out), depth_full)

    # ── SAM2 mask: resize to depth resolution and save ────────────────────────
    from PIL import Image as PILImage
    mask_orig = np.load(str(segmask_npy)).astype(bool)
    H, W = depth.shape
    print(f"  SAM2 mask: {mask_orig.shape} ({mask_orig.sum()} object px)  "
          f"→ resizing to {H}×{W}")
    m_img = PILImage.fromarray(mask_orig.astype(np.uint8) * 255).resize(
        (W, H), PILImage.NEAREST
    )
    mask_depth_res = (np.asarray(m_img) > 0).astype(np.uint8)
    print(f"  Mask at depth res: {mask_depth_res.sum()} object px / {H*W} total")

    mask_out = work_dir / "mask.npy"
    np.save(str(mask_out), mask_depth_res)

    # ── intrinsics ────────────────────────────────────────────────────────────
    if "intrinsics" in data:
        K = data["intrinsics"]
        if K.ndim == 3:
            K = K[0]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        print(f"  DA3 intrinsics: fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    else:
        fx = fy = W / (2 * np.tan(np.radians(60) / 2))
        cx, cy = W / 2.0, H / 2.0
        print(f"  Fallback (60° HFOV): fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

    intri_out = work_dir / "intrinsics.json"
    intri_out.write_text(json.dumps({"fx": fx, "fy": fy, "cx": cx, "cy": cy}, indent=2))

    # ── color: original image + white background outside object mask ──────────
    img = np.array(PILImage.open(str(original_image)).convert("RGB"))
    img_H, img_W = img.shape[:2]
    color_mask_img = PILImage.fromarray(mask_orig.astype(np.uint8) * 255).resize(
        (img_W, img_H), PILImage.NEAREST
    )
    color_mask = np.asarray(color_mask_img) > 0
    color_masked = img.copy()
    color_masked[~color_mask] = 255  # white background outside object
    color_out = work_dir / (original_image.stem + "_masked.png")
    PILImage.fromarray(color_masked).save(str(color_out))

    # ── depth visualization PNG ───────────────────────────────────────────────
    vis_out = work_dir / "depth_vis.png"
    _save_depth_vis(depth_full, mask_depth_res.astype(bool),
                    original_image, vis_out)
    print(f"  Depth vis: {vis_out.name}")

    print(f"  Saved: {depth_out.name}, {mask_out.name}, {intri_out.name}, {color_out.name}  → {work_dir}")
    return depth_out, mask_out, color_out, intri_out


def _save_depth_vis(depth: np.ndarray, mask: np.ndarray,
                    original_image: Path, out_path: Path):
    """
    Save a side-by-side depth visualization PNG:
      [original image]  |  [full depth colorized]  |  [object depth colorized]
    """
    from PIL import Image as PILImage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    H, W = depth.shape

    # Colorize full depth (turbo colormap, near=blue, far=red)
    d_min, d_max = float(depth.min()), float(depth.max())
    d_norm = (depth - d_min) / max(d_max - d_min, 1e-6)
    depth_rgb = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)

    # Colorize object-only depth (same scale)
    depth_obj = depth * mask.astype(np.float32)
    depth_obj_rgb = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_obj_rgb[~mask] = 20  # dark for background

    # Resize original to same H×W for alignment
    orig = PILImage.open(str(original_image)).convert("RGB").resize((W, H), PILImage.LANCZOS)

    panel = PILImage.new("RGB", (W * 3, H))
    panel.paste(orig, (0, 0))
    panel.paste(PILImage.fromarray(depth_rgb), (W, 0))
    panel.paste(PILImage.fromarray(depth_obj_rgb), (W * 2, 0))
    panel.save(str(out_path))


# ─── Stage 4: numcc ───────────────────────────────────────────────────────────

def stage_numcc(
    depth_npy: Path,
    mask_npy: Path,
    color_path: Path,
    intrinsics_json: Path,
    name: str,
    monitor: VRAMMonitor,
    udf_threshold: float = 0.23,
) -> tuple[Path, float, float]:
    header("STAGE 4 — numcc  (NU-MCC, no P2C)")

    input_dir = depth_npy.parent.resolve()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Mount local pipeline.py so fixes apply without image rebuild
    numcc_pipeline = PROJECT_ROOT / "numcc" / "pipeline.py"

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{ASSETS_DIR}:/output",
        "-v", f"{MODELS_NUMCC}:/opt/models:ro",
        "-v", f"{numcc_pipeline}:/app/pipeline.py:ro",
        "numcc:x86",
        "--depth",      f"/input/{depth_npy.name}",
        "--mask",       f"/input/{mask_npy.name}",
        "--color",      f"/input/{color_path.name}",
        "--intrinsics", f"/input/{intrinsics_json.name}",
        "--output", "/output",
        "--name",   name,
        "--no-p2c",
        "--udf-threshold", str(udf_threshold),
    ]

    t0 = time.time()
    run(cmd)
    t1 = time.time()

    sdf_path = ASSETS_DIR / name / f"{name}.sdf"
    if not sdf_path.exists():
        raise RuntimeError(f"Expected SDF not found: {sdf_path}")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  SDF: {sdf_path}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")
    return sdf_path, t0, t1


# ─── Stage 5: Drake ───────────────────────────────────────────────────────────

_DRAKE_VALIDATE = """\
import sys
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser
builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
Parser(plant).AddModels(sys.argv[1])
plant.Finalize()
print(f"  Drake SDF OK: {sys.argv[1]}")
print(f"  Bodies: {plant.num_bodies()}  Frames: {plant.num_frames()}")
"""

_DRAKE_MESHCAT = """\
import sys, time
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder,
    MeshcatVisualizer, Parser, Simulator, StartMeshcat,
)
sdf = sys.argv[1]
print(f"Loading SDF: {sdf}")
meshcat = StartMeshcat()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
Parser(plant).AddModels(sdf)
plant.Finalize()
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
diagram = builder.Build()
sim = Simulator(diagram)
sim.Initialize()
sim.AdvanceTo(0.01)
print(f"\\n  Meshcat URL: {meshcat.web_url()}")
print("  Open the URL above in a browser.  Ctrl+C to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
"""


def stage_drake(sdf_path: Path, monitor: VRAMMonitor, interactive: bool) -> tuple[float, float]:
    header("STAGE 5 — Drake  " + ("(Meshcat)" if interactive else "(SDF validation)"))

    script_code = _DRAKE_MESHCAT if interactive else _DRAKE_VALIDATE

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
        f.write(script_code)
        script = Path(f.name)

    t0 = time.time()
    try:
        run(["uv", "run", "python3", str(script), str(sdf_path)], cwd=PROJECT_ROOT)
    finally:
        script.unlink(missing_ok=True)
    t1 = time.time()

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")
    return t0, t1


# ─── VRAM report ──────────────────────────────────────────────────────────────

def print_report(monitor: VRAMMonitor, baseline_mb: int, stages: dict[str, tuple[float, float]], work_dir: Path):
    print("\n" + "="*60)
    print("  VRAM USAGE REPORT")
    print("="*60)
    print(f"  {'Baseline':<12}  {'':>9}  {'':>9}  {'':>6}")
    print(f"  {'Stage':<12}  {'Peak (MB)':>9}  {'Delta (MB)':>10}  {'Time (s)':>8}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*10}  {'-'*8}")
    print(f"  {'baseline':<12}  {baseline_mb:>9}  {'—':>10}  {'—':>8}")
    for stage, (t0, t1) in stages.items():
        peak  = monitor.peak_mb(t0, t1)
        delta = peak - baseline_mb
        dur   = t1 - t0
        print(f"  {stage:<12}  {peak:>9}  {delta:>+10}  {dur:>8.1f}")
    if stages:
        total = max(t1 for _, t1 in stages.values()) - min(t0 for t0, _ in stages.values())
        print(f"\n  Total wall time: {total:.1f}s")
    print("="*60)

    csv_path = work_dir / "vram_profile.csv"
    monitor.save_csv(csv_path)
    print(f"  Profile: {csv_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="DA3 → SAM2 → numcc → Drake  (full pipeline + VRAM measurement)"
    )
    ap.add_argument("image", type=Path, help="Input image (e.g. data/images/taza.jpeg)")
    ap.add_argument("--name", required=True, help="Asset name (used for all outputs)")
    ap.add_argument("--skip-da3",        action="store_true", help="Reuse existing DA3 npz")
    ap.add_argument("--skip-sam2",       action="store_true", help="Reuse existing SAM2 outputs")
    ap.add_argument("--skip-numcc",      action="store_true", help="Reuse existing numcc SDF")
    ap.add_argument("--skip-drake",      action="store_true", help="Skip Drake stage entirely")
    ap.add_argument("--drake-interactive", action="store_true",
                    help="Open Meshcat visualizer (blocks until Ctrl+C)")
    ap.add_argument("--udf-threshold", type=float, default=0.23,
                    help="NU-MCC UDF threshold for surface extraction (default: 0.23, "
                         "matches CO3D-V2 training; lower values keep far fewer surface points)")
    args = ap.parse_args()

    image_path = args.image.resolve()
    if not image_path.exists():
        sys.exit(f"Error: image not found: {image_path}")

    work_dir    = PROJECT_ROOT / "data" / "outputs" / "pipeline" / args.name
    da3_out_dir = work_dir / "da3_raw"
    numcc_in    = work_dir / "numcc_input"

    monitor = VRAMMonitor(poll_ms=200)
    monitor.start()
    baseline_mb = monitor.current_mb()
    print(f"\nBaseline VRAM: {baseline_mb} MB")

    stages: dict[str, tuple[float, float]] = {}

    try:
        # Stage 1: DA3
        if not args.skip_da3:
            npz_path, t0, t1 = stage_da3(image_path, da3_out_dir, monitor)
            stages["DA3"] = (t0, t1)
        else:
            npz_files = sorted(da3_out_dir.rglob("*.npz"))
            if not npz_files:
                sys.exit(f"--skip-da3: no .npz found under {da3_out_dir}")
            npz_path = npz_files[0]
            print(f"\n[--skip-da3] Using {npz_path}")

        # Stage 2: SAM2
        if not args.skip_sam2:
            _color_png, segmask_npy, t0, t1 = stage_sam2(image_path, args.name, monitor)
            stages["SAM2"] = (t0, t1)
        else:
            outputs_dir = PROJECT_ROOT / "data" / "outputs"
            segmask_npy = outputs_dir / f"{args.name}_segmask.npy"
            if not segmask_npy.exists():
                sys.exit(f"--skip-sam2: segmask not found: {segmask_npy}")
            print(f"\n[--skip-sam2] Using {segmask_npy}")

        # Stage 3: convert
        depth_npy, mask_npy, color_path, intrinsics_json = stage_convert(
            npz_path, segmask_npy, image_path, numcc_in
        )

        # Stage 4: numcc
        if not args.skip_numcc:
            sdf_path, t0, t1 = stage_numcc(
                depth_npy, mask_npy, color_path, intrinsics_json, args.name, monitor,
                udf_threshold=args.udf_threshold,
            )
            stages["numcc"] = (t0, t1)
        else:
            sdf_path = ASSETS_DIR / args.name / f"{args.name}.sdf"
            if not sdf_path.exists():
                sys.exit(f"--skip-numcc: SDF not found: {sdf_path}")
            print(f"\n[--skip-numcc] Using {sdf_path}")

        # Stage 5: Drake
        if not args.skip_drake:
            t0, t1 = stage_drake(sdf_path, monitor, interactive=args.drake_interactive)
            stages["Drake"] = (t0, t1)

    except subprocess.CalledProcessError as e:
        print(f"\n\nPipeline failed (exit code {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)
    finally:
        monitor.stop()

    print_report(monitor, baseline_mb, stages, work_dir)


if __name__ == "__main__":
    main()
