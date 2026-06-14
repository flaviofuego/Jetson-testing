#!/usr/bin/env python3
"""
Pipeline unificado: imagen (+ opcionales) → carpeta organizada por stage.

INPUTS PRIMARIOS
  image               Imagen RGB (.jpg, .png)
  --depth PATH        Depth map .npy (float32 m) ó .png (uint16 mm)  → salta DA3
  --mask  PATH        Máscara SAM2 .npy (uint8, 1=objeto)            → salta SAM2
  --intrinsics PATH   JSON {fx, fy, cx, cy}                          → salta estimación

ESTRUCTURA DE SALIDA
  <output-dir>/<name>/
  ├── 01_da3/                  NPZ + depth_full.npy + intrinsics.json + depth_vis.png
  ├── 02_sam2/                 segmask.npy + imagen fondo blanco + viz.png
  ├── 03_numcc_inputs/         depth_full.npy + mask.npy + intrinsics.json + *_masked.png
  ├── 04_pointcloud/           *_object_cloud.ply + *_numcc_surface.ply
  ├── 05_mesh/                 <name>.obj + <name>_smoothed.obj
  ├── 06_debug/                (sólo con --debug) dump NU-MCC inputs/outputs
  └── pipeline_report.json     tiempos, VRAM, parámetros, inventario de archivos

USOS
  # Desde imagen (DA3 → SAM2 → numcc → smooth)
  .venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto

  # Proveer depth (salta DA3)
  .venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \\
      --depth data/outputs/depth.npy

  # Proveer depth + máscara (salta DA3 y SAM2)
  .venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \\
      --depth data/outputs/depth.npy --mask data/outputs/mask.npy

  # RGBD real (Drake/AIRA) — depth + color + intrínsecas conocidas
  .venv/bin/python3 tools/pipeline.py data/aira_input_data/color.png --name aira_obj \\
      --depth data/aira_input_data/depth_image.npy \\
      --mask  data/aira_input_data/mask.npy \\
      --intrinsics data/aira_input_data/intrinsics.json

  # Reanudar saltando stages ya calculados
  .venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto \\
      --skip-da3 --skip-sam2

  # Con debug dump NU-MCC
  .venv/bin/python3 tools/pipeline.py data/images/objeto.jpg --name objeto --debug
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DA3_CLI      = Path("/home/worker-node-4/Documents/GitHub/UniWhere/.venv/bin/da3")
MODELS_NUMCC = Path.home() / "models" / "numcc"
MODELS_SAM2  = Path.home() / "models" / "sam2"
NKSR_CACHE   = Path.home() / "models" / "nksr_cache"
DOCKER_NUMCC = "numcc:x86"
DOCKER_SAM2  = "sam2:x86"


# ─── VRAM Monitor ─────────────────────────────────────────────────────────────

class VRAMMonitor:
    def __init__(self, poll_ms: int = 200):
        self._poll_ms = poll_ms
        self._data: list[tuple[float, int]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

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

    def peak_mb(self, t0: float, t1: float) -> int:
        with self._lock:
            vals = [mb for t, mb in self._data if t0 <= t <= t1]
        return max(vals) if vals else self.current_mb()

    def save_csv(self, path: Path):
        with self._lock:
            rows = list(self._data)
        with open(path, "w") as f:
            f.write("timestamp_s,used_mb\n")
            for t, mb in rows:
                f.write(f"{t:.3f},{mb}\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


def run(cmd: list) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True)


def _save_depth_vis(depth: np.ndarray, mask_bool: np.ndarray | None,
                    original_image: Path, out_path: Path):
    """Three-panel depth visualization: original | depth | object depth."""
    try:
        from PIL import Image as PILImage
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        H, W = depth.shape
        d_min, d_max = float(depth.min()), float(depth.max())
        d_norm = (depth - d_min) / max(d_max - d_min, 1e-6)
        depth_rgb = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)

        obj_rgb = depth_rgb.copy()
        if mask_bool is not None and mask_bool.shape == (H, W):
            obj_rgb[~mask_bool] = 20

        orig = PILImage.open(str(original_image)).convert("RGB").resize((W, H), PILImage.LANCZOS)
        panel = PILImage.new("RGB", (W * 3, H))
        panel.paste(orig, (0, 0))
        panel.paste(PILImage.fromarray(depth_rgb), (W, 0))
        panel.paste(PILImage.fromarray(obj_rgb), (W * 2, 0))
        panel.save(str(out_path))
    except Exception as e:
        print(f"      [depth_vis] skipped ({e})")


# ─── Stage 1: DA3 ─────────────────────────────────────────────────────────────

def stage_da3(image: Path, out_dir: Path, monitor: VRAMMonitor) -> dict:
    """
    Runs DA3 on image → out_dir (01_da3/).
    Returns dict with depth_npy, intrinsics_json, intrinsics_dict, t0, t1.
    """
    header("STAGE 1 — Depth Anything 3")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DA3_CLI.exists():
        sys.exit(
            f"DA3 CLI not found at {DA3_CLI}\n"
            "Activate the UniWhere venv first:\n"
            "  source /home/worker-node-4/Documents/GitHub/UniWhere/.venv/bin/activate"
        )

    t0 = time.time()
    run([str(DA3_CLI), "image", str(image),
         "--export-format", "npz",
         "--export-dir", str(out_dir),
         "--auto-cleanup"])
    t1 = time.time()

    npz_files = sorted(out_dir.rglob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"DA3 produced no .npz in {out_dir}")
    npz_path = npz_files[0]

    data = np.load(str(npz_path), allow_pickle=True)
    print(f"  NPZ keys: {list(data.keys())}")

    depth = data["depth"].astype(np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    print(f"  Raw depth (metric): shape={depth.shape}  range=[{depth.min():.4f}, {depth.max():.4f}] m")

    depth_clipped = np.clip(depth, 0.05, 20.0)
    depth_npy = out_dir / "depth_full.npy"
    np.save(str(depth_npy), depth_clipped)

    if "intrinsics" in data:
        K = data["intrinsics"]
        if K.ndim == 3:
            K = K[0]
        intri = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                 "cx": float(K[0, 2]), "cy": float(K[1, 2])}
        print(f"  DA3 intrinsics: fx={intri['fx']:.1f} fy={intri['fy']:.1f} "
              f"cx={intri['cx']:.1f} cy={intri['cy']:.1f}")
    else:
        H, W = depth.shape
        fx = W / (2 * np.tan(np.radians(60) / 2))
        intri = {"fx": fx, "fy": fx, "cx": float(W / 2), "cy": float(H / 2)}
        print(f"  Fallback intrinsics (60° HFOV): {intri}")

    intri_json = out_dir / "intrinsics.json"
    intri_json.write_text(json.dumps(intri, indent=2))

    _save_depth_vis(depth_clipped, None, image, out_dir / "depth_vis.png")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  depth_full.npy → {depth_npy}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")

    return {"depth_npy": depth_npy, "intrinsics_json": intri_json,
            "intrinsics": intri, "t0": t0, "t1": t1, "peak_mb": peak}


# ─── Stage 2: SAM2 ────────────────────────────────────────────────────────────

def stage_sam2(image: Path, name: str, out_dir: Path, monitor: VRAMMonitor,
               sam2_model: str = "small",
               pred_iou_thresh: float = 0.88,
               stability_score_thresh: float = 0.95) -> dict:
    """
    Runs SAM2 on image → out_dir (02_sam2/).
    Returns dict with segmask_npy, color_png, t0, t1.
    """
    header("STAGE 2 — SAM2 segmentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{image.resolve().parent}:/input:ro",
        "-v", f"{out_dir.resolve()}:/output",
        "-v", f"{MODELS_SAM2}:/opt/sam2/checkpoints:ro",
        "-v", f"{PROJECT_ROOT / 'submodules' / 'sam2' / 'pipeline.py'}:/opt/sam2/pipeline.py:ro",
        "--entrypoint", "python3",
        DOCKER_SAM2,
        "/opt/sam2/pipeline.py",
        "--input",  f"/input/{image.name}",
        "--output", "/output",
        "--name",   name,
        "--model",  sam2_model,
        "--pred-iou-thresh",         str(pred_iou_thresh),
        "--stability-score-thresh",  str(stability_score_thresh),
    ]

    t0 = time.time()
    run(cmd)
    t1 = time.time()

    segmask_npy = out_dir / f"{name}_segmask.npy"
    color_png   = out_dir / f"{name}.png"
    for p in (segmask_npy, color_png):
        if not p.exists():
            raise RuntimeError(f"SAM2 did not produce {p}")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  Segmask: {segmask_npy}")
    print(f"  Color:   {color_png}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")

    return {"segmask_npy": segmask_npy, "color_png": color_png,
            "t0": t0, "t1": t1, "peak_mb": peak}


# ─── Stage 3: Prepare numcc inputs ────────────────────────────────────────────

def stage_prepare(
    depth_npy: Path,
    segmask_npy: Path,
    original_image: Path,
    intrinsics_json: Path,
    out_dir: Path,
) -> dict:
    """
    Assembles numcc input directory (03_numcc_inputs/).
    All inputs must be ready (from DA3/SAM2 or user-provided).
    Returns dict with paths to prepared files.
    """
    header("STAGE 3 — Prepare numcc inputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image as PILImage

    # ── depth: clip and copy ──────────────────────────────────────────────────
    depth = np.load(str(depth_npy)).astype(np.float32)
    # Handle uint16 mm (e.g. from --depth .png loaded as npy by caller)
    if depth.max() > 100:
        print(f"  WARNING: depth max={depth.max():.1f} (>100) — assuming mm, converting to m")
        depth = depth / 1000.0
    depth = np.clip(depth, 0.05, 20.0)
    H, W = depth.shape
    depth_out = out_dir / "depth_full.npy"
    np.save(str(depth_out), depth)
    print(f"  depth:  {H}×{W}  [{depth.min():.3f},{depth.max():.3f}] m  → {depth_out.name}")

    # ── mask: resize to depth resolution ─────────────────────────────────────
    mask_orig = np.load(str(segmask_npy)).astype(bool)
    m_img = PILImage.fromarray(mask_orig.astype(np.uint8) * 255).resize(
        (W, H), PILImage.NEAREST
    )
    mask = (np.asarray(m_img) > 0).astype(np.uint8)
    mask_out = out_dir / "mask.npy"
    np.save(str(mask_out), mask)
    print(f"  mask:   orig={mask_orig.shape} obj_px={mask_orig.sum()} "
          f"→ depth_res={mask.shape} obj_px={mask.sum()}  → {mask_out.name}")

    # ── intrinsics: copy ──────────────────────────────────────────────────────
    intri_out = out_dir / "intrinsics.json"
    shutil.copy2(intrinsics_json, intri_out)
    intri = json.loads(intri_out.read_text())
    print(f"  intri:  fx={intri['fx']:.1f} fy={intri['fy']:.1f} "
          f"cx={intri['cx']:.1f} cy={intri['cy']:.1f}")

    # ── color masked: white background outside object ─────────────────────────
    img = np.array(PILImage.open(str(original_image)).convert("RGB"))
    img_H, img_W = img.shape[:2]
    color_mask = np.asarray(
        PILImage.fromarray(mask_orig.astype(np.uint8) * 255).resize(
            (img_W, img_H), PILImage.NEAREST
        )
    ) > 0
    color_masked = img.copy()
    color_masked[~color_mask] = 255
    color_out = out_dir / f"{original_image.stem}_masked.png"
    PILImage.fromarray(color_masked).save(str(color_out))
    print(f"  color:  {img_H}×{img_W}  fondo blanco  → {color_out.name}")

    # ── depth visualization ───────────────────────────────────────────────────
    _save_depth_vis(depth, mask.astype(bool), original_image, out_dir / "depth_vis.png")
    print(f"  depth_vis.png generado")

    return {
        "depth_npy":  depth_out,
        "mask_npy":   mask_out,
        "intri_json": intri_out,
        "color_png":  color_out,
    }


# ─── Stage 4: numcc ───────────────────────────────────────────────────────────

def stage_numcc(
    depth_npy: Path,
    mask_npy: Path,
    color_path: Path,
    intrinsics_json: Path,
    name: str,
    cloud_dir: Path,
    mesh_dir: Path,
    debug_dir: Path | None,
    monitor: VRAMMonitor,
    mesh_method: str = "noksr",
    udf_threshold: float = 0.23,
    udf_n_iter: int = 10,
    n_query: int = 200_000,
    poisson_depth: int = 10,
    no_p2c: bool = True,
    no_floor_cap: bool = False,
    no_sdf: bool = True,
) -> dict:
    """
    Runs numcc Docker container.
    PLY point clouds  → 04_pointcloud/
    OBJ + SDF + parts → 05_mesh/
    Debug dump        → 06_debug/  (if debug_dir set)
    """
    header("STAGE 4 — numcc  (NU-MCC surface reconstruction + mesh)")

    input_dir = depth_npy.parent.resolve()
    NKSR_CACHE.mkdir(parents=True, exist_ok=True)
    cloud_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # numcc outputs to /output/<name>/ inside Docker.
    # Fresh temp dir each run — avoids stale root-owned dirs from prior runs.
    tmp_root = Path(tempfile.mkdtemp(prefix=f"numcc_{name}_", dir=mesh_dir.parent))

    numcc_pipeline = PROJECT_ROOT / "models" / "numcc" / "pipeline.py"
    if not numcc_pipeline.exists():
        sys.exit(f"models/numcc/pipeline.py not found at {numcc_pipeline}")

    # Build color mount: color_path may be in input_dir already, or separate
    color_resolved = color_path.resolve()
    if color_resolved.parent == input_dir:
        color_mount = []
        color_arg   = f"/input/{color_path.name}"
    else:
        color_mount = ["-v", f"{color_resolved.parent}:/color_dir:ro"]
        color_arg   = f"/color_dir/{color_path.name}"

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "TORCH_HOME=/nksr_cache",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{tmp_root.resolve()}:/output",
        "-v", f"{MODELS_NUMCC}:/opt/models:ro",
        "-v", f"{NKSR_CACHE}:/nksr_cache",
        "-v", f"{numcc_pipeline}:/app/pipeline.py:ro",
        *color_mount,
        DOCKER_NUMCC,
        "--depth",          f"/input/{depth_npy.name}",
        "--color",          color_arg,
        "--mask",           f"/input/{mask_npy.name}",
        "--intrinsics",     f"/input/{intrinsics_json.name}",
        "--output",         "/output",
        "--name",           name,
        "--mesh-method",    mesh_method,
        "--udf-threshold",  str(udf_threshold),
        "--udf-n-iter",     str(udf_n_iter),
        "--n-query",        str(n_query),
        "--poisson-depth",  str(poisson_depth),
    ]
    if no_p2c:
        cmd.append("--no-p2c")
    if no_floor_cap:
        cmd.append("--no-floor-cap")
    if no_sdf:
        cmd.append("--no-sdf")
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cmd += ["--debug-dump", "/output/_debug"]

    t0 = time.time()
    run(cmd)
    t1 = time.time()

    # ── Sort outputs ──────────────────────────────────────────────────────────
    asset_tmp = tmp_root / name

    if not asset_tmp.exists():
        raise RuntimeError(
            f"numcc did not produce output in {asset_tmp}. Check Docker logs."
        )

    # 04_pointcloud: PLY + NPY clouds
    cloud_patterns = ["*_object_cloud.ply", "*_numcc_surface.ply", "*_pointcloud.npy"]
    moved_clouds = []
    for pat in cloud_patterns:
        for src in asset_tmp.glob(pat):
            dst = cloud_dir / src.name
            shutil.move(str(src), str(dst))
            moved_clouds.append(dst)
            print(f"  → 04_pointcloud/{src.name}")

    # 05_mesh: OBJ, SDF, parts dir (everything left in asset_tmp except debug)
    sdf_path = None
    obj_path = None
    moved_mesh = []
    for src in sorted(asset_tmp.iterdir()):
        dst = mesh_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))
            shutil.rmtree(str(src))
        else:
            shutil.move(str(src), str(dst))
        moved_mesh.append(dst)
        if dst.suffix == ".sdf":
            sdf_path = dst
        if dst.suffix == ".obj" and dst.stem == name:
            obj_path = dst
        print(f"  → 05_mesh/{src.name}")

    # 06_debug: move from /output/_debug → debug_dir
    debug_tmp = tmp_root / "_debug"
    if debug_dir is not None and debug_tmp.exists():
        if debug_dir.exists():
            shutil.rmtree(str(debug_dir))
        shutil.move(str(debug_tmp), str(debug_dir))
        print(f"  → 06_debug/ ({len(list(debug_dir.iterdir()))} archivos)")

    shutil.rmtree(str(tmp_root), ignore_errors=True)

    if obj_path is None or not obj_path.exists():
        raise RuntimeError(f"OBJ not generated. Expected: {mesh_dir}/{name}.obj")
    if not no_sdf and (sdf_path is None or not sdf_path.exists()):
        raise RuntimeError(f"SDF not generated. Expected: {mesh_dir}/{name}.sdf")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  OBJ: {obj_path}")
    if sdf_path and sdf_path.exists():
        print(f"  SDF: {sdf_path}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")

    return {
        "obj_path":  obj_path,
        "sdf_path":  sdf_path,
        "clouds":    moved_clouds,
        "t0": t0, "t1": t1, "peak_mb": peak,
    }


# ─── Stage 5: Smooth ──────────────────────────────────────────────────────────

def stage_smooth(obj_path: Path, mesh_dir: Path, name: str, monitor: VRAMMonitor) -> dict:
    """
    Runs smooth_preserve.py on the numcc OBJ.
    Accepts any PyMeshLab-supported format (OBJ, PLY, STL) — detected by extension.
    Output: <mesh_dir>/<name>_smoothed.obj
    """
    header("STAGE 5 — Smoothing  (Clustering → Taubin → Two Steps → Normal Smooth)")

    smooth_script = PROJECT_ROOT / "tools" / "smooth_preserve.py"
    out_obj = mesh_dir / f"{name}_smoothed.obj"

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        sys.exit(f"Venv python not found: {venv_python}")

    cmd = [str(venv_python), str(smooth_script), str(obj_path), str(out_obj)]

    t0 = time.time()
    run(cmd)
    t1 = time.time()

    if not out_obj.exists():
        raise RuntimeError(f"Smoothing did not produce {out_obj}")

    peak = monitor.peak_mb(t0, t1)
    print(f"\n  Smoothed OBJ: {out_obj}")
    print(f"  Time: {t1-t0:.1f}s  |  Peak VRAM: {peak} MB")

    return {"smoothed_obj": out_obj, "t0": t0, "t1": t1, "peak_mb": peak}


# ─── Report ───────────────────────────────────────────────────────────────────

def build_report(
    name: str,
    out_root: Path,
    args: argparse.Namespace,
    baseline_mb: int,
    stages: dict[str, dict],
    monitor: VRAMMonitor,
) -> dict:
    """Build and save pipeline_report.json."""

    # File inventory
    inventory = {}
    for stage_dir in sorted(out_root.iterdir()):
        if not stage_dir.is_dir():
            continue
        files = []
        for f in sorted(stage_dir.rglob("*")):
            if f.is_file():
                files.append({
                    "path": str(f.relative_to(out_root)),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
        inventory[stage_dir.name] = files

    stage_summary = {}
    for sname, info in stages.items():
        stage_summary[sname] = {
            "duration_s": round(info["t1"] - info["t0"], 2),
            "peak_vram_mb": info.get("peak_mb", 0),
        }

    total_s = None
    real = {k: v for k, v in stages.items() if v["t0"] > 0 or v["t1"] > 0}
    if real:
        t_all_start = min(v["t0"] for v in real.values())
        t_all_end   = max(v["t1"] for v in real.values())
        total_s = round(t_all_end - t_all_start, 2)

    report = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": {
            "image":         str(args.image),
            "depth_input":   str(args.depth) if args.depth else "DA3",
            "mask_input":    str(args.mask) if args.mask else "SAM2",
            "intrinsics":    str(args.intrinsics) if args.intrinsics else "DA3/fallback",
            "mesh_method":   args.mesh_method,
            "udf_threshold": args.udf_threshold,
            "udf_n_iter":    args.udf_n_iter,
            "n_query":       args.n_query,
            "poisson_depth": args.poisson_depth,
            "no_p2c":        args.no_p2c,
            "no_floor_cap":  args.no_floor_cap,
            "skip_smooth":   args.skip_smooth,
            "debug":         args.debug,
        },
        "vram": {
            "baseline_mb": baseline_mb,
            "stages":      stage_summary,
        },
        "total_duration_s": total_s,
        "output_root": str(out_root),
        "inventory": inventory,
    }

    report_path = out_root / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # CSV VRAM
    monitor.save_csv(out_root / "vram_profile.csv")

    return report


def print_report(report: dict, baseline_mb: int, stages: dict):
    print("\n" + "="*62)
    print("  PIPELINE REPORT")
    print("="*62)
    print(f"  {'Stage':<14}  {'Peak (MB)':>9}  {'Delta (MB)':>10}  {'Time (s)':>8}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*10}  {'-'*8}")
    print(f"  {'baseline':<14}  {baseline_mb:>9}  {'—':>10}  {'—':>8}")
    for sname, info in stages.items():
        peak  = info.get("peak_mb", 0)
        delta = peak - baseline_mb
        dur   = info["t1"] - info["t0"]
        print(f"  {sname:<14}  {peak:>9}  {delta:>+10}  {dur:>8.1f}")
    if report.get("total_duration_s"):
        print(f"\n  Total: {report['total_duration_s']}s")
    print(f"  Output: {report['output_root']}")
    print("="*62)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline unificado: imagen → OBJ + SDF Drake-ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── inputs primarios ──────────────────────────────────────────────────────
    ap.add_argument("image",         type=Path,
                    help="Imagen RGB de entrada (.jpg, .png)")
    ap.add_argument("--depth",       type=Path, default=None,
                    help="Depth map .npy (float32 m) ó .png (uint16 mm). "
                         "Si se provee, salta DA3.")
    ap.add_argument("--mask",        type=Path, default=None,
                    help="Máscara SAM2 .npy (uint8, 1=objeto). "
                         "Si se provee, salta SAM2.")
    ap.add_argument("--intrinsics",  type=Path, default=None,
                    help="JSON {fx, fy, cx, cy}. "
                         "Si no se provee, se usan intrínsecas de DA3 ó fallback 60° HFOV.")
    ap.add_argument("--name",        required=True,
                    help="Nombre del asset (determina subcarpeta de salida)")
    ap.add_argument("--output-dir",  type=Path, default=Path("output"),
                    help="Carpeta raíz de salida (default: ./output)")

    # ── control de stages ─────────────────────────────────────────────────────
    ap.add_argument("--skip-da3",    action="store_true",
                    help="Saltar DA3 (requiere --depth ó que 01_da3/depth_full.npy exista)")
    ap.add_argument("--skip-sam2",   action="store_true",
                    help="Saltar SAM2 (requiere --mask ó que 02_sam2/<name>_segmask.npy exista)")
    ap.add_argument("--skip-numcc",  action="store_true",
                    help="Saltar numcc (reutiliza mesh en 05_mesh/)")
    ap.add_argument("--skip-smooth", action="store_true",
                    help="Saltar smoothing (PyMeshLab)")

    # ── numcc params ──────────────────────────────────────────────────────────
    ap.add_argument("--mesh-method",    default="noksr",
                    choices=["poisson", "noksr", "both"],
                    help="Método de reconstrucción (default: noksr)")
    ap.add_argument("--udf-threshold",  type=float, default=0.23,
                    help="UDF threshold NU-MCC (default: 0.23)")
    ap.add_argument("--udf-n-iter",     type=int,   default=10,
                    help="Iteraciones move_points (default: 10)")
    ap.add_argument("--n-query",        type=int,   default=200_000,
                    help="Puntos de query UDF (default: 200000)")
    ap.add_argument("--poisson-depth",  type=int,   default=10,
                    help="Octree depth Poisson (default: 10)")
    ap.add_argument("--no-p2c",         action="store_true",
                    help="Saltar P2C completion (recomendado — el checkpoint actual "
                         "tiene mismatch de arquitectura)")
    ap.add_argument("--no-floor-cap",   action="store_true",
                    help="Desactivar tapa del plano de soporte")

    # ── sam2 params ───────────────────────────────────────────────────────────
    ap.add_argument("--sam2-model", default="small",
                    choices=["tiny", "small", "base_plus"],
                    help="Variante de modelo SAM2 (default: small)")
    ap.add_argument("--sam2-pred-iou-thresh", type=float, default=0.88,
                    help="SAM2 AMG predicted IoU threshold (default 0.88; 0.70 para estructuras delgadas como arcos)")
    ap.add_argument("--sam2-stability-thresh", type=float, default=0.95,
                    help="SAM2 AMG stability threshold (default 0.95; 0.80 para estructuras delgadas)")

    # ── extras ────────────────────────────────────────────────────────────────
    ap.add_argument("--debug",          action="store_true",
                    help="Guardar dump de NU-MCC (seen_xyz, anchors, UDF hist, etc.) "
                         "en 06_debug/")

    args = ap.parse_args()

    # ── validar imagen de entrada ─────────────────────────────────────────────
    image = args.image.resolve()
    if not image.exists():
        sys.exit(f"ERROR: imagen no encontrada: {image}")

    # ── preparar carpeta de salida ────────────────────────────────────────────
    out_root = args.output_dir.resolve() / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    dir_da3     = out_root / "01_da3"
    dir_sam2    = out_root / "02_sam2"
    dir_prepare = out_root / "03_numcc_inputs"
    dir_clouds  = out_root / "04_pointcloud"
    dir_mesh    = out_root / "05_mesh"
    dir_debug   = out_root / "06_debug" if args.debug else None

    print(f"\n  Asset:  {args.name}")
    print(f"  Output: {out_root}")

    monitor = VRAMMonitor(poll_ms=200)
    monitor.start()
    baseline_mb = monitor.current_mb()
    print(f"  Baseline VRAM: {baseline_mb} MB")

    stages: dict[str, dict] = {}

    try:
        # ── [1] DA3 ───────────────────────────────────────────────────────────
        if args.depth is not None:
            # User provided depth → skip DA3, but still need intrinsics
            depth_npy = args.depth.resolve()
            if args.intrinsics:
                intri_json = args.intrinsics.resolve()
            else:
                # Try to find intrinsics in 01_da3/ from previous run
                cached = dir_da3 / "intrinsics.json"
                if cached.exists():
                    intri_json = cached
                    print(f"\n[--depth] Usando intrínsecas de {intri_json}")
                else:
                    # Generate fallback from depth dimensions
                    dep = np.load(str(depth_npy)).astype(np.float32)
                    H, W = dep.shape if dep.ndim == 2 else dep.shape[1:]
                    fx = W / (2 * np.tan(np.radians(60) / 2))
                    fallback = {"fx": fx, "fy": fx, "cx": float(W/2), "cy": float(H/2)}
                    dir_da3.mkdir(parents=True, exist_ok=True)
                    intri_json = dir_da3 / "intrinsics.json"
                    intri_json.write_text(json.dumps(fallback, indent=2))
                    print(f"\n[--depth] Fallback intrínsecas 60° HFOV guardadas en {intri_json}")
            print(f"[--depth] Saltando DA3 — usando {depth_npy.name}")

        elif args.skip_da3:
            # Must find depth in 01_da3/
            depth_npy = dir_da3 / "depth_full.npy"
            intri_json = args.intrinsics.resolve() if args.intrinsics else dir_da3 / "intrinsics.json"
            for p in (depth_npy, intri_json):
                if not p.exists():
                    sys.exit(f"--skip-da3: archivo no encontrado: {p}")
            print(f"\n[--skip-da3] Usando {depth_npy}")

        else:
            result = stage_da3(image, dir_da3, monitor)
            stages["DA3"] = result
            depth_npy  = result["depth_npy"]
            intri_json = result["intrinsics_json"]

        # ── [2] SAM2 ──────────────────────────────────────────────────────────
        if args.mask is not None:
            segmask_npy = args.mask.resolve()
            print(f"\n[--mask] Saltando SAM2 — usando {segmask_npy.name}")

        elif args.skip_sam2:
            segmask_npy = dir_sam2 / f"{args.name}_segmask.npy"
            if not segmask_npy.exists():
                sys.exit(f"--skip-sam2: máscara no encontrada: {segmask_npy}")
            print(f"\n[--skip-sam2] Usando {segmask_npy}")

        else:
            result = stage_sam2(image, args.name, dir_sam2, monitor,
                                sam2_model=args.sam2_model,
                                pred_iou_thresh=args.sam2_pred_iou_thresh,
                                stability_score_thresh=args.sam2_stability_thresh)
            stages["SAM2"] = result
            segmask_npy = result["segmask_npy"]

        # ── [3+4] Prepare + numcc ─────────────────────────────────────────────
        if args.skip_numcc:
            obj_path = dir_mesh / f"{args.name}.obj"
            if not obj_path.exists():
                sys.exit(f"--skip-numcc: OBJ no encontrado: {obj_path}")
            print(f"\n[--skip-numcc] Saltando prepare + numcc — usando {obj_path}")
            stages["numcc"] = {"t0": 0, "t1": 0, "peak_mb": 0}  # placeholder

        else:
            prep = stage_prepare(depth_npy, segmask_npy, image, intri_json, dir_prepare)
            result = stage_numcc(
                depth_npy    = prep["depth_npy"],
                mask_npy     = prep["mask_npy"],
                color_path   = prep["color_png"],
                intrinsics_json = prep["intri_json"],
                name         = args.name,
                cloud_dir    = dir_clouds,
                mesh_dir     = dir_mesh,
                debug_dir    = dir_debug,
                monitor      = monitor,
                mesh_method  = args.mesh_method,
                udf_threshold= args.udf_threshold,
                udf_n_iter   = args.udf_n_iter,
                n_query      = args.n_query,
                poisson_depth= args.poisson_depth,
                no_p2c       = args.no_p2c,
                no_floor_cap = args.no_floor_cap,
                no_sdf       = True,
            )
            stages["numcc"] = result
            obj_path = result["obj_path"]

        # ── [5] Smooth ────────────────────────────────────────────────────────
        if not args.skip_smooth:
            result = stage_smooth(obj_path, dir_mesh, args.name, monitor)
            stages["smooth"] = result

    except subprocess.CalledProcessError as e:
        print(f"\n\nPipeline falló (exit code {e.returncode})", file=sys.stderr)
        monitor.stop()
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n\nPipeline falló: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        monitor.stop()
        sys.exit(1)
    finally:
        monitor.stop()

    # ── Reporte ───────────────────────────────────────────────────────────────
    report = build_report(args.name, out_root, args, baseline_mb, stages, monitor)
    print_report(report, baseline_mb, stages)

    # Resumen final de archivos
    print("\n  ARCHIVOS GENERADOS")
    print(f"  {'─'*50}")
    for stage_name, files in report["inventory"].items():
        if files:
            print(f"  {stage_name}/")
            for f in files:
                print(f"    {f['path'].split('/', 1)[-1]:<45} {f['size_kb']:>8.1f} KB")
    print(f"\n  pipeline_report.json  →  {out_root}/pipeline_report.json")
    print(f"  vram_profile.csv      →  {out_root}/vram_profile.csv\n")


if __name__ == "__main__":
    main()
