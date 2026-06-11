"""
Convierte los .npy de AIRA a PNG (imágenes) y PLY (nubes de puntos).

Uso:
    python3 tools/visualize_aira_npy.py [--input-dir aira] [--output-dir /tmp/aira_vis]

Salidas:
    color.png          — imagen RGB
    depth.png          — depth colorizado (inferno)
    depth_pnts.ply     — point cloud completa back-projected (307200 pts con color RGB)
    point_cloud.ply    — nube filtrada de Drake (480 pts, sin color)
"""
import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None):
    """Escribe un PLY ASCII con XYZ y opcionalmente RGB uint8."""
    n = len(xyz)
    has_color = rgb is not None

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            if has_color:
                r, g, b = rgb[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"  guardado: {path}  ({n} puntos)")


def depth_to_png(depth: np.ndarray, path: Path):
    valid = depth[np.isfinite(depth) & (depth < 9.9)]
    vmin, vmax = valid.min(), valid.max()
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-9), 0, 1)
    colored = (cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(path)
    print(f"  guardado: {path}  (rango depth: {vmin:.3f}–{vmax:.3f} m)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="aira",
                        help="Directorio con los .npy (default: aira/)")
    parser.add_argument("--output-dir", default="data/aira_input_data",
                        help="Directorio de salida (default: data/aira_input_data)")
    args = parser.parse_args()

    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # --- color image ---
    color_path = src / "color_image.npy"
    if color_path.exists():
        rgba = np.load(color_path)           # (480, 640, 4) uint8
        rgb  = rgba[:, :, :3]
        Image.fromarray(rgb).save(dst / "color.png")
        print(f"  guardado: {dst / 'color.png'}  {rgb.shape}")
    else:
        print(f"  [skip] {color_path} no encontrado")
        rgb = None

    # --- depth image ---
    depth_path = src / "depth_image.npy"
    if depth_path.exists():
        depth = np.load(depth_path)          # (480, 640) float32, metros
        depth_to_png(depth, dst / "depth.png")
    else:
        print(f"  [skip] {depth_path} no encontrado")
        depth = None

    # --- depth_pnts → PLY con color ---
    pnts_path = src / "depth_pnts.npy"
    if pnts_path.exists():
        pC = np.load(pnts_path).astype(np.float32)   # (307200, 3) XYZ cámara
        # filtrar puntos a >9.9 m (fondo clipeado)
        mask_valid = pC[:, 2] < 9.9
        pC_filt = pC[mask_valid]

        # color por pixel si tenemos la imagen
        if rgb is not None:
            h, w = rgb.shape[:2]
            flat_rgb = rgb.reshape(-1, 3)[mask_valid]
        else:
            flat_rgb = None

        save_ply(dst / "depth_pnts.ply", pC_filt, flat_rgb)
    else:
        print(f"  [skip] {pnts_path} no encontrado")

    # --- point_cloud (nube filtrada Drake) → PLY sin color ---
    pc_path = src / "point_cloud.npy"
    if pc_path.exists():
        pc = np.load(pc_path).astype(np.float32)     # (N, 3) XYZ
        save_ply(dst / "point_cloud.ply", pc)
    else:
        print(f"  [skip] {pc_path} no encontrado")

    print(f"\nListo. Salidas en: {dst}/")


if __name__ == "__main__":
    main()
