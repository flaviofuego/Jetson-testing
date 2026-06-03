"""
Quick test: image -> OBJ + SDF using TripoSR directly (no Docker).

Requirements (separate env from Drake):
    pip install triposr trimesh Pillow huggingface_hub

Usage:
    python quick_test.py path/to/image.png
    python quick_test.py path/to/image.png --name mug
    python quick_test.py path/to/image.png --device cpu   # sin GPU
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def preprocess(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img.thumbnail((512, 512), Image.LANCZOS)
    bg = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    bg.paste(img, ((512 - img.width) // 2, (512 - img.height) // 2), mask=img)
    return bg


def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    m.apply_scale(0.20 / m.extents.max())
    return m


def make_sdf(name: str, mesh: trimesh.Trimesh, parts: list[Path]) -> str:
    volume = mesh.volume if mesh.volume > 0 else 1e-6
    props = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    props.density = 0.1 / volume
    com = props.center_mass
    I = props.moment_inertia

    collisions = ""
    for i, p in enumerate(parts):
        collisions += f"""
      <collision name="collision_{i:03d}">
        <geometry><mesh>
          <uri>{p.as_posix()}</uri>
          <drake:declare_convex/>
        </mesh></geometry>
        <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_dynamic>1.000</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>"""

    return f"""<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="{name}">
    <link name="{name}_body_link">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <pose>{com[0]:.5f} {com[1]:.5f} {com[2]:.5f} 0 0 0</pose>
        <inertia>
          <ixx>{I[0,0]:.5e}</ixx><ixy>{I[0,1]:.5e}</ixy><ixz>{I[0,2]:.5e}</ixz>
          <iyy>{I[1,1]:.5e}</iyy><iyz>{I[1,2]:.5e}</iyz><izz>{I[2,2]:.5e}</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry><mesh><uri>{name}.obj</uri><scale>1 1 1</scale></mesh></geometry>
      </visual>{collisions}
    </link>
  </model>
</sdf>
"""


# ── TripoSR inference ──────────────────────────────────────────────────────────

def infer(image: Image.Image, device: str) -> trimesh.Trimesh:
    import torch
    from tsr.system import TSR

    print(f"  Cargando TripoSR en {device}...")
    system = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    ).to(device)
    system.renderer.set_chunk_size(8192)

    with torch.no_grad():
        codes = system([image], device=device)
        meshes = system.extract_mesh(codes, resolution=256)

    return meshes[0]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Imagen PNG/JPG")
    parser.add_argument("--name", default=None, help="Nombre del asset (default: nombre del archivo)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out", type=Path, default=Path("assets"), help="Carpeta de salida")
    args = parser.parse_args()

    name = args.name or args.image.stem
    asset_dir = args.out / name
    parts_dir = asset_dir / f"{name}_parts"
    asset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Preprocesando imagen: {args.image}")
    image = preprocess(args.image)

    print("[2/4] Inferencia TripoSR...")
    raw_mesh = infer(image, args.device)

    print("[3/4] Normalizando malla (→ 20 cm)...")
    mesh = normalize(raw_mesh)
    obj_path = asset_dir / f"{name}.obj"
    mesh.export(str(obj_path))
    print(f"      OBJ guardado: {obj_path}")

    print("[4/4] Generando SDF (hull convexo como colision)...")
    hull = trimesh.convex.convex_hull(mesh)
    parts_dir.mkdir(exist_ok=True)
    hull_path = parts_dir / "convex_piece_000.obj"
    hull.export(str(hull_path))
    sdf_path = asset_dir / f"{name}.sdf"
    sdf_path.write_text(make_sdf(name, mesh, [Path(f"{name}_parts/convex_piece_000.obj")]))
    print(f"      SDF guardado:  {sdf_path}")

    print(f"\nListo → {asset_dir}/")
    print(f"  {name}.obj   ({len(mesh.faces)} caras)")
    print(f"  {name}.sdf")
    print(f"  {name}_parts/convex_piece_000.obj")


if __name__ == "__main__":
    main()
