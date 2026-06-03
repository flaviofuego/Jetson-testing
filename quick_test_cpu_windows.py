"""
Quick test: image -> OBJ + SDF using TripoSR on Windows CPU (no GPU).

Requirements (separate env from Drake):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install triposr trimesh Pillow huggingface_hub

Usage:
  python quick_test_cpu_windows.py path/to/image.png
  python quick_test_cpu_windows.py path/to/image.png --name mug
"""
import argparse
from pathlib import Path

import numpy as np
import rembg
import trimesh
from PIL import Image


# -- helpers -----------------------------------------------------------------

def preprocess(path: Path, foreground_ratio: float = 0.85) -> Image.Image:
    from tsr.utils import remove_background, resize_foreground

    img = Image.open(path)
    # Remove background with rembg → clean RGBA mask
    session = rembg.new_session()
    img = remove_background(img, session)
    # Crop + pad so the object fills `foreground_ratio` of the frame
    img = resize_foreground(img, foreground_ratio)
    # Composite onto gray (0.5) — the background TripoSR was trained on
    arr = np.array(img, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + 0.5 * (1.0 - arr[:, :, 3:4])
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


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
    inertia = props.moment_inertia

    collisions = ""
    for i, p in enumerate(parts):
        collisions += f"""
      <collision name=\"collision_{i:03d}\">
        <geometry><mesh>
          <uri>{p.as_posix()}</uri>
          <drake:declare_convex/>
        </mesh></geometry>
        <drake:proximity_properties>
          <drake:rigid_hydroelastic/>
          <drake:mu_dynamic>1.000</drake:mu_dynamic>
        </drake:proximity_properties>
      </collision>"""

    return f"""<sdf xmlns:drake=\"drake.mit.edu\" version=\"1.7\">
  <model name=\"{name}\">
    <link name=\"{name}_body_link\">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <pose>{com[0]:.5f} {com[1]:.5f} {com[2]:.5f} 0 0 0</pose>
        <inertia>
          <ixx>{inertia[0,0]:.5e}</ixx><ixy>{inertia[0,1]:.5e}</ixy><ixz>{inertia[0,2]:.5e}</ixz>
          <iyy>{inertia[1,1]:.5e}</iyy><iyz>{inertia[1,2]:.5e}</iyz><izz>{inertia[2,2]:.5e}</izz>
        </inertia>
      </inertial>
      <visual name=\"visual\">
        <geometry><mesh><uri>{name}.obj</uri><scale>1 1 1</scale></mesh></geometry>
      </visual>{collisions}
    </link>
  </model>
</sdf>
"""


# -- TripoSR inference --------------------------------------------------------

def infer_cpu(image: Image.Image) -> trimesh.Trimesh:
    import torch
    from tsr.system import TSR

    device = "cpu"
    print("  Loading TripoSR on CPU...")
    system = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    ).to(device)
    system.renderer.set_chunk_size(8192)

    with torch.no_grad():
        codes = system([image], device=device)
        meshes = system.extract_mesh(codes, has_vertex_color=False, resolution=256)

    return meshes[0]


# -- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Input PNG/JPG image")
    parser.add_argument("--name", default=None, help="Asset name (default: file name)")
    parser.add_argument("--out", type=Path, default=Path("assets"), help="Output folder")
    args = parser.parse_args()

    name = args.name or args.image.stem
    asset_dir = args.out / name
    parts_dir = asset_dir / f"{name}_parts"
    asset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Preprocessing image: {args.image}")
    image = preprocess(args.image)

    print("[2/4] TripoSR inference (CPU)...")
    raw_mesh = infer_cpu(image)

    print("[3/4] Normalizing mesh (to 20 cm)...")
    mesh = normalize(raw_mesh)
    obj_path = asset_dir / f"{name}.obj"
    mesh.export(str(obj_path))
    print(f"      OBJ saved: {obj_path}")

    print("[4/4] Generating SDF (single convex hull)...")
    hull = trimesh.convex.convex_hull(mesh)
    parts_dir.mkdir(exist_ok=True)
    hull_path = parts_dir / "convex_piece_000.obj"
    hull.export(str(hull_path))
    sdf_path = asset_dir / f"{name}.sdf"
    sdf_path.write_text(make_sdf(name, mesh, [Path(f"{name}_parts/convex_piece_000.obj")]))
    print(f"      SDF saved:  {sdf_path}")

    print(f"\nDone -> {asset_dir}/")
    print(f"  {name}.obj   ({len(mesh.faces)} faces)")
    print(f"  {name}.sdf")
    print(f"  {name}_parts/convex_piece_000.obj")


if __name__ == "__main__":
    main()
