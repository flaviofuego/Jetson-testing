from pathlib import Path
import trimesh
import numpy as np


_MASS = 0.1  # kg, consistent with all existing AIRA assets


def generate_sdf(name: str, mesh: trimesh.Trimesh, parts: list[Path]) -> str:
    """Generate Drake SDF XML string for an asset."""
    volume = mesh.volume if mesh.volume > 0 else 1e-6
    density = _MASS / volume
    props = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    props.density = density
    com = props.center_mass
    I = props.moment_inertia  # 3x3

    collision_blocks = ""
    for i, part_path in enumerate(parts):
        collision_blocks += f"""
      <collision name="collision_{i:03d}">
        <geometry>
          <mesh>
            <uri>{part_path.as_posix()}</uri>
            <drake:declare_convex/>
          </mesh>
        </geometry>
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
        <mass>{_MASS}</mass>
        <pose>{com[0]:.5f} {com[1]:.5f} {com[2]:.5f} 0 0 0</pose>
        <inertia>
          <ixx>{I[0,0]:.5e}</ixx>
          <ixy>{I[0,1]:.5e}</ixy>
          <ixz>{I[0,2]:.5e}</ixz>
          <iyy>{I[1,1]:.5e}</iyy>
          <iyz>{I[1,2]:.5e}</iyz>
          <izz>{I[2,2]:.5e}</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>{name}.obj</uri>
            <scale>1.0 1.0 1.0</scale>
          </mesh>
        </geometry>
      </visual>{collision_blocks}
    </link>
  </model>
</sdf>
"""
