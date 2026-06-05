# P2C + NU-MCC API Notes
Date: 2026-06-05
Source: github.com/CuiRuikai/Partial2Complete + github.com/sail-sg/numcc

---

## P2C (Partial2Complete)

### Model class
File: `models/P2C.py`

```python
from models.P2C import P2C
```

Registered with `@MODELS.register_module()` but can be instantiated directly.

### Config (required at init, even for inference)

```python
from easydict import EasyDict

config = EasyDict({
    "num_group": 128,
    "group_size": 32,
    "mask_ratio": [48, 48, 32],  # must sum to num_group=128
    "feat_dim": 1024,
    "n_points": 2048,            # output point count
    "nbr_ratio": 1,
    # loss params — required by __init__ even at inference
    "support": 8,
    "neighborhood_size": 32,
    "shape_matching_weight": 10.0,
    "shape_recon_weight": 1.0,
    "latent_weight": 1.0,
    "manifold_weight": 0.1,
})
model = P2C(config)
```

### Load checkpoint

```python
ckpt = torch.load("p2c_checkpoint.pth", map_location="cuda")
state = ckpt.get("model", ckpt)
if isinstance(state, dict) and "base_model" in state:
    state = state["base_model"]
model.load_state_dict(state)
model.eval().cuda()
```

### Forward (inference)

```python
# Input: (B, N, 3) float32 CUDA tensor
# Output: (B, M, 3) float32 CUDA tensor — DIRECT, no dict wrapper
pred = model(partial_pts_tensor)
```

### CUDA extensions required

P2C needs two compiled CUDA extensions in `extensions/`:

```bash
pip install extensions/chamfer_dist
pip install extensions/pointops
```

Both are built from source — included in the Dockerfile build steps.

### Key dependency: pytorch3d

P2C's `ManifoldnessConstraint` calls `pytorch3d.ops.points_normals.estimate_pointcloud_normals`.
No ARM64 wheel — must be built from source (see Dockerfile).

### Checkpoint download

Official checkpoint not found at a stable URL at time of writing.
Check the repo README: https://github.com/CuiRuikai/Partial2Complete
Update `tools/download_models_numcc.py` CHECKPOINTS dict once URL is known.

---

## NU-MCC

### Model class
File: `src/model/nu_mcc.py`

```python
from src.model.nu_mcc import NUMCC
```

### Args namespace (required at init)

```python
import argparse
args = argparse.Namespace(
    nneigh=45,            # neighbors for feature aggregation
    shrink_threshold=10.0, # threshold for point coordinate shrinking
    xyz_size=112,         # image size for xyz processing
    xyz_size_hr=224,      # high-res image size (used when hr=1)
    hr=0,                 # high-res mode (0=off, 1=on)
    device="cuda",
    n_query_udf=40000,    # queries per forward pass batch
    udf_threshold=0.03,   # UDF threshold for surface extraction
    udf_n_iter=3,         # gradient descent iterations on surface
)
model = NUMCC(args=args)
```

### Load checkpoint

```python
ckpt = torch.load("numcc_checkpoint.pth", map_location="cuda")
state = ckpt.get("model", ckpt)
model.load_state_dict(state)
model.eval().cuda()
```

### Forward

```python
# Inputs:
#   seen_images:    (B, 3, H, W) float32 CUDA tensor, values in [0, 1]
#   seen_xyz:       (B, N, 3) float32 CUDA tensor — visible point cloud
#   query_xyz:      (B, Q, 3) float32 CUDA tensor — 3D positions to query
#   valid_seen_xyz: (B, N) bool CUDA tensor — True where points are valid
#
# Output: UDF values at query_xyz + anchor positions
out, anchors_xyz = model(seen_images, seen_xyz, query_xyz, valid_seen_xyz)
# out shape: (B, Q, 1 + 256*3) — first channel is UDF value
```

### Inference pattern (batched over query grid)

```python
from src.fns import shrink_points_beyond_threshold, preprocess_img

# Encode once
seen_images_proc = preprocess_img(seen_images.clone())
seen_xyz_shrunk = shrink_points_beyond_threshold(seen_xyz, shrink_threshold)
latent, up_grid_fea = model.encoder(seen_images_proc, seen_xyz_shrunk, valid_seen)
fea = model.decoderl1(latent)

# Decode in batches, find surface where UDF < threshold
for batch in query_batches:
    pred = model.decoderl2(batch, seen_xyz_shrunk, valid_seen, fea, up_grid_fea)
    pred = model.fc_out(pred)
    udf = F.relu(pred[:, :, :1])  # first channel is UDF
    surface_mask = udf.squeeze(-1) < udf_threshold
    surface_pts = batch[0][surface_mask[0]]  # (K, 3)
```

### Output: point cloud, not mesh

NU-MCC outputs surface **points** (where UDF < threshold), not a mesh.
Convert to mesh with Poisson reconstruction (open3d) — see `_points_to_mesh()` in pipeline.py.

### Checkpoint download

Check the official repo README: https://github.com/sail-sg/numcc
Update `tools/download_models_numcc.py` CHECKPOINTS dict once URL is known.

---

## Key pipeline insight

P2C and NU-MCC are used in sequence:
1. Depth → partial point cloud (back-projection)
2. P2C: partial (N=2048) → completed (M=2048) point cloud
3. NU-MCC: RGB image + completed point cloud (as seen_xyz) → UDF predictions → surface points
4. Poisson/BPA: surface points → mesh
5. normalize → coacd → SDF (reusing existing utilities)

The P2C completed cloud is passed to NU-MCC as `seen_xyz`, giving NU-MCC full object coverage
even for occluded regions that the depth sensor couldn't see directly.
