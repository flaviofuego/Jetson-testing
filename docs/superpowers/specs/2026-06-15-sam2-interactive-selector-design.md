# SAM2 Interactive Mask Selector — Design Spec

**Date:** 2026-06-15  
**Status:** Approved

## Problem

`pipeline.py` uses AMG (Automatic Mask Generator) with fixed thresholds. Objects with complex
backgrounds require threshold tuning; objects that are simple but misidentified need manual
correction. No interactive fallback exists.

## Solution

`pipeline_interactive.py` — a separate script that opens a matplotlib window, lets the user
click positive/negative points on the image, runs SAM2 `SAM2ImagePredictor` in real-time, and
saves output in the same format as `pipeline.py`.

## Script Location

`submodules/sam2/pipeline_interactive.py`

## CLI

```bash
python3 submodules/sam2/pipeline_interactive.py \
  --input data/images/taza/taza.jpeg \
  --output data/outputs \
  --name taza \
  [--model small|tiny|base_plus]   # default: small
```

## Auto-Docker Relaunch

Same mechanism as `pipeline.py`:
- Detects `/.dockerenv`
- If not in Docker → builds and execs `docker run` with:
  - `-v <input_parent>:/input:ro`
  - `-v <output>:/output`
  - `-v ~/models/sam2:/opt/sam2/checkpoints:ro`
  - `-v <script>:/opt/sam2/pipeline_interactive.py:ro`
  - `-e DISPLAY=$DISPLAY`
  - `-v /tmp/.X11-unix:/tmp/.X11-unix`
  - `--network host` (for X11 on some distros)

## UI Layout

Two-panel matplotlib figure (side by side):

| Left panel | Right panel |
|---|---|
| Original image + clicked points | Current segmentation result (white background) |
| Green dots = positive points | Updates after every click |
| Red dots = negative points | Shows mask area in px² |

Title bar: `"Left click = add object | Right click = exclude | R = reset | Enter = save | Q = quit"`

## Interaction Loop

1. `set_image(image_np)` once at startup (expensive ~1-2s)
2. On left click → append `(x, y, label=1)` to points list
3. On right click → append `(x, y, label=0)` to points list
4. After each click → call `predictor.predict(point_coords, point_labels, multimask_output=True)` → select `masks[scores.argmax()]` → redraw right panel
5. `R` key → clear points list → clear right panel
6. `Enter` key → save outputs → close window
7. `Q` key → exit without saving

## Output Format

Identical to `pipeline.py`:
- `<output>/<name>.png` — cropped object on white background (RGB PNG)
- `<output>/<name>_segmask.npy` — binary mask uint8, original image dimensions
- `<output>/<name>_viz.png` — original image with mask overlay + clicked points

## Compatibility

Output files are drop-in compatible with the rest of the pipeline:
- `tools/pipeline.py` reads `_segmask.npy` for depth masking in stage `02_sam2/`
- `pipeline_multiview.py` reads the same format

## Dependencies

All available inside `sam2:x86` Docker image:
- `matplotlib` (already installed)
- `sam2.sam2_image_predictor.SAM2ImagePredictor`
- `numpy`, `PIL`

X11 forwarding required on host (`DISPLAY` env var must be set).

## What This Does NOT Do

- Does not replace `pipeline.py` — AMG mode stays for automated/batch use
- Does not run AMG first — goes straight to predictor mode (faster startup)
- Does not support multi-object selection in one session
