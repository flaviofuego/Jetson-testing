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

Same path-rewriting mechanism as `pipeline.py` (only `--input` and `--output` need path
rewriting; all other args pass through unchanged).

If not in Docker → builds and execs `docker run` with:
- `-v <input_parent>:/input:ro`
- `-v <output>:/output`
- `-v ~/models/sam2:/opt/sam2/checkpoints:ro`
- `-v <script>:/opt/sam2/pipeline_interactive.py:ro`
- `-e DISPLAY=$DISPLAY`
- `-e XAUTHORITY=$XAUTHORITY` (if set)
- `-v /tmp/.X11-unix:/tmp/.X11-unix`
- `--network host`

**Host X11 prerequisite:** before running, the host must allow Docker access to the X server:
```bash
xhost +local:docker
```
The script checks `$DISPLAY` at startup; if unset, prints a clear error and exits with code 1.

## UI Layout

Two-panel matplotlib figure (side by side). Backend set to `TkAgg` explicitly
(`matplotlib.use('TkAgg')` before any other matplotlib import). `python3-tk` must be available
in the Docker image — add `apt-get install -y python3-tk` to `sam2:x86` Dockerfile if missing.

| Left panel | Right panel |
|---|---|
| Original image + clicked points | Current segmentation result (white background) |
| Green dots = positive points | Updates after every click |
| Red dots = negative points | Shows mask area in px² |

Title bar: `"Left click = add | Right click = exclude | R = reset | Enter = save | Q = quit"`

## Interaction Loop

1. Check `$DISPLAY` set — exit code 1 with message if not
2. `set_image(image_np)` once at startup (~1-2s, shown as loading message in title)
3. On left click → append `(x, y, label=1)` to points list → call predict → redraw
4. On right click → append `(x, y, label=0)` to points list → call predict → redraw
5. `predict(point_coords, point_labels, multimask_output=True)` returns:
   - `masks`: `(3, H, W)` bool — three mask candidates
   - `scores`: `(3,)` float — confidence per candidate
   - Select `masks[scores.argmax()]` → shape `(H, W)`
6. `R` key → clear points list → clear right panel → reset title
7. `Enter` key:
   - If no points clicked → flash warning in title bar ("No points — click the object first"), keep window open
   - If points exist → save outputs → exit code 0
8. `Q` key → exit code 1 (no save; caller can detect abandonment via non-zero exit)

## Output Format

- `<output>/<name>.png` — cropped object on white background (RGB PNG) — **drop-in compatible**
- `<output>/<name>_segmask.npy` — binary mask uint8, original image dimensions — **drop-in compatible**
- `<output>/<name>_viz.png` — original image with final mask overlay + clicked points marked
  (format differs from AMG viz which shows all masks; content is different but filename is same)

## Compatibility

`<name>.png` and `<name>_segmask.npy` are drop-in compatible with:
- `tools/pipeline.py` stage `02_sam2/` (reads `_segmask.npy` for depth masking)
- `pipeline_multiview.py` (reads same format)

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Mask saved successfully |
| 1 | Quit without saving, X11 unavailable, or error |

## Dependencies

Inside `sam2:x86` Docker image:
- `matplotlib` with `TkAgg` backend → requires `python3-tk` (`apt-get install -y python3-tk`)
- `sam2.sam2_image_predictor.SAM2ImagePredictor`
- `numpy`, `PIL`

Do NOT import from `pipeline.py` — the interactive script is standalone. Do NOT call
`matplotlib.use('Agg')`.

## What This Does NOT Do

- Does not replace `pipeline.py` — AMG mode stays for automated/batch use
- Does not run AMG first — goes straight to predictor mode (faster startup)
- Does not support multi-object selection in one session
