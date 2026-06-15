# SAM2 Interactive Mask Selector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `submodules/sam2/pipeline_interactive.py` — a standalone script that opens a matplotlib GUI for point-click SAM2 segmentation, producing drop-in compatible output with `pipeline.py`.

**Architecture:** Script auto-relaunches inside `sam2:x86` Docker (same mechanism as `pipeline.py`). Inside Docker, loads `SAM2ImagePredictor` once, opens a two-panel matplotlib TkAgg window, and updates the mask preview on every click. Saves `<name>.png` + `<name>_segmask.npy` + `<name>_viz.png` on Enter.

**Tech Stack:** Python 3.11, SAM2ImagePredictor (sam2), matplotlib TkAgg, numpy, PIL, Docker X11 forwarding.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `submodules/sam2/pipeline_interactive.py` | **Create** | Interactive selector — complete standalone script |
| `submodules/sam2/Dockerfile.x86` | **Modify** | Add `python3-tk` to runtime stage apt-get |
| `submodules/sam2/Dockerfile.x86-server` | **Modify** | Same — add `python3-tk` |

No changes to `pipeline.py`, `tools/`, or any other file.

---

## Task 1: Add python3-tk to Dockerfile.x86 runtime stage

`python3-tk` is required for matplotlib's TkAgg backend. The current runtime stage only installs
`libgl1 libglib2.0-0`. Without this, the interactive window will never open.

**Files:**
- Modify: `submodules/sam2/Dockerfile.x86:45-47`
- Modify: `submodules/sam2/Dockerfile.x86-server` (same change, locate the runtime apt-get block)

- [ ] **Step 1: Add python3-tk to Dockerfile.x86 runtime stage**

In `submodules/sam2/Dockerfile.x86`, find the runtime `apt-get install` block (currently lines 45-47):

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

Change to:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 python3-tk \
    && rm -rf /var/lib/apt/lists/*
```

- [ ] **Step 2: Apply same change to Dockerfile.x86-server**

Find the equivalent runtime `apt-get` block in `submodules/sam2/Dockerfile.x86-server` and add `python3-tk` there as well.

- [ ] **Step 3: Rebuild sam2:x86**

```bash
docker build -t sam2:x86 -f submodules/sam2/Dockerfile.x86 submodules/sam2/
```

Expected: build completes, `python3-tk` installed in runtime stage.

Verify:
```bash
docker run --rm sam2:x86 python3 -c "import tkinter; print('tk ok')"
```
Expected output: `tk ok`

- [ ] **Step 4: Commit**

```bash
git -C submodules/sam2 add Dockerfile.x86 Dockerfile.x86-server
git -C submodules/sam2 commit -m "feat(docker): add python3-tk for matplotlib TkAgg interactive support"
git add submodules/sam2
git commit -m "chore(submodules): update sam2 ref — python3-tk in runtime"
```

---

## Task 2: Implement pipeline_interactive.py — skeleton + Docker relaunch

Create the script with the auto-Docker-relaunch block, DISPLAY check, and CLI parsing.
No SAM2 logic yet — just the scaffolding that can be run from the host and enters the container.

**Files:**
- Create: `submodules/sam2/pipeline_interactive.py`

- [ ] **Step 1: Create the script with relaunch + CLI**

Create `submodules/sam2/pipeline_interactive.py`:

```python
"""
SAM2 interactive mask selector. Auto-relaunches inside Docker when run from the host.
Left click = positive point, right click = negative point.
R = reset, Enter = save, Q = quit without saving.

Usage (host — auto Docker):
  python3 submodules/sam2/pipeline_interactive.py --input data/images/taza/taza.jpeg --output data/outputs --name taza

Usage (inside container):
  python3 /opt/sam2/pipeline_interactive.py --input /input/taza.jpeg --output /output --name taza
"""
# matplotlib backend MUST be set before any other import that might pull in pyplot
import matplotlib
matplotlib.use('TkAgg')

import argparse
import os
import subprocess
import sys
from pathlib import Path

IN_DOCKER = Path('/.dockerenv').exists()

if IN_DOCKER:
    sys.path.insert(0, '/opt/sam2')

import numpy as np
from PIL import Image

DOCKER_IMAGE   = 'sam2:x86'
CKPTS_DIR_HOST = Path.home() / 'models' / 'sam2'
SCRIPT_HOST    = Path(__file__).resolve()

MODELS = {
    'tiny':      ('sam2.1_hiera_tiny.pt',      'configs/sam2.1/sam2.1_hiera_t.yaml'),
    'small':     ('sam2.1_hiera_small.pt',     'configs/sam2.1/sam2.1_hiera_s.yaml'),
    'base_plus': ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
}


def _relaunch_in_docker(args_raw: list[str]) -> None:
    display = os.environ.get('DISPLAY', '')
    if not display:
        print('ERROR: $DISPLAY not set. Run: xhost +local:docker && export DISPLAY=:0', file=sys.stderr)
        sys.exit(1)

    input_path = output_path = None
    for i, a in enumerate(args_raw):
        if a in ('--input', '-input') and i + 1 < len(args_raw):
            input_path = Path(args_raw[i + 1]).resolve()
        if a in ('--output', '-output') and i + 1 < len(args_raw):
            output_path = Path(args_raw[i + 1]).resolve()

    if input_path is None or output_path is None:
        print('ERROR: --input and --output are required', file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    forwarded = []
    i = 0
    while i < len(args_raw):
        a = args_raw[i]
        if a in ('--input', '-input') and i + 1 < len(args_raw):
            forwarded += ['--input', f'/input/{Path(args_raw[i+1]).name}']
            i += 2
        elif a in ('--output', '-output') and i + 1 < len(args_raw):
            forwarded += ['--output', '/output']
            i += 2
        else:
            forwarded.append(a)
            i += 1

    cmd = [
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '--network', 'host',
        '-e', f'DISPLAY={display}',
        '-v', '/tmp/.X11-unix:/tmp/.X11-unix',
        '-v', f'{input_path.parent}:/input:ro',
        '-v', f'{output_path}:/output',
        '-v', f'{CKPTS_DIR_HOST}:/opt/sam2/checkpoints:ro',
        '-v', f'{SCRIPT_HOST}:/opt/sam2/pipeline_interactive.py:ro',
        '--entrypoint', 'python3',
        DOCKER_IMAGE,
        '/opt/sam2/pipeline_interactive.py',
    ] + forwarded

    xauth = os.environ.get('XAUTHORITY', '')
    if xauth:
        cmd = cmd[:3] + ['-e', f'XAUTHORITY={xauth}', '-v', f'{xauth}:{xauth}'] + cmd[3:]

    print(f'Launching Docker: {" ".join(cmd)}')
    sys.exit(subprocess.run(cmd).returncode)


def parse_args():
    parser = argparse.ArgumentParser(description='SAM2 interactive mask selector')
    parser.add_argument('--input',  required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--name',   required=True)
    parser.add_argument('--model',  default='small', choices=list(MODELS.keys()))
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    if not IN_DOCKER:
        _relaunch_in_docker(sys.argv[1:])

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Verify DISPLAY inside container too
    if not os.environ.get('DISPLAY'):
        print('ERROR: $DISPLAY not set inside container', file=sys.stderr)
        sys.exit(1)

    image_np = np.array(Image.open(args.input).convert('RGB'))
    print(f'Image loaded: {args.input.name} {image_np.shape[1]}x{image_np.shape[0]}')

    # SAM2 + GUI logic added in Task 3
    print('TODO: interactive GUI not yet implemented')
    sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test relaunch from host**

```bash
.venv/bin/python3 submodules/sam2/pipeline_interactive.py \
  --input data/images/taladro.JPG \
  --output data/outputs \
  --name test_interactive
```

Expected: prints `Launching Docker: ...`, enters container, prints `Image loaded: taladro.JPG 860x672`, then prints `TODO: interactive GUI not yet implemented` and exits 1.

- [ ] **Step 3: Commit skeleton**

```bash
git -C submodules/sam2 add pipeline_interactive.py
git -C submodules/sam2 commit -m "feat(sam2): add pipeline_interactive.py skeleton with Docker relaunch"
git add submodules/sam2
git commit -m "chore(submodules): update sam2 ref — interactive pipeline skeleton"
```

---

## Task 3: Implement the interactive GUI and SAM2 predictor loop

Replace the `TODO` stub in `main()` with the full matplotlib TkAgg interaction loop.

**Files:**
- Modify: `submodules/sam2/pipeline_interactive.py`

- [ ] **Step 1: Replace main() body with full implementation**

Replace everything after `image_np = np.array(...)` in `main()` with:

```python
    h, w = image_np.shape[:2]
    print(f'Image loaded: {args.input.name} {w}x{h}')

    # Load SAM2 predictor
    print(f'Loading SAM2 {args.model}...')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import matplotlib.pyplot as plt
    ckpt_name, cfg = MODELS[args.model]
    sam = build_sam2(cfg, f'/opt/sam2/checkpoints/{ckpt_name}', device=args.device)
    predictor = SAM2ImagePredictor(sam)

    print('Encoding image (one-time, ~1-2s)...')
    predictor.set_image(image_np)
    print('Ready. Click on the object.')

    # Interaction state
    points  = []   # list of [x, y]
    labels  = []   # list of 0 or 1
    current_mask = None

    fig, (ax_orig, ax_result) = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.manager.set_window_title('SAM2 Interactive — Left=add | Right=exclude | R=reset | Enter=save | Q=quit')

    ax_orig.imshow(image_np)
    ax_orig.set_title('Left click = add object point\nRight click = exclude point')
    ax_orig.axis('off')

    # Keep persistent reference — never use ax_result.images[0] (brittle after .remove())
    result_im = ax_result.imshow(np.ones_like(image_np) * 255)
    ax_result.set_title('Result (no points yet)')
    ax_result.axis('off')

    plt.tight_layout()

    point_artists = []
    mask_artist   = [None]

    def _update_display():
        # Redraw left panel points
        for a in point_artists:
            a.remove()
        point_artists.clear()
        for (px, py), lbl in zip(points, labels):
            color = 'lime' if lbl == 1 else 'red'
            artist, = ax_orig.plot(px, py, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
            point_artists.append(artist)

        # Redraw right panel via persistent result_im reference
        if mask_artist[0] is not None:
            mask_artist[0].remove()
            mask_artist[0] = None

        if current_mask is not None:
            result = np.ones_like(image_np) * 255
            result[current_mask] = image_np[current_mask]
            result_im.set_data(result)
            ax_result.set_title(f'Mask area: {current_mask.sum():,} px²')

            # Overlay on left panel
            overlay = np.zeros((h, w, 4), dtype=np.float32)
            overlay[current_mask] = [0.0, 1.0, 0.0, 0.35]
            mask_artist[0] = ax_orig.imshow(overlay)
        else:
            result_im.set_data(np.ones_like(image_np) * 255)
            ax_result.set_title('Result (no points yet)')

        fig.canvas.draw_idle()

    def _run_predict():
        nonlocal current_mask
        if not points:
            return
        pts_arr = np.array(points, dtype=np.float32)
        lbl_arr = np.array(labels, dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=pts_arr,
            point_labels=lbl_arr,
            multimask_output=True,
        )
        # masks: (3, H, W) bool; scores: (3,)
        current_mask = masks[scores.argmax()].astype(bool)

    def on_click(event):
        if event.inaxes is not ax_orig:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        label = 1 if event.button == 1 else 0
        points.append([x, y])
        labels.append(label)
        _run_predict()
        _update_display()

    def on_key(event):
        nonlocal current_mask
        if event.key == 'r':
            points.clear()
            labels.clear()
            current_mask = None
            _update_display()
            fig.canvas.manager.set_window_title('SAM2 Interactive — Reset. Click on the object.')
        elif event.key == 'enter':
            if not points or current_mask is None:
                fig.canvas.manager.set_window_title('SAM2 Interactive — No points! Click the object first.')
                return
            _save_and_exit()
        elif event.key == 'q':
            print('Quit without saving.')
            plt.close(fig)
            sys.exit(1)

    def _save_and_exit():
        from PIL import Image as PILImage
        # Crop to bbox with padding
        rows = np.any(current_mask, axis=1)
        cols = np.any(current_mask, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = 20
        r0 = max(0, r0 - pad); r1 = min(h, r1 + pad)
        c0 = max(0, c0 - pad); c1 = min(w, c1 + pad)

        rgba = np.ones((h, w, 4), dtype=np.uint8) * 255
        rgba[:, :, :3] = image_np
        rgba[:, :, 3]  = current_mask.astype(np.uint8) * 255
        crop = PILImage.fromarray(rgba[r0:r1, c0:c1])
        white = PILImage.new('RGB', crop.size, (255, 255, 255))
        white.paste(crop, mask=crop.split()[3])
        out_png = args.output / f'{args.name}.png'
        white.save(out_png)
        print(f'Saved: {out_png}')

        # segmask
        out_npy = args.output / f'{args.name}_segmask.npy'
        np.save(str(out_npy), current_mask.astype(np.uint8))
        print(f'Saved: {out_npy}')

        # viz — original + overlay + points
        viz_img = image_np.copy()
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[current_mask] = [0, 200, 0, 100]
        viz_pil = PILImage.fromarray(viz_img)
        ov_pil  = PILImage.fromarray(overlay, 'RGBA')
        viz_pil = viz_pil.convert('RGBA')
        viz_pil.alpha_composite(ov_pil)
        out_viz = args.output / f'{args.name}_viz.png'
        viz_pil.convert('RGB').save(out_viz)
        print(f'Saved: {out_viz}')

        plt.close(fig)
        sys.exit(0)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
    sys.exit(1)  # reached only if window closed via X button (no save)
```

- [ ] **Step 2: Test — run on taladro image**

```bash
xhost +local:docker
.venv/bin/python3 submodules/sam2/pipeline_interactive.py \
  --input data/images/taladro.JPG \
  --output data/outputs \
  --name taladro_interactive
```

Expected: window opens, click on drill, green overlay appears, press Enter → files saved:
- `data/outputs/taladro_interactive.png` (drill on white bg)
- `data/outputs/taladro_interactive_segmask.npy`
- `data/outputs/taladro_interactive_viz.png`

Verify output PNG shows drill (not background):
```bash
# File should exist and be non-zero
ls -lh data/outputs/taladro_interactive.png
```

- [ ] **Step 3: Test — verify output compatibility with pipeline.py format**

```python
# Quick Python check
import numpy as np
mask = np.load('data/outputs/taladro_interactive_segmask.npy')
print(mask.dtype, mask.shape, mask.max())
# Expected: uint8 (672, 860) 1
```

- [ ] **Step 4: Test — Q key exits with code 1 (no files saved)**

Run the script, immediately press Q. Verify no output files were written (or pre-existing ones unchanged).

- [ ] **Step 5: Test — Enter with no clicks shows warning, keeps window open**

Run the script, immediately press Enter (no clicks). Window should stay open with title updated to warning message. Then click once and press Enter — saves correctly.

- [ ] **Step 6: Test — negative points exclude region**

Run on taza image, click on table (label=right click / exclude), then click on cup. Mask should exclude the table region.

```bash
.venv/bin/python3 submodules/sam2/pipeline_interactive.py \
  --input data/images/taza/taza.jpeg \
  --output data/outputs \
  --name taza_interactive
```

- [ ] **Step 7: Commit**

```bash
git -C submodules/sam2 add pipeline_interactive.py
git -C submodules/sam2 commit -m "feat(sam2): implement interactive mask selector with TkAgg GUI"
git add submodules/sam2
git commit -m "chore(submodules): update sam2 ref — interactive selector complete"
```

---

## Task 4: Copy pipeline_interactive.py into Docker image (standalone use)

Same pattern as `pipeline.py` — include a copy in the image for standalone use, even though
it gets overridden by the volume mount at runtime.

**Files:**
- Modify: `submodules/sam2/Dockerfile.x86` (runtime stage — add COPY line)
- Modify: `submodules/sam2/Dockerfile.x86-server` (same)

- [ ] **Step 1: Add COPY to both Dockerfiles**

In both `Dockerfile.x86` and `Dockerfile.x86-server`, after the existing `COPY pipeline.py ./` line in the runtime stage, add:

```dockerfile
COPY pipeline_interactive.py ./
```

- [ ] **Step 2: Rebuild image**

```bash
docker build -t sam2:x86 -f submodules/sam2/Dockerfile.x86 submodules/sam2/
```

- [ ] **Step 3: Verify both scripts exist in image**

```bash
docker run --rm sam2:x86 ls /opt/sam2/
```

Expected output includes: `pipeline.py  pipeline_interactive.py`

- [ ] **Step 4: Commit**

```bash
git -C submodules/sam2 add Dockerfile.x86
git -C submodules/sam2 commit -m "feat(docker): include pipeline_interactive.py in sam2:x86 image"
git add submodules/sam2
git commit -m "chore(submodules): update sam2 ref — interactive script in Docker image"
```

---

## Task 5: Update CLAUDE.md with new script docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add section for pipeline_interactive.py**

In `CLAUDE.md`, after the existing SAM2 section (after the benchmarks table), add:

```markdown
### SAM2 — selector interactivo (point-click)

Para casos donde AMG elige la máscara equivocada o el fondo contamina el resultado.

```bash
# Prerequisito X11 (una sola vez por sesión)
xhost +local:docker

# Correr selector interactivo
.venv/bin/python3 submodules/sam2/pipeline_interactive.py \
  --input data/images/objeto.jpg \
  --output data/outputs \
  --name objeto
```

**Controles:**
- Click izquierdo → punto positivo (incluir)
- Click derecho → punto negativo (excluir)
- `R` → resetear puntos
- `Enter` → guardar máscara y salir (exit 0)
- `Q` → salir sin guardar (exit 1)

**Output:** mismo formato que `pipeline.py` — `<name>.png` + `<name>_segmask.npy` + `<name>_viz.png`.
Compatible con `--skip-sam2` en `tools/pipeline.py` si se copian los outputs a `output/<name>/02_sam2/`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): add SAM2 interactive selector usage"
```
