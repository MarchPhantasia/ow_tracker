# TODO

## P0 - Training Completion And Model Test

- [ ] Let the current server training finish without `--use-all-data`.
- [ ] Locate the produced weights:

```bash
find runs -path "*/weights/best.pt" -print
find runs -path "*/results.csv" -print
```

- [ ] Run validation on the held-out validation split:

```bash
yolo val \
  model=<BEST_PT> \
  data=runs/datasets/ally_enemy_s/data.yaml \
  imgsz=1280 \
  device=0 \
  split=val
```

- [ ] Run validation on the held-out test split:

```bash
yolo val \
  model=<BEST_PT> \
  data=runs/datasets/ally_enemy_s/data.yaml \
  imgsz=1280 \
  device=0 \
  split=test
```

- [ ] Run visual prediction on test images, first enemy-only:

```bash
python predict.py \
  --weights <BEST_PT> \
  --source datasets/test/images \
  --device 0 \
  --name enemy_test \
  --exist-ok
```

- [ ] Run visual prediction with ally boxes included for manual confusion checks:

```bash
python predict.py \
  --weights <BEST_PT> \
  --source datasets/test/images \
  --include-ally \
  --device 0 \
  --name ally_enemy_test \
  --exist-ok
```

- [ ] Inspect these output files from the training run:
  - `results.png`
  - `confusion_matrix.png`
  - `PR_curve.png`
  - `F1_curve.png`
  - `weights/best.pt`

- [ ] Acceptance criteria for the first runtime candidate:
  - Enemy precision is high enough that ally-to-enemy mistakes are rare.
  - Enemy recall is high enough that normal visible enemies are not frequently missed.
  - `confusion_matrix.png` does not show serious `ally -> enemy` confusion.
  - Manual prediction images do not show obvious map/building false positives as `enemy`.

- [ ] Copy the accepted model to the standard path:

```bash
mkdir -p models
cp <BEST_PT> models/best.pt
```

- [ ] Commit only code/config/docs changes. Do not commit `datasets/`, `runs/`, or `models/*.pt`.

## P1 - Runtime MVP

- [ ] Add runtime dependencies back only where needed:
  - `dxcam` for screen capture on Windows.
  - `pywin32` for `SendInput` mouse movement.
  - `pynput` only if needed for X2 hotkey listening.

- [ ] Create runtime modules:
  - `src/head_tracker/runtime/config.py`
  - `src/head_tracker/runtime/capture.py`
  - `src/head_tracker/runtime/detector.py`
  - `src/head_tracker/runtime/target_selector.py`
  - `src/head_tracker/runtime/mouse_mover.py`
  - `src/head_tracker/runtime/main.py`

- [ ] Runtime default behavior:
  - Load `models/best.pt`.
  - Capture center region for 2560x1440 monitor:

```yaml
screen_width: 2560
screen_height: 1440
fov_width: 1920
fov_height: 1080
```

  - Run YOLO at `imgsz=1280`.
  - Keep `enemy` detections only.
  - Ignore `ally` detections for target selection.
  - Aim at an estimated upper-body point inside the enemy box:

```python
aim_x = (x1 + x2) * 0.5
aim_y = y1 + (y2 - y1) * 0.30
```

  - Move only while X2 is held.
  - Stop immediately when X2 is released or no valid target exists.

- [ ] Add visual debug mode:
  - Draw capture region.
  - Draw ally/enemy boxes when enabled.
  - Draw selected enemy and aim point.
  - Print FPS and per-stage latency.

## P2 - Runtime Stability And False-Lock Protection

- [ ] Add target selection guards:
  - Minimum enemy confidence, start around `0.55`.
  - Maximum acquisition distance from crosshair.
  - Box size sanity checks.
  - Optional aspect-ratio sanity checks.
  - Prefer target closest to crosshair.
  - Keep current target unless another enemy is clearly better.

- [ ] Add temporal confirmation:
  - Require a fresh target to appear for 2 consecutive frames before moving.
  - Keep a locked target through short one-frame misses.
  - Drop target quickly after sustained misses.

- [ ] Add motion smoothing:
  - Kalman or One Euro filter for selected target point.
  - Configurable lead time for moving targets.
  - Direction-reversal braking to reduce overrun.
  - Deadzone near crosshair to reduce micro jitter.

- [ ] Add runtime logs:
  - `runs/runtime/latency.csv`
  - selected target confidence/class/box
  - capture/inference/postprocess/mouse timing

- [ ] Add offline replay test mode:
  - Run detector and selector on saved images/video.
  - Save annotated output.
  - Do not move the mouse in replay mode.

## P3 - Deferred / Do Not Implement Now

- [ ] Do not implement hidden input backends.
- [ ] Do not implement anti-cheat bypass features.
- [ ] Do not optimize TensorRT/ONNX until PyTorch runtime quality is accepted.
- [ ] Do not build multi-model ensembles until YOLO11s is measured and accepted/rejected.
