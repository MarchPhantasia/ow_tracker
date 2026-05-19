# Ally/Enemy YOLO Framework

This repository is now a compact training and inference framework for a two-class YOLO dataset:

- `0: ally`
- `1: enemy`

Runtime inference defaults to `enemy` only. The `ally` class is still trained so the model learns what not to target.

## Dataset Layout

Put the YOLO dataset under `datasets/`:

```text
datasets/
  data.yaml
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

Roboflow exports often write `../train/images` in `data.yaml`. The tools repair that automatically and generate a clean config under `runs/datasets/.../data.yaml`.

## Environment

```powershell
mamba env create -f environment.yml
mamba activate head_tracker
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Use the CUDA wheel that matches the server. For your A800 server, `cu128` is fine if the driver supports it.

## Inspect Dataset

```powershell
python tools/inspect_dataset.py --data datasets/data.yaml
```

This prints image counts, label counts, empty labels, invalid labels, and class instance counts.

## Train

Train the first official YOLO11s model while keeping `valid` and `test` as held-out evaluation splits:

```powershell
python train/train.py --data datasets/data.yaml --model yolo11s.pt --imgsz 1280 --batch 32 --epochs 200 --device 0 --name ally_enemy_s --workers 8 --copy-best always
```

If training stops at `AMP: running Automatic Mixed Precision (AMP) checks...` for several minutes, disable Ultralytics AMP checks and mixed precision:

```powershell
python train/train.py --data datasets/data.yaml --model yolo11s.pt --imgsz 1280 --batch 32 --epochs 200 --device 0 --name ally_enemy_s --workers 8 --copy-best always --amp false
```

The final command copies the best validation weight to `models/best.pt`.

Do not use `--use-all-data` for the normal run. This dataset is large enough to keep validation/test splits for model selection.

## Test A Trained Model

Evaluate on the test split:

```powershell
yolo val model=models/best.pt data=runs/datasets/ally_enemy_s/data.yaml imgsz=1280 device=0 split=test
```

Run enemy-only visual prediction:

```powershell
python predict.py --weights models/best.pt --source datasets/test/images --device 0 --name enemy_test --exist-ok
```

## Predict

Enemy-only prediction on the test set:

```powershell
python predict.py --weights models/best.pt --source datasets/test/images --device 0
```

Predict both ally and enemy:

```powershell
python predict.py --weights models/best.pt --source datasets/test/images --include-ally --device 0
```

Outputs are saved under `runs/predict/`.
