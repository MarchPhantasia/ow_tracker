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

Diagnostic training with a validation split:

```powershell
python train/train.py --data datasets/data.yaml --model yolo11s.pt --imgsz 1280 --batch 16 --epochs 200 --device 0 --name ally_enemy_diag --workers 8
```

Recommend the best final epoch from a diagnostic run:

```powershell
python train/train.py --recommend-epoch-from runs/train/ally_enemy_diag/results.csv --select-metric "metrics/mAP50-95(B)"
```

Final full-data training after choosing an epoch:

```powershell
python train/train.py --data datasets/data.yaml --model yolo11s.pt --imgsz 1280 --batch 16 --epochs <EPOCH> --device 0 --name ally_enemy_final --workers 8 --use-all-data --copy-best always
```

The final command copies the best weight to `models/best.pt`.

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
