# Ally/Enemy YOLO 训练与推理框架

这个仓库现在是一个精简的二分类 YOLO 框架，用来训练和测试守望先锋画面中的敌我识别模型。

类别固定为：

- `0: ally`
- `1: enemy`

推理默认只输出 `enemy`。`ally` 仍然参与训练，因为模型需要学会“哪些目标不能当敌人处理”。

## 数据集结构

把 YOLO 格式数据集放到 `datasets/` 下：

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

Roboflow 导出的 `data.yaml` 里经常会写 `../train/images` 这种路径。本项目的训练和检查工具会自动修正路径，并在 `runs/datasets/.../data.yaml` 下生成一份 Ultralytics 可以直接使用的配置。

## 环境安装

```powershell
mamba env create -f environment.yml
mamba activate head_tracker
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CUDA wheel 需要和服务器/本机驱动匹配。A800 服务器可以先用 `cu128`。

## 检查数据集

```powershell
python tools/inspect_dataset.py --data datasets/data.yaml
```

这个命令会输出：

- 每个 split 的图片数量
- 标签数量
- 空标签数量
- 无效标签数量
- `ally/enemy` 实例数量
- 自动生成后的 YOLO data config 路径

## 训练

当前数据量已经足够保留 `valid` 和 `test`，正常训练不要使用 `--use-all-data`。

首个正式模型建议先训练 YOLO11s：

```powershell
python train/train.py --data datasets/data.yaml --model yolo11s.pt --imgsz 1280 --batch 32 --epochs 200 --device 0 --name ally_enemy_s --workers 8 --copy-best always
```

如果训练卡在下面这一行超过几分钟：

```text
AMP: running Automatic Mixed Precision (AMP) checks...
```

关闭 AMP 后重跑：

```powershell
python train/train.py --data datasets/data.yaml --model yolo11s.pt --imgsz 1280 --batch 32 --epochs 200 --device 0 --name ally_enemy_s --workers 8 --copy-best always --amp false
```

训练完成后，验证集上最好的权重会复制到：

```text
models/best.pt
```

## 测试训练好的模型

先在 test split 上跑指标：

```powershell
yolo val model=models/best.pt data=runs/datasets/ally_enemy_s/data.yaml imgsz=1280 device=0 split=test
```

再跑 enemy-only 可视化预测：

```powershell
python predict.py --weights models/best.pt --source datasets/test/images --device 0 --name enemy_test --exist-ok
```

如需同时查看 ally 和 enemy，用于人工检查混淆情况：

```powershell
python predict.py --weights models/best.pt --source datasets/test/images --include-ally --device 0 --name ally_enemy_test --exist-ok
```

重点检查：

- `confusion_matrix.png` 里是否有明显 `ally -> enemy` 错误
- `Enemy` precision 是否足够高
- `Enemy` recall 是否足够高
- 可视化结果里是否有建筑、地图物体被识别成 `enemy`

## 离线推理

默认只输出 enemy：

```powershell
python predict.py --weights models/best.pt --source datasets/test/images --device 0
```

同时输出 ally 和 enemy：

```powershell
python predict.py --weights models/best.pt --source datasets/test/images --include-ally --device 0
```

推理结果会保存到：

```text
runs/predict/
```

## 后续开发

当前仓库已经完成：

- 数据集检查
- YOLO11s 训练入口
- 训练结果权重复制
- 离线图片/目录/视频推理
- enemy-only 推理过滤

还没有完成：

- 实时屏幕捕获
- 游戏内实时 YOLO 推理循环
- X2 热键
- 鼠标移动
- 目标选择、追踪和平滑

后续任务见 `TODO.md`。
