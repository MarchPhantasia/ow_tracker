# TODO

## P0 - 训练完成后的模型测试

- [ ] 等当前服务器训练完成。正常训练不要使用 `--use-all-data`，保留 `valid/test` 用来评估模型。

- [ ] 找到训练产物：

```bash
find runs -path "*/weights/best.pt" -print
find runs -path "*/results.csv" -print
```

- [ ] 在验证集上跑指标：

```bash
yolo val \
  model=<BEST_PT> \
  data=runs/datasets/ally_enemy_s/data.yaml \
  imgsz=1280 \
  device=0 \
  split=val
```

- [ ] 在测试集上跑指标：

```bash
yolo val \
  model=<BEST_PT> \
  data=runs/datasets/ally_enemy_s/data.yaml \
  imgsz=1280 \
  device=0 \
  split=test
```

- [ ] 跑 enemy-only 可视化预测：

```bash
python predict.py \
  --weights <BEST_PT> \
  --source datasets/test/images \
  --device 0 \
  --name enemy_test \
  --exist-ok
```

- [ ] 跑 ally/enemy 双类别可视化预测，用来人工检查混淆：

```bash
python predict.py \
  --weights <BEST_PT> \
  --source datasets/test/images \
  --include-ally \
  --device 0 \
  --name ally_enemy_test \
  --exist-ok
```

- [ ] 检查训练输出文件：
  - `results.png`
  - `confusion_matrix.png`
  - `PR_curve.png`
  - `F1_curve.png`
  - `weights/best.pt`

- [ ] 首个 runtime 候选模型的验收标准：
  - `Enemy` precision 足够高，`ally -> enemy` 错误很少。
  - `Enemy` recall 足够高，正常可见敌人不会频繁漏检。
  - `confusion_matrix.png` 没有严重的 `ally -> enemy` 混淆。
  - 可视化预测中没有明显建筑、地图物体被识别成 `enemy`。

- [ ] 接受模型后复制到标准路径：

```bash
mkdir -p models
cp <BEST_PT> models/best.pt
```

- [ ] 只提交代码、配置和文档。不要提交：
  - `datasets/`
  - `runs/`
  - `models/*.pt`

## P1 - 实时 Runtime MVP

- [x] 只按需要加回 runtime 依赖：
  - `dxcam`：Windows 屏幕捕获，放在 `requirements-runtime-windows.txt`。
  - `pywin32`：保留在 runtime 依赖文件里，当前鼠标移动使用 ctypes 调用普通 Windows `SendInput`。
  - `pynput`：未加入，X2 用 Windows `GetAsyncKeyState` 轮询，不额外引入监听依赖。

- [x] 创建 runtime 模块：
  - `src/head_tracker/runtime/config.py`
  - `src/head_tracker/runtime/capture.py`
  - `src/head_tracker/runtime/detector.py`
  - `src/head_tracker/runtime/target_selector.py`
  - `src/head_tracker/runtime/mouse_mover.py`
  - `src/head_tracker/runtime/main.py`

- [x] Runtime 默认行为：
  - 加载 `models/best.pt`。
  - 针对 2560x1440 显示器捕获中心区域：

```yaml
screen_width: 2560
screen_height: 1440
fov_width: 1920
fov_height: 1080
```

  - YOLO 推理默认使用 `imgsz=960`，可在 `config.yaml` 改成 `1280`。
  - 只保留 `enemy` 检测框。
  - `ally` 不进入目标选择。
  - 在 enemy 整体框内估算上半身/头部附近瞄准点：

```python
aim_x = (x1 + x2) * 0.5
aim_y = y1 + (y2 - y1) * 0.30
```

  - 仅按住 X2 时移动鼠标。
  - 松开 X2 或没有有效目标时立即停止移动。

- [x] 添加调试可视化：
  - 绘制捕获区域。
  - 可选绘制 ally/enemy 框。
  - 绘制当前选中的 enemy 和瞄准点。
  - 打印 FPS 和各阶段延迟。

## P2 - Runtime 稳定性和防误锁

- [x] 添加目标选择保护：
  - enemy 最低置信度，初始建议 `0.55`。
  - 最大吸附距离，避免准星附近没有敌人时大幅拉向远处误检。
  - 检测框尺寸合理性检查。
  - 可选检测框宽高比检查。
  - 优先选择离准星最近的 enemy。
  - 当前目标保持锁定，除非另一个 enemy 明显更优。

- [x] 添加时间确认机制：
  - 新目标需要连续出现 2 帧才开始移动。
  - 当前锁定目标允许短暂 1 帧漏检。
  - 持续漏检后快速丢弃目标。

- [x] 添加运动平滑：
  - 对选中的目标点使用 Kalman 或 One Euro filter。
  - 移动目标支持可配置 lead time。
  - 方向反转时立即制动，减少过冲。
  - 准星附近加 deadzone，减少 1-2 像素微抖。

- [x] 添加 runtime 日志：
  - `runs/runtime/latency.csv`
  - 当前目标置信度、类别、框坐标
  - capture / inference / postprocess / mouse 各阶段耗时

- [x] 添加离线 replay 测试模式：
  - 在保存的图片或视频上跑 detector 和 selector。
  - 保存带框输出。
  - replay 模式绝不移动鼠标。

## P3 - 暂缓 / 现在不实现

- [ ] 不实现隐藏输入后端。
- [ ] 不实现反检测或绕过类功能。
- [ ] 在 PyTorch runtime 效果验收前，不做 TensorRT/ONNX 优化。
- [ ] 在 YOLO11s 实测并接受/拒绝前，不做多模型 ensemble。
