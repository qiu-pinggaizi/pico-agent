# Detection Skill — 目标检测全流程编排

> **Skill ID**: `detection`  
> **Toolset**: `detection`  
> **版本**: 1.0  
> **最后更新**: 2025-01

---

## 1. 触发条件

当用户的请求涉及以下关键词或意图时，激活此 Skill：

| 关键词 / 意图 | 示例 |
|---|---|
| 目标检测 / object detection | "帮我训练一个目标检测模型" |
| YOLO / ultralytics | "用 YOLO 训练我的数据集" |
| MMDetection / mmdet / RTMDet | "用 mmdet 配置一个 RTMDet" |
| 数据标注 / 标注 / annotation | "帮我用 SAM 标注数据" |
| mAP / 评估 / evaluate | "评估一下模型效果" |
| ONNX / TensorRT / 导出 | "把模型导出成 ONNX" |
| 推理 / detect / predict | "用这个模型检测一下图片" |
| 训练监控 / loss / epoch | "训练到哪了？loss 是多少？" |
| bad case / 误检 / 漏检 | "帮我分析一下哪些检测不好" |
| 数据集 / dataset / 拆分 | "帮我拆分数据集" |

---

## 2. 全流程编排策略

### 2.1 标准检测 Pipeline

```
用户需求理解
  │
  ├─→ 数据准备阶段
  │     ├─ dataset_explore      — 探查数据集格式、类别分布
  │     ├─ dataset_split        — 拆分 train/val/test
  │     ├─ dataset_convert      — 格式转换（COCO↔VOC↔YOLO）
  │     ├─ annotation_check     — 标注质量检查
  │     ├─ annotation_visualize — 标注可视化验证
  │     └─ annotation_convert   — 标注格式互转
  │
  ├─→ 模型配置阶段
  │     ├─ recommend_config     — 自动推荐超参
  │     ├─ yolo_config          — 生成 YOLO data.yaml
  │     └─ mmdet_config         — 生成 MMDet config
  │
  ├─→ 训练阶段
  │     ├─ train_start          — 启动训练
  │     ├─ train_monitor        — 监控训练进度
  │     ├─ train_analyze        — 分析训练趋势（过拟合/欠拟合/平台期）
  │     └─ train_resume         — 断点续训
  │
  ├─→ 评估阶段
  │     ├─ evaluate             — mAP / precision / recall
  │     ├─ pr_curve             — PR 曲线
  │     ├─ confusion_matrix     — 混淆矩阵
  │     ├─ bad_cases            — Bad case 分析
  │     └─ compare_models       — 多模型对比
  │
  └─→ 部署阶段
        ├─ export_onnx          — 导出 ONNX
        ├─ export_tensorrt      — 导出 TensorRT
        └─ benchmark            — 速度基准测试
```

### 2.2 编排原则

1. **阶段性推进**: 每个阶段完成后向用户确认，再进入下一阶段。
2. **智能推荐**: 根据数据集大小、GPU 显存自动推荐模型大小和超参。
3. **远程优先**: 当用户指定 `server` 参数时，所有操作通过 SSH 在远程执行。
4. **错误恢复**: 训练中断时自动检测 checkpoint 并建议续训。
5. **渐进式详细**: 初次对话给出简洁建议，深入交互时给出详细分析。

---

## 3. YOLO + MMDetection 双框架最佳实践

### 3.1 YOLO (ultralytics) 推荐配置

#### 模型选择指南

| 数据规模 | GPU 显存 | 推荐模型 | 批大小 | 轮数 |
|---|---|---|---|---|
| < 500 张 | ≤ 6 GB | yolov8n / yolo11n | 8-16 | 300 |
| 500-2K 张 | 8 GB | yolov8s / yolo11s | 16 | 300 |
| 2K-10K 张 | 8-12 GB | yolov8m / yolo11m | 8-16 | 200 |
| 10K-50K 张 | 16+ GB | yolov8l / yolo11l | 4-8 | 150 |
| > 50K 张 | 24+ GB | yolov8x / yolo11x | 2-4 | 100-150 |

#### YOLO 训练命令

```bash
# 基础训练
yolo train data=data.yaml model=yolo11s.pt epochs=200 batch=16 imgsz=640 lr0=0.01

# 分布式训练（多 GPU）
yolo train data=data.yaml model=yolo11s.pt epochs=200 device=0,1,2,3

# 断点续训
yolo train resume model=path/to/last.pt epochs=50

# 评估
yolo val model=best.pt data=data.yaml split=val plots=True

# 单图推理
yolo predict model=best.pt source=image.jpg conf=0.25 save=True

# 导出 ONNX
yolo export model=best.pt format=onnx opset=17 dynamic=True simplify=True

# 导出 TensorRT
yolo export model=best.pt format=engine half=True workspace=4
```

#### YOLO 训练日志解析

- **位置**: `runs/detect/train*/results.csv`
- **关键列**:
  - `epoch`: 当前轮次
  - `train/box_loss`, `train/cls_loss`, `train/dfl_loss`: 训练损失
  - `val/box_loss`, `val/cls_loss`, `val/cls_loss`: 验证损失
  - `metrics/precision(B)`, `metrics/recall(B)`: 精度和召回
  - `metrics/mAP50(B)`, `metrics/mAP50-95(B)`: mAP 指标

### 3.2 MMDetection 推荐配置

#### 模型选择指南

| 场景 | 推荐模型 | 配置文件 |
|---|---|---|
| 快速原型 | RTMDet-tiny | rtmdet_tiny_8xb32-300e_coco.py |
| 精度平衡 | RTMDet-s | rtmdet_s_8xb32-300e_coco.py |
| 高精度 | RTMDet-l | rtmdet_l_8xb32-300e_coco.py |
| 两阶段经典 | Faster R-CNN | faster-rcnn_r50_fpn_1x_coco.py |
| Transformer | Deformable DETR | deformable-detr_r50_16xb2-50e_coco.py |

#### MMDet 训练命令

```bash
# 单 GPU 训练
python tools/train.py config.py

# 分布式训练
bash tools/dist_train.sh config.py ${GPU_NUM} --work-dir work_dirs/exp

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh config.py 2

# 断点续训
bash tools/dist_train.sh config.py 1 --resume-from work_dirs/exp/latest.pth

# 评估
python tools/test.py config.py work_dirs/exp/best_coco_bbox_mAP_epoch_100.pth --eval bbox

# 推理
python demo/image_demo.py image.jpg config.py --weights best.pth --out-dir outputs/
```

#### MMDet 训练日志解析

- **位置**: `work_dirs/*/log.json`（每行一个 JSON 对象）
- **关键字段**:
  - `epoch`: 轮次
  - `loss`, `loss_cls`, `loss_bbox`, `loss_iou`: 损失值
  - `coco/bbox_mAP`, `coco/bbox_mAP_50`, `coco/bbox_mAP_75`: mAP

---

## 4. 远程服务器工作流

### 4.1 前置条件

在 `~/.pico/config.yaml` 中配置远程服务器：

```yaml
servers:
  gpu-server:
    host: 192.168.1.100
    port: 22
    username: user
    key_file: ~/.ssh/id_rsa
    work_dir: /data/projects
```

### 4.2 远程工作流

```
1. 本地准备数据集 → 上传至远程服务器
2. 本地生成配置文件 → 上传至远程
3. 远程启动训练（background=True）
4. 本地定期调用 train_monitor 查看进度
5. 训练完成后 → 本地调用 evaluate 评估
6. 本地调用 export_onnx / export_tensorrt 导出
7. 下载模型到本地
```

### 4.3 远程参数传递

所有工具都支持 `server` 参数：
- `server=""` (默认): 本地执行
- `server="gpu-server"`: 通过 SSH 在远程执行

示例：
```python
# 本地执行
train_start(config_path="data.yaml", framework="yolo")

# 远程执行
train_start(config_path="/data/projects/my_project/data.yaml", framework="yolo", server="gpu-server")
```

---

## 5. SAM 标注工作流

### 5.1 流程概述

```
原始图片 → SAM 零样本检测/分割 → YOLO/COCO 格式标注 → 人工审核修正 → 训练
```

### 5.2 使用方法

```python
# 使用文本提示的零样本标注
sam_annotate(
    image_dir="/data/raw_images",
    output_dir="/data/annotations",
    prompts=["cat", "dog", "person"],
    model="sam_b"
)

# 检查标注质量
annotation_check(data_dir="/data/annotations", format="yolo")

# 可视化标注结果
annotation_visualize(
    image_dir="/data/annotations/images",
    label_dir="/data/annotations/labels",
    output_dir="/data/vis_results",
    max_images=100
)
```

### 5.3 最佳实践

1. **逐步标注**: 先标注少量图片验证效果，再批量标注。
2. **prompt 优化**: 使用具体的物体名称而非笼统描述。
3. **人工审核**: SAM 标注结果需要人工审核和修正。
4. **类别一致性**: 确保 SAM 检测的类别与最终训练的类别映射一致。

---

## 6. 分布式训练指导

### 6.1 YOLO 多 GPU 训练

```bash
# 自动分配 GPU
yolo train data=data.yaml model=yolo11s.pt device=0,1,2,3

# 通过工具调用
train_start(
    config_path="data.yaml",
    framework="yolo",
    gpu="0,1,2,3"
)
```

### 6.2 MMDet 分布式训练

```bash
# 4 GPU 分布式
bash tools/dist_train.sh config.py 4 --work-dir work_dirs/exp_4gpu

# 通过工具调用
train_start(
    config_path="config.py",
    framework="mmdet",
    gpu="0,1,2,3",
    work_dir="work_dirs/exp_4gpu"
)
```

### 6.3 注意事项

- 学习率应随 GPU 数量线性缩放：`lr = base_lr × num_gpus`
- 批大小同理：`batch = base_batch × num_gpus`
- MMDet 需要在 config 中设置 `auto_scale_lr` 以自动缩放

---

## 7. 常见问题诊断表

| 问题 | 可能原因 | 解决方案 |
|---|---|---|
| `CUDA out of memory` | 批大小太大 | 减小 batch_size；使用更小模型；减小 imgsz |
| `loss` 不下降 | 学习率过小 / 数据问题 | 提高 lr0；检查数据标注是否正确 |
| `loss` 先降后升 | 过拟合 | 增加数据增强；降低 lr；使用早停 |
| `val loss` 上升而 `train loss` 下降 | 过拟合 | 增加 dropout / weight_decay；减少训练轮数 |
| `mAP` 很低 | 数据质量 / 模型容量不足 | 检查标注质量；增大模型；增加训练轮数 |
| `mAP` 平台期 | 模型能力饱和 | 更大模型；更大输入分辨率；更强增强 |
| 漏检（False Negative） | 阈值太高 / 训练不足 | 降低 conf 阈值；继续训练 |
| 误检（False Positive） | 阈值太低 / 标注问题 | 提高 conf 阈值；检查是否有漏标 |
| 类别混淆 | 类别间相似 / 标注不一致 | 增加该类训练数据；检查标注一致性 |
| MMDet config 报错 | 配置文件路径错误 | 检查 `_base_` 路径；使用绝对路径 |
| 远程训练断连 | SSH 超时 / 网络中断 | 使用 `nohup` 或 `tmux`；调大 SSH timeout |
| ONNX 导出失败 | 算子不支持 | 更新 ultralytics；降低 opset version |
| TensorRT 导出失败 | 版本不兼容 | 检查 TensorRT 版本；使用对应 ONNX opset |

---

## 8. 用户交互示例

### 示例 1: 完整训练流程

```
用户: 我有一个猫狗数据集，帮我训练一个 YOLO 模型

Agent:
好的！让我帮你完成整个流程。首先让我看看你的数据集结构。

→ 调用 dataset_explore(data_dir="/path/to/dataset")

[返回数据集信息]

你的数据集有 2000 张图片，2 个类别（cat, dog），已经是 YOLO 格式。
让我为你推荐训练参数：

→ 调用 recommend_config(data_dir="/path/to/dataset", framework="yolo")

[返回推荐]

基于你的数据规模和 GPU（RTX 3080 10GB），我推荐：
- 模型: yolo11s.pt
- 批大小: 16
- 训练轮数: 200
- 学习率: 0.01

要我开始训练吗？
```

### 示例 2: 训练监控

```
用户: 训练到哪了？

Agent:
→ 调用 train_monitor(project_dir="runs/detect/train3", framework="yolo")

当前训练进度：
- 轮次: 87/200
- mAP50: 0.823
- mAP50-95: 0.612
- train/box_loss: 0.0234 (持续下降 ✓)
- val/box_loss: 0.0312 (稳定 ✓)

→ 调用 train_analyze(project_dir="runs/detect/train3", framework="yolo")

训练状态：健康 ✅
mAP 持续提升中，预计还可继续提升。建议继续训练至 200 轮。
```

### 示例 3: Bad Case 分析

```
用户: 帮我看看模型哪里检测不好

Agent:
→ 调用 bad_cases(model_path="runs/detect/train3/weights/best.pt",
                  data_dir="/path/to/dataset",
                  output_dir="bad_case_analysis")

分析完成！在 300 张验证图片中发现：
- False Positive (误检): 23 个
- False Negative (漏检): 15 个
- Misclassification (分类错误): 8 个

主要问题：
1. 小目标漏检较多（60% 的漏检发生在小物体上）
2. 光照不足时容易误检
3. 猫和狗的分类偶尔混淆

建议：
- 增加小目标训练数据或使用更大输入分辨率 (imgsz=1280)
- 增加弱光环境数据增强
- 增加易混淆类别的困难样本
```

### 示例 4: 远程服务器训练

```
用户: 帮我在 gpu-server 上训练，数据在 /data/cat_dog

Agent:
好的，我将在远程服务器 gpu-server 上执行训练。

→ 调用 dataset_explore(data_dir="/data/cat_dog", server="gpu-server")
→ 调用 recommend_config(data_dir="/data/cat_dog", framework="yolo", server="gpu-server")
→ 调用 yolo_config(data_dir="/data/cat_dog", output_path="/data/cat_dog/data.yaml", server="gpu-server")
→ 调用 train_start(config_path="/data/cat_dog/data.yaml", framework="yolo", server="gpu-server", background=True)

训练已在远程服务器后台启动！

你可以随时问我"训练到哪了"来查看进度。
```

### 示例 5: 模型导出与部署

```
用户: 把 best.pt 导出成 ONNX 和 TensorRT

Agent:
→ 调用 export_onnx(model_path="runs/detect/train3/weights/best.pt", opset=17, dynamic=True)

✅ ONNX 导出成功！
- 路径: runs/detect/train3/weights/best.onnx
- 大小: 22.4 MB
- Opset: 17
- 动态批大小: 是

→ 调用 export_tensorrt(model_path="runs/detect/train3/weights/best.pt", half=True, workspace=4)

✅ TensorRT 导出成功！
- 路径: runs/detect/train3/weights/best.engine
- 大小: 18.7 MB
- FP16: 是

→ 调用 benchmark(model_path="runs/detect/train3/weights/best.pt")

速度基准测试（RTX 3080, imgsz=640）:
| 批大小 | 延迟(ms) | 吞吐(FPS) | 单图(ms) |
|--------|----------|-----------|----------|
| 1      | 8.2      | 122.0     | 8.2      |
| 4      | 21.5     | 186.0     | 5.4      |
| 8      | 38.9     | 205.7     | 4.9      |
| 16     | 72.3     | 221.3     | 4.5      |

最大吞吐量: ~221 FPS (batch=16)
```

---

## 9. 工具注册表

本 Skill 包含以下已注册工具：

| 工具名 | 文件 | 说明 |
|---|---|---|
| `dataset_explore` | dataset.py | 数据集探查 |
| `dataset_split` | dataset.py | 数据集拆分 |
| `dataset_convert` | dataset.py | 数据集格式转换 |
| `sam_annotate` | annotation.py | SAM 自动标注 |
| `annotation_check` | annotation.py | 标注质量检查 |
| `annotation_visualize` | annotation.py | 标注可视化 |
| `annotation_convert` | annotation.py | 标注格式转换 |
| `yolo_config` | config_gen.py | YOLO 配置生成 |
| `mmdet_config` | config_gen.py | MMDet 配置生成 |
| `recommend_config` | config_gen.py | 超参推荐 |
| `train_start` | trainer.py | 启动训练 |
| `train_monitor` | trainer.py | 训练监控 |
| `train_analyze` | trainer.py | 训练分析 |
| `train_resume` | trainer.py | 断点续训 |
| `evaluate` | evaluator.py | 模型评估 |
| `pr_curve` | evaluator.py | PR 曲线 |
| `confusion_matrix` | evaluator.py | 混淆矩阵 |
| `bad_cases` | evaluator.py | Bad case 分析 |
| `compare_models` | evaluator.py | 多模型对比 |
| `inference_image` | inference.py | 单图推理 |
| `inference_batch` | inference.py | 批量推理 |
| `inference_video` | inference.py | 视频推理 |
| `export_onnx` | exporter.py | ONNX 导出 |
| `export_tensorrt` | exporter.py | TensorRT 导出 |
| `benchmark` | exporter.py | 速度基准测试 |
