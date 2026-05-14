# Pico Agent — 目标检测全流程自动化 Agent

## 一、系统总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户层 (CLI / API)                     │
│  "帮我训练一个 YOLOv8 检测模型，数据在 ./datasets/traffic/"     │
├─────────────────────────────────────────────────────────────┤
│                    Pico Agent 核心层                           │
│  对话循环 · 工具调度 · 上下文管理 · 记忆 · 子Agent委派           │
├─────────────────────────────────────────────────────────────┤
│               领域技能层 (Detection Skill)                     │
│  流程编排提示词 · 阶段判断 · 最佳实践知识库                      │
├─────────────────────────────────────────────────────────────┤
│                   检测专用工具层                                │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ 数据管理  │ │ 模型训练  │ │ 评估分析  │ │ 部署导出  │        │
│  │ tools    │ │ tools    │ │ tools    │ │ tools    │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ 可视化   │ │ 标注工具  │ │ 配置生成  │                     │
│  │ tools    │ │ tools    │ │ tools    │                     │
│  └──────────┘ └──────────┘ └──────────┘                     │
├─────────────────────────────────────────────────────────────┤
│                 基础工具层（通用 Agent）                        │
│  file_ops · terminal · web_search · vision                   │
└─────────────────────────────────────────────────────────────┘
```

## 二、目标检测全流程（Agent 自动编排）

Agent 根据用户意图自动判断当前阶段，引导或直接执行：

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 1. 数据   │───▶│ 2. 配置   │───▶│ 3. 训练   │───▶│ 4. 评估   │
│ 准备      │    │ 生成      │    │ 监控      │    │ 分析      │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │                                                │
     │  ┌──────────┐    ┌──────────┐                  │
     └─▶│ 0. 标注   │    │ 5. 推理   │◀────────────────┘
        │ 辅助      │    │ 测试      │
        └──────────┘    └──────────┘
                            │
                        ┌──────────┐
                        │ 6. 导出   │
                        │ 部署      │
                        └──────────┘
```

### 阶段详解

| 阶段 | Agent 自动完成的工作 |
|------|---------------------|
| **0. 标注辅助** | 调用 SAM 自动标注、格式转换（COCO↔VOC↔YOLO）、标注质量检查 |
| **1. 数据准备** | 数据集探索、统计分析、自动划分 train/val/test、数据增强配置、YAML 生成 |
| **2. 配置生成** | 根据数据集自动生成训练配置（模型选择、超参数、数据路径） |
| **3. 训练监控** | 启动训练、实时监控 loss/mAP、自动 early stopping、断点续训 |
| **4. 评估分析** | mAP/PR 曲线、混淆矩阵、bad case 分析、逐类别统计 |
| **5. 推理测试** | 单图/批量/视频推理、结果可视化、置信度分析 |
| **6. 导出部署** | ONNX/TensorRT/CoreML 导出、速度基准测试、部署包生成 |

## 三、新增项目结构

在原 pico-agent 基础上增加：

```
pico-agent/
├── pico/
│   ├── agent.py             # 核心对话循环（已有）
│   ├── llm.py               # LLM 适配（已有）
│   ├── tools/
│   │   ├── ...              # 通用工具（已有）
│   │   └── detection/       # ★ 新增：检测专用工具
│   │       ├── __init__.py
│   │       ├── dataset.py       # 数据集探索、统计、划分
│   │       ├── annotation.py    # 标注格式转换、质量检查、SAM辅助
│   │       ├── config_gen.py    # 自动生成 YOLO/MMDet 训练配置
│   │       ├── trainer.py       # 训练启动、监控、断点续训
│   │       ├── evaluator.py     # mAP 计算、PR 曲线、混淆矩阵
│   │       ├── inference.py     # 推理、结果可视化
│   │       └── exporter.py      # ONNX/TensorRT 导出
│   └── skills/
│       └── detection.md     # ★ 检测全流程技能文档（提示词知识库）
```

## 四、各工具详细设计

### 4.1 dataset.py — 数据集工具

```python
# 工具 1: 数据集探索
dataset_explore(path: str) -> dict
# 返回: {"total_images": 5000, "total_annotations": 12340,
#         "num_classes": 5, "classes": ["car","person",...],
#         "class_distribution": {"car": 8000, "person": 4340, ...},
#         "format": "yolo",  # or "coco", "voc"
#         "image_sizes": {"min": [640,480], "max": [1920,1080], "avg": [1280,720]},
#         "has_train_val_test": true,
#         "issues": ["class 'bicycle' has only 10 annotations (< 50 minimum)"]}

# 工具 2: 数据集划分
dataset_split(path: str, ratios: str = "8:1:1", seed: int = 42) -> dict
# 自动按比例划分 train/val/test，保持类别平衡（分层采样）

# 工具 3: 格式转换
dataset_convert(source_path: str, source_format: str, target_format: str) -> dict
# 支持: coco ↔ voc ↔ yolo ↔ custom_json

# 工具 4: 数据增强配置
dataset_augment_config(task_type: str = "detection") -> str
# 生成 albumentations / yolov8 增强配置 YAML
```

### 4.2 annotation.py — 标注工具

```python
# 工具 5: SAM 自动标注
annotation_sam(images_dir: str, points_or_boxes: str = "auto") -> dict
# 使用 SAM/SAM2 对图片自动生成分割标注

# 工具 6: 标注质量检查
annotation_check(annotations_path: str) -> dict
# 返回: {"issues": [...], "duplicate_boxes": 3, "tiny_boxes": 15,
#         "out_of_bounds": 2, "empty_images": 5}

# 工具 7: 标注可视化
annotation_visualize(image_path: str, annotation_path: str) -> str
# 在图片上绘制标注框，保存到临时文件，返回路径
```

### 4.3 config_gen.py — 配置生成

```python
# 工具 8: YOLO 配置生成
config_yolo(dataset_path: str, model_size: str = "n", task: str = "detect") -> str
# 生成 data.yaml + 训练命令
# model_size: n(ano), s(mall), m(edium), l(arge), x(large)
# task: detect, segment, classify, pose

# 工具 9: MMDetection 配置生成
config_mmdet(dataset_path: str, model: str = "rtmdet_tiny") -> str
# 生成 mmdet 配置文件

# 工具 10: 超参数推荐
config_recommend(dataset_stats: dict, gpu_info: dict) -> dict
# 根据数据集大小、GPU 显存自动推荐 batch_size, lr, epochs 等
```

### 4.4 trainer.py — 训练工具

```python
# 工具 11: 启动训练
train_start(config_path: str, framework: str = "yolo", gpu: str = "0") -> dict
# 后台启动训练，返回 {"pid": 12345, "log_path": "...", "project_dir": "..."}

# 工具 12: 训练监控
train_monitor(project_dir: str) -> dict
# 返回: {"epoch": 50, "total_epochs": 100, "train_loss": 0.023,
#         "val_mAP50": 0.85, "val_mAP50_95": 0.67,
#         "lr": 0.001, "eta": "2h30m", "best_epoch": 45,
#         "status": "running"}  # or "completed", "error"

# 工具 13: 训练日志分析
train_analyze(project_dir: str) -> dict
# 返回: {"converged": true, "overfitting": false, "best_mAP": 0.72,
#         "suggestions": ["Consider increasing augmentation", ...]}

# 工具 14: 断点续训
train_resume(checkpoint_path: str, additional_epochs: int = 50) -> dict
```

### 4.5 evaluator.py — 评估工具

```python
# 工具 15: 模型评估
eval_model(weights: str, dataset_yaml: str, split: str = "val") -> dict
# 返回: {"mAP50": 0.85, "mAP50_95": 0.67, "precision": 0.82,
#         "recall": 0.78, "per_class": {...}, "confusion_matrix": "path/to/cm.png"}

# 工具 16: PR 曲线生成
eval_pr_curve(weights: str, dataset_yaml: str) -> str
# 返回生成的 PR 曲线图片路径

# 工具 17: Bad case 分析
eval_bad_cases(weights: str, dataset_yaml: str, threshold: float = 0.3) -> dict
# 返回: {"false_positives": [...], "false_negatives": [...],
#         "worst_classes": [...], "visualization_paths": [...]}

# 工具 18: 模型对比
eval_compare(weights_list: list[str], dataset_yaml: str) -> dict
# 多模型对比表格: 各模型的 mAP、速度、参数量
```

### 4.6 inference.py — 推理工具

```python
# 工具 19: 单图推理
infer_image(weights: str, image_path: str, conf: float = 0.25) -> dict
# 返回: {"detections": [...], "annotated_image": "path/to/result.jpg"}

# 工具 20: 批量推理
infer_batch(weights: str, images_dir: str, output_dir: str, conf: float = 0.25) -> dict
# 返回: {"total_images": 100, "total_detections": 342, "output_dir": "..."}

# 工具 21: 视频推理
infer_video(weights: str, video_path: str, output_path: str, conf: float = 0.25) -> dict
# 返回: {"output_path": "...", "fps": 30, "frames": 900, "detections_total": 5432}

# 工具 22: 摄像头实时推理
infer_camera(weights: str, camera_id: int = 0, duration: int = 30) -> dict
# 实时展示检测结果（需要 GUI 环境）
```

### 4.7 exporter.py — 导出工具

```python
# 工具 23: ONNX 导出
export_onnx(weights: str, imgsz: int = 640, simplify: bool = True) -> dict
# 返回: {"onnx_path": "...", "size_mb": 12.5, "input_shape": [1,3,640,640]}

# 工具 24: TensorRT 导出
export_tensorrt(weights: str, half: bool = True, workspace: int = 4) -> dict
# 返回: {"engine_path": "...", "size_mb": 25.0, "precision": "fp16"}

# 工具 25: 速度基准测试
export_benchmark(weights: str, imgsz: int = 640, n_iters: int = 100) -> dict
# 返回: {"latency_ms": 8.5, "throughput_fps": 117, "gpu_memory_mb": 1024}
```

## 五、检测全流程 Skill 文档

创建 `skills/detection.md`，作为 Agent 的领域知识库：

```markdown
# 目标检测全流程自动化

## 触发条件
用户提到：检测、目标检测、YOLO、训练模型、mAP、数据集、标注、推理、导出 等关键词时加载此技能。

## 流程编排策略
1. 先用 dataset_explore 了解数据现状
2. 如有标注问题，用 annotation_check 检查
3. 用 config_recommend 获取超参数建议
4. 生成配置文件
5. 启动训练并监控
6. 评估分析 + bad case
7. 导出部署

## 最佳实践
- 小数据集（<1000张）：先用预训练模型 fine-tune，epochs 50-100
- 大数据集（>10000张）：可从头训练，epochs 200-300
- 类别不平衡：使用 focal loss 或对少数类做 oversampling
- mAP50 > 0.5 算初步可用，> 0.75 算较好，> 0.9 算优秀
- 训练 loss 降但 val loss 升 → 过拟合，需增加数据或正则化

## 常见问题诊断
| 症状 | 可能原因 | 建议 |
|------|---------|------|
| mAP 很低 | 数据质量差 / 类别太少 | 检查标注、增加数据 |
| 过拟合 | 数据量不足 / 模型太大 | 增强、正则化、换小模型 |
| 推理速度慢 | 模型太大 | 用更小模型或 TensorRT |
| 漏检多 | 置信度阈值太高 | 降低 conf threshold |
| 误检多 | 置信度阈值太低 | 提高 conf / 增加负样本 |
```

## 六、典型用户交互流程

```
用户: 帮我训练一个 YOLO 目标检测模型，数据在 ./datasets/coco128/

Agent: 好的，我先探索一下你的数据集情况。
       [调用 dataset_explore]
       
       数据集概况：
       - 128 张图片，649 个标注
       - 格式：YOLO
       - 4 个类别：person(200), car(250), bicycle(120), dog(79)
       - ⚠️ dog 类别样本较少（79个），建议做数据增强
       
       推荐配置：
       - 模型：YOLOv8s（你的数据集较小，用 small 模型平衡速度和精度）
       - Batch size：16
       - Epochs：100
       - 学习率：0.01（余弦退火）
       
       我来生成配置并开始训练？

用户: 好的，开始吧

Agent: [调用 config_yolo → 生成 data.yaml]
       [调用 train_start → 后台启动训练]
       
       训练已启动（PID: 12345）
       日志：./runs/detect/train/train.log
       
       我每 10 分钟自动检查一次进度。你也可以随时问我"训练进度如何？"

用户: 训练怎么样了？

Agent: [调用 train_monitor]
       
       当前进度：Epoch 67/100 (67%)
       - train/box_loss: 0.023
       - val/mAP50: 0.823
       - val/mAP50-95: 0.614
       - 最佳结果在 Epoch 62（mAP50=0.831）
       - 预计剩余时间：1h15m
       
       分析：模型仍在稳步提升，未见明显过拟合。

用户: 训练完了，评估一下效果

Agent: [调用 eval_model → 全量评估]
       [调用 eval_bad_cases → 分析错误案例]
       
       评估结果：
       ┌──────────┬────────┬──────────┬───────────┐
       │ 类别      │ mAP50  │ Precision│ Recall    │
       ├──────────┼────────┼──────────┼───────────┤
       │ person   │ 0.89   │ 0.87     │ 0.82      │
       │ car      │ 0.91   │ 0.90     │ 0.88      │
       │ bicycle  │ 0.78   │ 0.75     │ 0.71      │
       │ dog      │ 0.72   │ 0.68     │ 0.65      │  ← 最弱
       └──────────┴────────┴──────────┴───────────┘
       整体 mAP50-95: 0.672
       
       Bad case 分析：
       - dog 类漏检率最高（35%），主要在遮挡场景
       - person 与 bicycle 存在混淆（互相误检 12 次）
       
       建议：增加遮挡场景的训练数据，或使用更强的数据增强。
       需要我导出 ONNX 模型吗？

用户: 导出 ONNX，然后测一下速度

Agent: [调用 export_onnx → 导出]
       [调用 export_benchmark → 测速]
       
       导出完成：
       - ONNX 文件：./runs/detect/train/weights/best.onnx（24.5 MB）
       
       速度测试（640×640, 100次迭代）：
       - 平均延迟：8.3ms
       - 吞吐量：120 FPS
       - GPU 显存占用：1.2GB
       
       全流程完成！ 🎉
```

## 七、依赖

在原 requirements.txt 基础上新增：
```
ultralytics>=8.0       # YOLOv8/v11 框架
onnx>=1.14             # ONNX 导出
onnxruntime>=1.16      # ONNX 推理（可选）
opencv-python>=4.8     # 图像处理
Pillow>=10.0           # 图像处理
matplotlib>=3.7        # 可视化
seaborn>=0.12          # 混淆矩阵
pandas>=2.0            # 数据分析
numpy>=1.24            # 数值计算
```

## 八、开发优先级

| 优先级 | 模块 | 说明 |
|--------|------|------|
| P0 | dataset_explore + config_gen + train_start + train_monitor + eval_model | 核心流程：数据→配置→训练→评估 |
| P1 | inference + exporter + eval_bad_cases | 推理和导出 |
| P2 | annotation tools + dataset_convert | 标注辅助 |
| P3 | 模型对比、摄像头推理 | 高级功能 |
