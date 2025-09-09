# Ultralytics 离线使用指南

## 概述

在离线环境中使用 Ultralytics 库而不触发任何下载的完整解决方案。

## 方法总结

### 方法1：预下载权重文件（推荐）

**步骤1：在有网络的环境中下载权重**
```bash
# 运行预下载脚本
python download_weights_offline.py
```

**步骤2：将权重文件复制到离线环境**
权重文件默认保存在 `~/.config/Ultralytics/weights/` 目录下，将整个目录复制到离线环境的相同位置。

**步骤3：在离线环境中正常使用**
```python
from ultralytics import YOLO

# 这样使用时会优先查找本地权重文件
model = YOLO("yolo11n.pt")
```

### 方法2：使用离线补丁

在你的代码开头导入离线补丁：

```python
# 导入离线补丁（必须在导入YOLO之前）
import offline_patch

from ultralytics import YOLO

# 现在如果权重不存在会报错而不是下载
try:
    model = YOLO("yolo11n.pt")
except FileNotFoundError as e:
    print(f"权重文件不存在: {e}")
```

### 方法3：从头开始训练

```python
from ultralytics import YOLO

# 使用YAML配置文件，不加载预训练权重
model = YOLO("yolo11n.yaml")  # 注意是.yaml而不是.pt

# 训练时明确禁用预训练
results = model.train(
    data="your_dataset.yaml",
    epochs=100,
    pretrained=False,  # 重要：禁用预训练
    imgsz=640
)
```

## 权重文件位置

Ultralytics 会按以下顺序查找权重文件：

1. 当前工作目录
2. `~/.config/Ultralytics/weights/` （默认权重目录）
3. 如果都没找到，才会尝试下载

## 常用权重文件列表

### YOLO11 系列
- **检测**: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- **分割**: `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo11l-seg.pt`, `yolo11x-seg.pt`  
- **分类**: `yolo11n-cls.pt`, `yolo11s-cls.pt`, `yolo11m-cls.pt`, `yolo11l-cls.pt`, `yolo11x-cls.pt`
- **姿态**: `yolo11n-pose.pt`, `yolo11s-pose.pt`, `yolo11m-pose.pt`, `yolo11l-pose.pt`, `yolo11x-pose.pt`

### 其他常用模型
- **SAM**: `sam_b.pt`, `sam_l.pt`
- **FastSAM**: `FastSAM-x.pt`, `FastSAM-s.pt`
- **YOLOv8**: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`

## 实际使用示例

### 离线训练示例
```python
# 确保离线模式
import offline_patch

from ultralytics import YOLO

# 方案1：使用预下载的权重
model = YOLO("yolo11n.pt")  # 确保该文件已预下载

# 方案2：从头开始训练
model = YOLO("yolo11n.yaml")

# 训练
results = model.train(
    data="coco8.yaml", 
    epochs=100,
    pretrained=False
)
```

### 离线推理示例
```python
import offline_patch
from ultralytics import YOLO

# 使用预训练模型进行推理
model = YOLO("yolo11n.pt")
results = model("image.jpg")
```

## 文件结构

```
your_project/
├── offline_patch.py              # 离线补丁
├── download_weights_offline.py   # 预下载脚本  
├── offline_training_example.py   # 离线训练示例
└── README_OFFLINE.md            # 本文档
```

## 注意事项

1. **预下载是最可靠的方法**：提前下载所需权重文件到正确位置
2. **检查权重目录**：确保 `~/.config/Ultralytics/weights/` 目录存在且有权限
3. **使用 .yaml 文件**：如果不需要预训练权重，直接使用配置文件
4. **设置 pretrained=False**：在训练时明确禁用预训练权重加载
5. **离线补丁**：在代码开头导入 `offline_patch` 来禁用所有下载逻辑

## 故障排除

如果遇到下载相关错误：

1. 检查权重文件是否存在于正确位置
2. 确认文件名是否正确（包括扩展名）
3. 使用离线补丁获得更清晰的错误信息
4. 考虑使用 .yaml 配置文件而不是 .pt 权重文件
