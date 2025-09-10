# 🚫 禁用AMP检查下载的完整解决方案

## 问题描述

当运行Ultralytics训练时，会看到：
```
AMP: running Automatic Mixed Precision (AMP) checks...
```

这个检查过程会自动下载 `yolo11n.pt` 模型来测试AMP功能，在离线环境中会导致失败。

## 解决方案

### 方法1：使用离线补丁（推荐）

**步骤1：导入离线补丁**
```python
# 在任何ultralytics导入之前，先导入离线补丁
import offline_patch

# 然后正常导入和使用
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # 使用本地权重
```

**步骤2：离线补丁会自动：**
- ✅ 禁用所有自动下载
- ✅ 替换AMP检查为离线版本
- ✅ 跳过下载测试，直接返回AMP结果
- ✅ 提供详细的错误信息和解决建议

### 方法2：直接禁用AMP

如果不需要AMP功能，可以直接禁用：

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data="your_dataset.yaml",
    epochs=100,
    amp=False  # 直接禁用AMP
)
```

### 方法3：环境变量控制

设置环境变量来控制行为：
```bash
export ULTRALYTICS_OFFLINE=1  # 启用离线模式
export YOLO_AMP=False         # 禁用AMP检查
```

## AMP检查的作用

AMP (Automatic Mixed Precision) 检查的目的：
1. **性能测试**：验证GPU是否支持混合精度训练
2. **兼容性检查**：某些老GPU (如GTX 1660等) 可能有AMP问题
3. **结果验证**：确保AMP不会导致NaN loss或zero-mAP

## 离线模式下的AMP策略

我们的离线补丁采用以下策略：

1. **CPU/MPS设备**：直接返回False (AMP不适用)
2. **已知问题GPU**：返回False并警告
3. **其他CUDA GPU**：返回True (假设支持AMP)
4. **出现问题时**：建议设置 `amp=False`

## 使用示例

### 完整的离线训练示例

```python
# 1. 导入离线补丁
import offline_patch

# 2. 正常使用ultralytics
from ultralytics import YOLO

# 3. 使用预下载的权重
model = YOLO("yolo11n.pt")  # 确保权重文件存在

# 4. 开始训练
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="auto",  # 自动选择设备
    # amp=True,     # 可选：明确启用AMP (离线补丁会跳过下载测试)
    # amp=False,    # 可选：如果遇到问题可以禁用AMP
)
```

### 故障排除

如果仍然遇到AMP相关问题：

1. **检查GPU兼容性**
   ```python
   import torch
   print(f"CUDA可用: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU型号: {torch.cuda.get_device_name()}")
   ```

2. **强制禁用AMP**
   ```python
   model.train(data="dataset.yaml", amp=False)
   ```

3. **检查权重文件**
   ```python
   import simple_weights
   simple_weights.create_offline_config()
   ```

## 文件说明

- `offline_patch.py` - 主要的离线补丁，禁用下载和AMP检查
- `simple_weights.py` - 权重文件管理工具
- `test_amp_disable.py` - 测试AMP禁用功能
- `README_AMP_DISABLE.md` - 本说明文件

## 验证设置

运行测试脚本验证配置：
```bash
python test_amp_disable.py
```

应该看到类似输出：
```
✅ check_amp已被离线版本替换
✅ AMP检查结果: True/False
✅ 没有尝试下载任何模型文件
```

## 注意事项

1. **导入顺序很重要**：必须在导入ultralytics之前导入offline_patch
2. **AMP性能**：如果GPU支持，AMP能显著提升训练速度
3. **兼容性**：老显卡可能需要禁用AMP
4. **监控训练**：注意loss是否正常，如有异常请禁用AMP
