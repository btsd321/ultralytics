"""
离线模式补丁 - 禁用Ultralytics的自动下载功能
在离线环境中导入此模块来禁用所有下载逻辑
"""

import logging
from pathlib import Path
from ultralytics.utils import LOGGER

# 保存原始函数的引用
_original_attempt_download_asset = None
_original_check_amp = None

def disable_downloads():
    """禁用所有下载功能"""
    global _original_attempt_download_asset, _original_check_amp
    
    # 导入需要修改的模块
    from ultralytics.utils import downloads
    from ultralytics.nn import tasks
    from ultralytics.utils import checks
    
    # 保存原始函数
    if _original_attempt_download_asset is None:
        _original_attempt_download_asset = downloads.attempt_download_asset
    
    def no_download_attempt_download_asset(file, **kwargs):
        """替代attempt_download_asset的离线版本"""
        file_path = Path(file)
        
        # 1. 如果是绝对路径且文件存在，直接返回
        if file_path.is_absolute() and file_path.exists():
            LOGGER.info(f"📁 找到本地文件: {file_path}")
            return str(file_path)
        
        # 2. 检查当前目录
        if file_path.exists():
            LOGGER.info(f"📁 找到当前目录文件: {file_path.absolute()}")
            return str(file_path)
        
        # 3. 检查权重目录
        from ultralytics.utils import SETTINGS
        weights_file = Path(SETTINGS["weights_dir"]) / file_path.name
        if weights_file.exists():
            LOGGER.info(f"📁 找到权重目录文件: {weights_file}")
            return str(weights_file)
        
        # 4. 检查常见的权重存放位置
        common_locations = [
            Path.cwd() / "weights" / file_path.name,
            Path.cwd() / "models" / file_path.name,
            Path.home() / ".ultralytics" / "weights" / file_path.name,
            Path("/tmp/ultralytics/weights") / file_path.name,
        ]
        
        for location in common_locations:
            if location.exists():
                LOGGER.info(f"📁 找到权重文件: {location}")
                return str(location)
        
        # 5. 如果文件不存在，提供详细的错误信息
        search_locations = [
            f"当前目录: {file_path.absolute()}",
            f"权重目录: {weights_file.absolute()}",
        ] + [f"常见位置: {loc}" for loc in common_locations]
        
        raise FileNotFoundError(
            f"🚫 离线模式: 无法找到权重文件 '{file_path.name}'。\n"
            f"已搜索以下位置:\n" + 
            "\n".join(f"  • {loc}" for loc in search_locations) + 
            f"\n\n💡 解决方法:\n"
            f"  1. 使用 download_weights_offline.py 脚本预下载权重\n"
            f"  2. 手动将权重文件放到 {weights_file.parent}\n"
            f"  3. 使用相对路径指向现有文件\n"
            f"  4. 使用 YAML 配置文件从头开始训练"
        )
    
    # 保存并替换 check_amp 函数
    if _original_check_amp is None:
        _original_check_amp = checks.check_amp
    
    def no_download_check_amp(model):
        """离线版本的AMP检查 - 跳过下载直接返回结果"""
        device = next(model.parameters()).device if hasattr(model, 'parameters') and any(model.parameters()) else None
        
        if device is None:
            LOGGER.warning("🚫 AMP检查跳过: 无法获取模型设备")
            return False
            
        if device.type in {"cpu", "mps"}:
            LOGGER.info("🚫 AMP检查跳过: AMP仅在CUDA设备上使用")
            return False  # AMP only used on CUDA devices
        
        # 检查GPU是否有已知的AMP问题
        try:
            import torch
            import re
            gpu = torch.cuda.get_device_name(device)
            pattern = re.compile(
                r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", 
                re.IGNORECASE
            )
            
            if bool(pattern.search(gpu)):
                LOGGER.warning(f"🚫 AMP检查: {gpu} GPU可能导致AMP问题，建议禁用AMP")
                return False
        except Exception as e:
            LOGGER.warning(f"🚫 AMP GPU检查失败: {e}")
        
        # 在离线模式下，假设AMP工作正常（避免下载模型进行测试）
        LOGGER.info("✅ AMP检查跳过下载测试 - 离线模式假设AMP可用")
        LOGGER.info("💡 如果训练中遇到NaN loss或zero-mAP，请设置 amp=False")
        return True
    
    # 替换函数
    downloads.attempt_download_asset = no_download_attempt_download_asset
    tasks.attempt_download_asset = no_download_attempt_download_asset
    checks.check_amp = no_download_check_amp
    
    LOGGER.info("🚫 已启用离线模式 - 禁用所有自动下载和AMP检查下载")

def enable_downloads():
    """重新启用下载功能"""
    global _original_attempt_download_asset, _original_check_amp
    
    if _original_attempt_download_asset is not None:
        from ultralytics.utils import downloads
        from ultralytics.nn import tasks
        
        downloads.attempt_download_asset = _original_attempt_download_asset
        tasks.attempt_download_asset = _original_attempt_download_asset
        
    if _original_check_amp is not None:
        from ultralytics.utils import checks
        checks.check_amp = _original_check_amp
        
    LOGGER.info("✅ 已恢复下载功能和AMP检查")

# 自动启用离线模式
disable_downloads()
