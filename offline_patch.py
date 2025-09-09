"""
离线模式补丁 - 禁用Ultralytics的自动下载功能
在离线环境中导入此模块来禁用所有下载逻辑
"""

import logging
from pathlib import Path
from ultralytics.utils import LOGGER

# 保存原始函数的引用
_original_attempt_download_asset = None

def disable_downloads():
    """禁用所有下载功能"""
    global _original_attempt_download_asset
    
    # 导入需要修改的模块
    from ultralytics.utils import downloads
    from ultralytics.nn import tasks
    
    # 保存原始函数
    if _original_attempt_download_asset is None:
        _original_attempt_download_asset = downloads.attempt_download_asset
    
    def no_download_attempt_download_asset(file, **kwargs):
        """替代attempt_download_asset的离线版本"""
        file_path = Path(file)
        
        # 如果文件存在，直接返回
        if file_path.exists():
            return str(file_path)
        
        # 检查权重目录
        from ultralytics.utils import SETTINGS
        weights_file = Path(SETTINGS["weights_dir"]) / file_path.name
        if weights_file.exists():
            return str(weights_file)
        
        # 如果文件不存在，抛出错误而不是下载
        raise FileNotFoundError(
            f"离线模式: 无法找到权重文件 '{file}'。\n"
            f"请确保文件存在于以下位置之一:\n"
            f"1. 当前目录: {file_path.absolute()}\n"
            f"2. 权重目录: {weights_file.absolute()}\n"
            f"或者使用预下载脚本下载所需权重。"
        )
    
    # 替换函数
    downloads.attempt_download_asset = no_download_attempt_download_asset
    tasks.attempt_download_asset = no_download_attempt_download_asset
    
    LOGGER.info("🚫 已启用离线模式 - 禁用所有自动下载")

def enable_downloads():
    """重新启用下载功能"""
    global _original_attempt_download_asset
    
    if _original_attempt_download_asset is not None:
        from ultralytics.utils import downloads
        from ultralytics.nn import tasks
        
        downloads.attempt_download_asset = _original_attempt_download_asset
        tasks.attempt_download_asset = _original_attempt_download_asset
        
        LOGGER.info("✅ 已恢复下载功能")

# 自动启用离线模式
disable_downloads()
