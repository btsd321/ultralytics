#!/usr/bin/env python3
"""
离线环境准备脚本 - 预下载所需的模型权重
在有网络的环境中运行此脚本，将权重下载到指定目录
"""

import os
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset, GITHUB_ASSETS_NAMES
from ultralytics.utils import SETTINGS

def download_common_weights():
    """下载常用的YOLO权重文件"""
    
    # 创建权重目录
    weights_dir = Path(SETTINGS["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # 常用权重列表 - 可根据需要调整
    common_weights = [
        # YOLO11 检测模型
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        
        # YOLO11 分割模型
        "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
        
        # YOLO11 分类模型
        "yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt",
        
        # YOLO11 姿态估计模型
        "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
        
        # YOLO11 OBB模型
        "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt",
        
        # YOLOv8 常用模型
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",

        # YOLOv8 分割模型
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",

        # YOLOv8 OBB模型
        "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt",
    ]
    
    print(f"开始下载权重文件到: {weights_dir}")
    
    for weight in common_weights:
        if weight in GITHUB_ASSETS_NAMES:
            try:
                print(f"正在下载: {weight}")
                file_path = attempt_download_asset(weight)
                print(f"✓ 下载完成: {file_path}")
            except Exception as e:
                print(f"✗ 下载失败 {weight}: {e}")
        else:
            print(f"⚠ 跳过未知权重: {weight}")
    
    print("\n下载完成！")
    print(f"权重文件保存在: {weights_dir}")

def list_all_available_weights():
    """列出所有可用的权重文件"""
    print("所有可用的预训练权重:")
    for i, weight in enumerate(sorted(GITHUB_ASSETS_NAMES), 1):
        print(f"{i:3d}. {weight}")

if __name__ == "__main__":
    print("=== Ultralytics 离线环境准备工具 ===\n")
    
    print("1. 列出所有可用权重")
    list_all_available_weights()
    
    print("\n" + "="*50 + "\n")
    
    print("2. 下载常用权重")
    download_common_weights()
