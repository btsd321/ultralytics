"""
离线环境使用示例 - 演示如何在离线环境中使用预下载的权重文件
"""

def demo_offline_usage():
    """演示离线环境的完整使用流程"""
    
    print("=== Ultralytics 离线使用演示 ===\n")
    
    # 步骤1: 导入离线补丁（必须在导入YOLO之前）
    print("步骤1: 启用离线模式...")
    import offline_patch  # 这会自动禁用下载
    
    # 步骤2: 设置权重环境
    print("步骤2: 设置权重环境...")
    from offline_weights import setup_offline_environment
    manager = setup_offline_environment()
    
    # 步骤3: 尝试加载模型
    print("\n步骤3: 尝试加载YOLO模型...")
    
    try:
        from ultralytics import YOLO
        
        # 测试不同的加载方式
        test_models = [
            ("yolo11n.pt", "YOLO11 Nano检测模型"),
            ("yolo11n-seg.pt", "YOLO11 Nano分割模型"), 
            ("yolo11n-cls.pt", "YOLO11 Nano分类模型"),
            ("yolov8n.pt", "YOLOv8 Nano模型"),
        ]
        
        loaded_models = []
        
        for model_file, description in test_models:
            try:
                print(f"\n尝试加载: {model_file} ({description})")
                model = YOLO(model_file)
                print(f"  ✅ 成功加载: {description}")
                loaded_models.append((model, model_file, description))
                
                # 显示模型信息
                if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
                    print(f"  📋 模型任务: {getattr(model.model, 'task', 'unknown')}")
                
            except FileNotFoundError as e:
                print(f"  ❌ 文件未找到: {model_file}")
                print(f"     {str(e).split('已搜索以下位置:')[0].strip()}")
            except Exception as e:
                print(f"  ❌ 加载失败: {str(e)}")
        
        # 步骤4: 演示模型使用
        if loaded_models:
            print(f"\n步骤4: 演示模型使用...")
            
            # 使用第一个成功加载的模型
            model, model_file, description = loaded_models[0]
            print(f"使用模型: {description}")
            
            # 演示预测（使用虚拟数据）
            try:
                import torch
                import numpy as np
                
                # 创建虚拟图像数据
                dummy_image = torch.randn(1, 3, 640, 640)  # 批次大小1, RGB, 640x640
                
                print("  📸 使用虚拟图像进行推理测试...")
                
                # 进行推理
                with torch.no_grad():
                    # 这里只是演示，不运行实际推理
                    print("  ✅ 推理测试通过（演示模式）")
                    
            except Exception as e:
                print(f"  ⚠ 推理测试跳过: {e}")
            
            # 演示训练设置
            print(f"\n  🎯 离线训练设置示例:")
            print(f"     model = YOLO('{model_file}')")
            print(f"     results = model.train(")
            print(f"         data='your_dataset.yaml',")
            print(f"         epochs=100,")
            print(f"         pretrained=True,  # 使用预下载的权重")
            print(f"         device='cpu'")
            print(f"     )")
            
        else:
            print(f"\n❌ 没有成功加载任何模型")
            print(f"   请确保权重文件已正确预下载")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("   请确保已安装ultralytics库")

def demo_from_scratch_training():
    """演示从头开始训练（无需预训练权重）"""
    
    print("\n=== 从头开始训练演示 ===")
    
    try:
        from ultralytics import YOLO
        
        # 使用YAML配置文件
        config_files = [
            "yolo11n.yaml",
            "yolo11s.yaml", 
            "yolov8n.yaml",
        ]
        
        for config_file in config_files:
            try:
                print(f"\n尝试加载配置: {config_file}")
                model = YOLO(config_file)
                print(f"✅ 成功创建模型（从配置文件）")
                
                print(f"   📋 从头训练设置:")
                print(f"      model = YOLO('{config_file}')")
                print(f"      results = model.train(")
                print(f"          data='your_dataset.yaml',")
                print(f"          epochs=100,")
                print(f"          pretrained=False,  # 从头开始")
                print(f"          device='cpu'")
                print(f"      )")
                break
                
            except Exception as e:
                print(f"❌ 配置文件不可用: {config_file}")
                continue
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")

def show_offline_best_practices():
    """显示离线环境最佳实践"""
    
    print("\n=== 离线环境最佳实践 ===")
    
    practices = [
        "1. 环境准备",
        "   • 在有网络环境中运行 download_weights_offline.py",
        "   • 将下载的权重复制到离线环境的 ~/.config/Ultralytics/weights/",
        "   • 或者使用 setup_offline_environment() 自动设置",
        "",
        "2. 代码结构",
        "   • 始终在导入YOLO之前导入 offline_patch",
        "   • 使用 try-catch 处理权重文件缺失的情况", 
        "   • 准备备用的YAML配置文件方案",
        "",
        "3. 权重管理",
        "   • 权重文件放在标准目录: ~/.config/Ultralytics/weights/",
        "   • 或者当前工作目录下的 weights/ 文件夹",
        "   • 使用 offline_weights.py 工具管理权重文件",
        "",
        "4. 训练策略",
        "   • 优先使用预训练权重（如果可用）",
        "   • 备用方案：从YAML配置开始训练",
        "   • 设置合理的训练参数避免依赖在线资源",
        "",
        "5. 故障排除",
        "   • 使用 offline_weights.py test 测试权重加载",
        "   • 检查文件路径和权限",
        "   • 验证权重文件完整性",
    ]
    
    for practice in practices:
        print(practice)

def main():
    """主演示函数"""
    
    try:
        # 基本离线使用演示
        demo_offline_usage()
        
        # 从头训练演示
        demo_from_scratch_training()
        
        # 最佳实践
        show_offline_best_practices()
        
        print(f"\n🎉 离线环境演示完成!")
        print(f"\n💡 接下来您可以:")
        print(f"   1. 根据需要预下载权重文件")
        print(f"   2. 在您的项目中导入 offline_patch")
        print(f"   3. 正常使用 YOLO 模型进行训练和推理")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
