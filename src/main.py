# -*- coding: utf-8 -*-
"""
Image2CAD 主入口文件
整合所有组件并启动应用程序
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入必要的模块
from src.server.server import start_server
from src.common.config_manager import ConfigManager
from src.common.log_manager import setup_logging

def main():
    """应用程序主入口"""
    # 初始化日志
    setup_logging()
    
    # 加载配置
    config = ConfigManager()
    
    # 启动服务器
    start_server(config)

if __name__ == "__main__":
    main()