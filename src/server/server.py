# -*- coding: utf-8 -*-
"""
主服务器模块
启动FastAPI应用并配置中间件
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.routes import router
from tasks.celery_tasks import app as celery_app
from common.config_manager import ConfigManager
from api.file_utils import file_utils

# 创建FastAPI应用实例
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定允许的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 注册路由
app.include_router(router)

# 定期清理旧文件的后台任务
@app.on_event("startup")
async def startup_event():
    """服务器启动时的初始化操作"""
    # 读取配置
    config = ConfigManager()
    
    # 设置文件清理间隔（默认7天）
    max_age_days = config.get_file_retention_days()
    
    # 清理旧文件
    file_utils.clean_old_files(max_age_days)

if __name__ == "__main__":
    # 从配置文件获取服务器设置
    config = ConfigManager()
    host = config.get_server_host()
    port = config.get_server_port()
    
    # 启动服务器
    uvicorn.run("src.server.server:app", host=host, port=port, reload=True)