# -*- coding: utf-8 -*-
"""
API 路由定义
包含所有HTTP和WebSocket端点
"""
from fastapi import APIRouter, File, UploadFile, Form, WebSocket, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
from typing import Dict, Any

from .task_manager import task_manager
from .websocket_manager import websocket_manager
from .file_utils import file_utils
from common.utils import Util  # 假设 Util 在 common.utils 中
from common.config_manager import ConfigManager

# 获取允许的文件扩展名
config_manager = ConfigManager()
ALLOWED_IMAGE_EXTENSIONS = config_manager.get_allow_imgExt()

# 创建路由器
router = APIRouter()

@router.get("/")
async def root():
    """重定向到测试页面"""
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html"))

@router.post("/upload/image/")
async def upload_file(file: UploadFile = File(...), task_name: str = Form(...)):
    """
    处理图像文件上传并启动相应的处理任务
    
    Args:
        file: 上传的图像文件
        task_name: 任务类型，支持 "png_to_dxf" 或 "ocr_image"
        
    Returns:
        包含任务ID和WebSocket URL的JSON响应
    """
    # 验证任务类型
    if task_name not in ["png_to_dxf", "ocr_image"]:
        return {"error": "Unknown task type"}
    
    # 验证文件类型
    if not Util.validate_extname(file.filename, ALLOWED_IMAGE_EXTENSIONS, False):
        return {"error": "Invalid file type"}
    
    # 保存上传的文件
    file_path = file_utils.save_uploaded_file(file)
    
    # 确定输出目录
    output_dir = file_utils.get_output_dir(ALLOWED_IMAGE_EXTENSIONS, file_path, task_name)
    
    try:
        # 创建任务
        task = task_manager.create_task(task_name, file_path, output_dir)
        
        return {
            "task_id": task.id, 
            "message": "Processing started", 
            "websocket_url": f"/ws/task/{task.id}"
        }
    except ValueError as e:
        return {"error": str(e)}

@router.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...), task_name: str = Form(...)):
    """
    处理PDF文件上传并启动转换为图像的任务
    
    Args:
        file: 上传的PDF文件
        task_name: 任务类型，必须为 "pdf_to_images"
        
    Returns:
        包含任务ID和WebSocket URL的JSON响应
    """
    # 验证任务类型
    if task_name != "pdf_to_images":
        return {"error": "Unknown task type"}
    
    # 验证文件类型
    if not Util.validate_extname(file.filename, [".pdf"], False):
        return {"error": "Invalid file type"}
    
    # 保存上传的文件
    file_path = file_utils.save_uploaded_file(file)
    
    # 确定输出目录并启动任务
    output_dir = file_utils.get_output_dir([".pdf"], file_path, task_name)
    task = task_manager.create_task(task_name, file_path, output_dir)

    return {
        "task_id": task.id, 
        "message": "PDF to images processing started", 
        "websocket_url": f"/ws/task/{task.id}"
    }

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        包含任务状态和元数据的JSON响应
    """
    return task_manager.get_task_status(task_id)

@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """
    取消正在运行的任务
    
    Args:
        task_id: 要取消的任务ID
        
    Returns:
        取消操作的结果
    """
    result = task_manager.cancel_task(task_id)
    
    # 关闭相关的WebSocket连接
    websocket = websocket_manager.get_connection(task_id)
    if websocket:
        try:
            # 先发送取消通知
            await websocket.send_json({
                "task_id": task_id,
                "status": "REVOKED",
                "message": "Task has been cancelled by user request"
            })
            # 等待一小段时间确保消息发送
            import asyncio
            await asyncio.sleep(0.1)
            # 关闭WebSocket连接
            await websocket_manager.close_connection(task_id, 1000, "Task cancelled")
            result["websocket_closed"] = True
        except Exception as e:
            print(f"Error in WebSocket communication for task {task_id}: {e}")
            result["websocket_closed"] = False
    
    return result

@router.websocket("/ws/task/{task_id}")
async def websocket_task_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket端点，用于实时监控任务进度
    
    Args:
        websocket: WebSocket连接对象
        task_id: 任务ID
    """
    await websocket_manager.monitor_task_progress(task_id, websocket, task_manager.get_task)

@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    下载处理后的文件
    
    Args:
        filename: 要下载的文件名
        
    Returns:
        文件响应对象
    """
    file_path = os.path.join(file_utils.output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='application/octet-stream', filename=filename)