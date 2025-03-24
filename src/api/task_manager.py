# -*- coding: utf-8 -*-
"""
任务管理器
处理任务的创建、状态查询和取消
"""
from fastapi import HTTPException
from typing import Optional, Any, Dict
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.tasks.celery_tasks import process_cad_image, convert_pdf_to_images, ocr_image, app as celery_app

class TaskManager:
    """任务管理器，处理任务的创建、状态查询和取消"""
    
    @staticmethod
    def get_task(task_id: str):
        """
        获取Celery任务对象
        
        Args:
            task_id: 任务ID
            
        Returns:
            Celery AsyncResult 对象
        """
        # 尝试从不同的任务类型中获取结果
        tasks = [process_cad_image, convert_pdf_to_images, ocr_image]
        for task_type in tasks:
            task = task_type.AsyncResult(task_id)
            if task.state != 'PENDING':  # 如果找到了任务
                return task
        return process_cad_image.AsyncResult(task_id)  # 默认返回
    
    @staticmethod
    def create_task(task_name: str, file_path: str, output_dir: str) -> Any:
        """
        创建新任务
        
        Args:
            task_name: 任务类型名称
            file_path: 输入文件路径
            output_dir: 输出目录路径
            
        Returns:
            创建的任务对象
            
        Raises:
            ValueError: 如果任务类型无效
        """
        if task_name == "png_to_dxf":
            return process_cad_image.delay(file_path, output_dir)
        elif task_name == "ocr_image":
            return ocr_image.delay(file_path, output_dir)
        elif task_name == "pdf_to_images":
            return convert_pdf_to_images.delay(file_path, output_dir)
        else:
            raise ValueError(f"Unknown task type: {task_name}")
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            包含任务状态和元数据的字典
        """
        task = TaskManager.get_task(task_id)
        
        # 构建基本响应
        response = {"task_id": task_id, "status": task.status}
        
        # 添加任务元数据（如果有）
        try:
            if hasattr(task, 'info') and task.info:
                if isinstance(task.info, dict):
                    for key, value in task.info.items():
                        response[key] = value
        except Exception as e:
            print(f"Error getting task info: {e}")
        
        # 添加结果（如果任务已完成）
        if task.ready():
            try:
                response["result"] = task.result
            except Exception as e:
                response["error"] = str(e)
        
        return response
    
    @staticmethod
    def cancel_task(task_id: str) -> Dict[str, Any]:
        """
        取消任务
        
        Args:
            task_id: 要取消的任务ID
            
        Returns:
            取消操作的结果
            
        Raises:
            HTTPException: 如果任务不存在或取消过程中出错
        """
        task = TaskManager.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.state in ['SUCCESS', 'FAILURE']:
            raise HTTPException(
                status_code=400,
                content={"message": f"Task already in final state: {task.state}"}
            )
        
        # 尝试撤销任务
        celery_app.control.revoke(task_id, terminate=True)
        
        return {
            "message": "Task cancellation successful",
            "task_id": task_id,
            "status": "REVOKED"
        }

# 创建全局任务管理器实例
task_manager = TaskManager()