# -*- coding: utf-8 -*-
"""
WebSocket 连接管理器
处理任务进度的实时通知
"""
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from tasks.celery_tasks import app as celery_app

class WebSocketManager:
    """WebSocket连接管理器，处理任务进度的实时通知"""
    
    def __init__(self):
        """初始化WebSocket管理器"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_timeout_minutes: int = 30  # 默认超时时间
    
    def set_timeout(self, minutes: int):
        """设置任务超时时间"""
        self.task_timeout_minutes = minutes
    
    def get_connection(self, task_id: str) -> Optional[WebSocket]:
        """获取指定任务ID的WebSocket连接"""
        return self.active_connections.get(task_id)
    
    def register_connection(self, task_id: str, websocket: WebSocket):
        """注册新的WebSocket连接"""
        self.active_connections[task_id] = websocket
    
    async def close_connection(self, task_id: str, code: int = 1000, reason: str = "Task completed"):
        """
        安全关闭WebSocket连接并清理资源
        
        Args:
            task_id: 任务ID
            code: 关闭代码
            reason: 关闭原因
        """
        if task_id in self.active_connections:
            websocket = self.active_connections[task_id]
            try:
                await websocket.close(code=code, reason=reason)
            except Exception as e:
                print(f"Error closing websocket for task {task_id}: {e}")
            finally:
                # 确保清理连接记录
                if task_id in self.active_connections:
                    del self.active_connections[task_id]
    
    async def monitor_task_progress(self, task_id: str, websocket: WebSocket, get_task_func):
        """
        监控任务进度并通过WebSocket发送更新
        
        Args:
            task_id: 任务ID
            websocket: WebSocket连接
            get_task_func: 获取任务对象的函数
        """
        await websocket.accept()
        self.register_connection(task_id, websocket)
        
        start_time = datetime.now()
        last_status = None
        last_progress = None
        progress_counter = 0
        
        try:
            while True:
                # 检查是否超时
                if datetime.now() - start_time > timedelta(minutes=self.task_timeout_minutes):
                    await websocket.send_json({
                        "task_id": task_id,
                        "status": "error",
                        "message": f"Task timed out after {self.task_timeout_minutes} minutes"
                    })
                    # 取消超时的任务
                    celery_app.control.revoke(task_id, terminate=True)
                    break

                task = get_task_func(task_id)
                status = task.status
                
                # 获取任务的元数据和进度信息
                task_meta = {}
                progress = None
                current_operation = None
                
                try:
                    if hasattr(task, 'info') and task.info:
                        if isinstance(task.info, dict):
                            task_meta = task.info
                            progress = task_meta.get('progress')
                            current_operation = task_meta.get('current')
                except Exception as e:
                    print(f"Error getting task metadata: {str(e)}")
                
                # 只有当状态发生变化、进度发生变化或者每10次循环发送一次更新
                status_changed = status != last_status
                progress_changed = progress is not None and progress != last_progress
                periodic_update = progress_counter % 10 == 0
                
                if status_changed or progress_changed or periodic_update:
                    # 构建状态消息
                    status_message = {
                        "task_id": task_id,
                        "status": status,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if progress is not None:
                        status_message["progress"] = progress
                        last_progress = progress
                    
                    if current_operation:
                        status_message["operation"] = current_operation
                    
                    # 添加任务元数据
                    for key in ['input', 'output', 'error']:
                        if key in task_meta:
                            status_message[key] = task_meta[key]
                    
                    # 添加任务结果（如果已完成）
                    if task.ready():
                        try:
                            result = task.result
                            if isinstance(result, dict) and 'status' in result:
                                status_message["result"] = result
                            else:
                                status_message["result"] = {"data": result}
                        except Exception as e:
                            status_message["error"] = str(e)
                    
                    await websocket.send_json(status_message)
                    last_status = status
                
                # 如果任务已完成，发送最终状态并退出循环
                if status in ['SUCCESS', 'FAILURE', 'REVOKED']:
                    final_message = {
                        "task_id": task_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if status == 'SUCCESS':
                        final_message.update({
                            "status": "completed",
                            "message": "Task completed successfully"
                        })
                        try:
                            result = task.result
                            if result:
                                final_message["result"] = result
                        except Exception as e:
                            final_message["error"] = str(e)
                            
                    elif status == 'FAILURE':
                        error_info = "Unknown error"
                        try:
                            if task.result:
                                error_info = str(task.result)
                            elif hasattr(task, 'traceback') and task.traceback:
                                error_info = task.traceback
                        except Exception:
                            pass
                            
                        final_message.update({
                            "status": "error",
                            "message": f"Task failed: {error_info}"
                        })
                        
                    elif status == 'REVOKED':
                        final_message.update({
                            "status": "cancelled",
                            "message": "Task was cancelled"
                        })
                    
                    await websocket.send_json(final_message)
                    break
                
                progress_counter += 1
                await asyncio.sleep(2)  # 每2秒检查一次状态
                
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for task {task_id}")
        except Exception as e:
            print(f"Error during progress monitoring: {e}")
            
        finally:
            try:
                # WebSocket断开时取消任务
                celery_app.control.revoke(task_id, terminate=True)
                # 清理WebSocket连接
                if task_id in self.active_connections:
                    del self.active_connections[task_id]
            except Exception as e:
                print(f"Error during task cleanup: {e}")
            finally:
                try:
                    await websocket.close()
                except Exception:
                    pass

# 创建全局WebSocket管理器实例
websocket_manager = WebSocketManager()