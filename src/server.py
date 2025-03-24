# -*- coding: utf-8 -*-
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
from util import Util
from configManager import ConfigManager
from tasks import process_cad_image, convert_pdf_to_images, ocr_image, app as celery_app  # Celery 任务和应用实例

# 从配置文件获取任务超时时间
TASK_TIMEOUT_MINUTES = ConfigManager.get_task_timeout()
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建static目录（如果不存在）
STATIC_DIR = "static"
os.makedirs(os.path.join(os.path.dirname(__file__), STATIC_DIR), exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), STATIC_DIR)), name="static")

# 存储任务ID到WebSocket连接的映射
websocket_connections: Dict[str, WebSocket] = {}

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

allow_imgExt = ConfigManager.get_allow_imgExt()
    
def get_outputDir(fileAllowExt, file_path, task_name):
    def get_dir_name(file_name, fileAllowExt, dir_suffix):
        dir_name = file_name
        for ext in fileAllowExt:
            if dir_name.endswith(ext):
                dir_name = dir_name.replace(ext, dir_suffix)
                break  # 找到匹配的扩展名后可以退出循环
        return  os.path.abspath(os.path.join(OUTPUT_DIR, dir_name))
    filename = os.path.basename(file_path)
    if task_name == "png_to_dxf":
        return get_dir_name(filename, fileAllowExt, "_dxf")       
    elif task_name == "ocr_image":
        return get_dir_name(filename, fileAllowExt, "_ocr")               
    elif task_name == "pdf_to_images":
        return get_dir_name(filename, fileAllowExt, "_images")    
    else:
        return None
        
@app.post("/upload/image/")
async def upload_file(file: UploadFile = File(...), task_name: str = Form(...)):
    if task_name != "png_to_dxf" and task_name != "ocr_image":
        return {"error": "Unknown task type"}
    if not Util.validate_extname(file.filename, allow_imgExt, False):
        return {"error": "Invalid file type"}
    file_path = os.path.abspath(os.path.join(UPLOAD_DIR, file.filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    output_dir = get_outputDir(allow_imgExt, file_path, task_name)
    if task_name == "png_to_dxf":        
        task = process_cad_image.delay(file_path, output_dir)    
    elif task_name == "ocr_image":       
        task = ocr_image.delay(file_path, output_dir)
    else:
        return {"error": "Unknown task type"}


    return {"task_id": task.id, "message": "Processing started", "websocket_url": f"/ws/task/{task.id}"}

@app.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...), task_name: str = Form(...)):
    if task_name != "pdf_to_images":
        return {"error": "Unknown task type"}
    if not Util.validate_extname(file, [".pdf"], False):
        return {"error": "Invalid file type"}
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_dir = get_outputDir({".pdf"}, file_path, task_name)
    # 异步执行 PDF 转换任务
    task = convert_pdf_to_images.delay(file_path, output_dir)


    return {"task_id": task.id, "message": "PDF to images processing started", "websocket_url": f"/ws/task/{task.id}"}

def get_celery_task(task_id: str):
    """获取Celery任务对象，根据任务类型返回对应的AsyncResult"""
    # 尝试从不同的任务类型中获取结果
    tasks = [process_cad_image, convert_pdf_to_images, ocr_image]
    for task_type in tasks:
        task = task_type.AsyncResult(task_id)
        if task.state != 'PENDING':  # 如果找到了任务
            return task
    return process_cad_image.AsyncResult(task_id)  # 默认返回

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = get_celery_task(task_id)
    
    # 构建基本响应
    response = {"task_id": task_id, "status": task.status}
    
    # 添加任务元数据（如果有）
    try:
        if hasattr(task, 'info') and task.info:
            if isinstance(task.info, dict):
                for key, value in task.info.items():
                    response[key] = value
    except:
        pass
        
    # 添加结果（如果任务已完成）
    if task.ready():
        try:
            response["result"] = task.result
        except Exception as e:
            response["error"] = str(e)
    
    return response

@app.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """取消正在运行的任务"""
    task = get_celery_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.state in ['SUCCESS', 'FAILURE']:
        return JSONResponse(
            status_code=400,
            content={"message": f"Task already in final state: {task.state}"}
        )
    
    # 尝试撤销任务
    celery_app.control.revoke(task_id, terminate=True)
    try:
        cancel_message = {
            "task_id": task_id,
            "status": "REVOKED",
            "message": "Task has been cancelled by user request"
        }
        
        # 关闭相关的WebSocket连接
        if task_id in websocket_connections:
            try:
                websocket = websocket_connections[task_id]
                # 先发送取消通知
                await websocket.send_json(cancel_message)
                # 等待一小段时间确保消息发送
                await asyncio.sleep(0.1)
                # 正常关闭WebSocket连接
                await websocket.close(code=1000, reason="Task cancelled")
            except WebSocketDisconnect:
                print(f"WebSocket for task {task_id} was already disconnected")
            except Exception as e:
                print(f"Error in WebSocket communication for task {task_id}: {e}")
            finally:
                # 确保清理连接记录
                if task_id in websocket_connections:
                    del websocket_connections[task_id]
        
        return {
            "message": "Task cancellation successful",
            "task_id": task_id,
            "websocket_closed": task_id in websocket_connections
        }
        
    except Exception as e:
        # 如果在取消过程中发生错误，确保资源被清理
        if task_id in websocket_connections:
            try:
                websocket = websocket_connections[task_id]
                await websocket.close(code=1011, reason="Server error during cancellation")
                del websocket_connections[task_id]
            except Exception:
                pass
                
        raise HTTPException(
            status_code=500,
            detail=f"Error during task cancellation: {str(e)}"
        )

@app.websocket("/ws/task/{task_id}")
async def websocket_task_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    # 存储WebSocket连接
    websocket_connections[task_id] = websocket
    start_time = datetime.now()
    last_status = None
    last_progress = None
    progress_counter = 0
    
    try:
        while True:
            # 检查是否超时
            if datetime.now() - start_time > timedelta(minutes=TASK_TIMEOUT_MINUTES):
                await websocket.send_json({
                    "task_id": task_id,
                    "status": "error",
                    "message": f"Task timed out after {TASK_TIMEOUT_MINUTES} minutes"
                })
                # 取消超时的任务
                celery_app.control.revoke(task_id, terminate=True)
                break

            task = get_celery_task(task_id)
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
                
                for key in ['input', 'output', 'error']:
                    if key in task_meta:
                        status_message[key] = task_meta[key]
                
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
                    except:
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
    except Exception as e:
        print(f"Error during progress monitoring: {e}")
        
    finally:
        try:
            print(f"WebSocket disconnected for task {task_id}")
            celery_app.control.revoke(task_id, terminate=True)  # WebSocket断开时取消任务
            # 清理WebSocket连接
            if task_id in websocket_connections:
                del websocket_connections[task_id]
        except Exception as e:
            print(f"Error during task cleanup: {e}")
            try:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Connection error: {str(e)}"
                })
            except:
                pass
        finally:
            try:
                await websocket.close()
            except:
                pass

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(file_path, media_type='application/octet-stream', filename=filename)

@app.get("/")
async def root():
    """重定向到测试页面"""
    return FileResponse(os.path.join(os.path.dirname(__file__), STATIC_DIR, "index.html"))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
