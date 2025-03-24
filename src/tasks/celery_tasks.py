# -*- coding: utf-8 -*-
"""
Celery任务定义
处理异步任务的执行和状态更新
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from celery import Celery, states
from celery.exceptions import Ignore
import time
from src.processors.image2cad import pdf_to_images, png_to_dxf
from celery.signals import task_revoked

# 创建Celery应用
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# 配置Celery
app.conf.update(
    broker_connection_retry_on_startup=True,  # 在启动时重试连接代理
    broker_connection_max_retries=10,         # 最大重试次数
    task_serializer='json',                   # 任务序列化格式
    accept_content=['json'],                  # 接受的内容类型
    result_serializer='json',                 # 结果序列化格式
    timezone='Asia/Shanghai',                 # 时区设置
    enable_utc=False,                         # 不使用UTC
)

@task_revoked.connect
def handle_task_revoked(sender=None, request=None, terminated=None, signum=None, expired=None, **kwargs):
    """处理任务被撤销的情况"""
    task_id = request.id if request else "Unknown"
    print(f"Task {task_id} was revoked. Terminated: {terminated}, Expired: {expired}")
    # 可以在这里添加清理资源的代码

@app.task(bind=True)
def process_cad_image(self, input_path, output_path=None):
    """处理CAD图像转换任务，支持进度报告"""
    try:
        print(f"Processing {input_path} -> {output_path}")
        
        # 更新任务状态为开始处理
        self.update_state(state='PROCESSING',
                         meta={'progress': 0,
                               'current': 'Starting CAD processing',
                               'input': input_path,
                               'output': output_path})
                               
        # 执行转换
        png_to_dxf(input_path, output_path)
        
        # 更新任务状态为完成
        self.update_state(state='SUCCESS',
                         meta={'progress': 100,
                               'current': 'Completed',
                               'input': input_path,
                               'output': output_path})
                               
        return {'status': 'success', 'output_path': output_path}
        
    except Exception as e:
        # 更新任务状态为失败
        self.update_state(state='FAILURE',
                         meta={'error': str(e),
                               'input': input_path})
        raise
    
@app.task(bind=True)
def convert_pdf_to_images(self, pdf_path, output_dir=None, dpi=None):
    """转换PDF到图片任务，支持进度报告"""
    try:
        print(f"Converting pdf to images {pdf_path} to {output_dir}")
        
        # 更新任务状态为开始处理
        self.update_state(state='PROCESSING',
                         meta={'progress': 0,
                               'current': 'Starting PDF conversion',
                               'input': pdf_path,
                               'output': output_dir})
                               
        # 执行转换
        pdf_to_images(pdf_path, output_dir)
        
        # 更新任务状态为完成
        self.update_state(state='SUCCESS',
                         meta={'progress': 100,
                               'current': 'Completed',
                               'input': pdf_path,
                               'output': output_dir})
                               
        return {'status': 'success', 'output_dir': output_dir}
        
    except Exception as e:
        # 更新任务状态为失败
        self.update_state(state='FAILURE',
                         meta={'error': str(e),
                               'input': pdf_path})
        raise

@app.task(bind=True)
def ocr_image(self, image_path, output_dir=None):
    """OCR图像识别任务，支持进度报告"""
    try:
        print(f"OCR processing {image_path} to {output_dir}")
        
        # 更新任务状态为开始处理
        self.update_state(state='PROCESSING',
                         meta={'progress': 0,
                               'current': 'Starting OCR processing',
                               'input': image_path,
                               'output': output_dir})
        
        # 这里应该调用OCR处理函数
        # 由于原代码中没有实现，这里只是模拟处理
        time.sleep(2)  # 模拟处理时间
        
        # 更新任务状态为完成
        self.update_state(state='SUCCESS',
                         meta={'progress': 100,
                               'current': 'Completed',
                               'input': image_path,
                               'output': output_dir})
                               
        return {'status': 'success', 'output_dir': output_dir}
        
    except Exception as e:
        # 更新任务状态为失败
        self.update_state(state='FAILURE',
                         meta={'error': str(e),
                               'input': image_path})
        raise