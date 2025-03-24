# -*- coding: utf-8 -*-
from celery import Celery, states
from celery.exceptions import Ignore
import time
from image2CAD import pdf_to_images, png_to_dxf
from celery.signals import task_revoked
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

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
def ocr_image(self, image_path, scale_factor=5, max_block_size=512, overlap=20, output_path=None):
    """OCR图像处理任务，支持进度报告"""
    try:
        print(f"OCR Processing {image_path} -> {output_path}")
        
        # 更新任务状态为开始处理
        self.update_state(state='PROCESSING',
                         meta={'progress': 0,
                               'current': 'Starting OCR processing',
                               'input': image_path,
                               'output': output_path})
        
        # 导入OCR处理模块
        from ocrProcess import OCRProcess
        ocr_process = OCRProcess()
        
        # 执行OCR处理
        self.update_state(state='PROCESSING',
                         meta={'progress': 50,
                               'current': 'Running OCR analysis',
                               'input': image_path})
                               
        parsed_results, original_height = ocr_process.get_file_rapidOCR(image_path)
        
        # 更新任务状态为完成
        self.update_state(state='SUCCESS',
                         meta={'progress': 100,
                               'current': 'Completed',
                               'input': image_path,
                               'output': output_path,
                               'results': parsed_results})
                               
        print("OCR processing finished")
        return {'status': 'success', 
                'results': parsed_results, 
                'original_height': original_height}
                
    except Exception as e:
        # 更新任务状态为失败
        self.update_state(state='FAILURE',
                         meta={'error': str(e),
                               'input': image_path})
        raise