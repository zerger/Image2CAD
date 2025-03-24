# -*- coding: utf-8 -*-
"""
文件处理工具
处理文件上传、保存和目录管理
"""
import os
import shutil
from typing import List, Optional
from fastapi import UploadFile

class FileUtils:
    """文件处理工具类，处理文件上传、保存和目录管理"""
    
    def __init__(self, upload_dir: str = "uploads", output_dir: str = "outputs"):
        """
        初始化文件工具
        
        Args:
            upload_dir: 上传文件保存目录
            output_dir: 输出文件保存目录
        """
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        
        # 确保目录存在
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_uploaded_file(self, file: UploadFile) -> str:
        """
        保存上传的文件
        
        Args:
            file: FastAPI UploadFile 对象
            
        Returns:
            保存后的文件绝对路径
        """
        file_path = os.path.abspath(os.path.join(self.upload_dir, file.filename))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path
    
    def get_output_dir(self, file_extensions: List[str], file_path: str, task_name: str) -> Optional[str]:
        """
        根据任务类型和文件名生成输出目录路径
        
        Args:
            file_extensions: 允许的文件扩展名列表
            file_path: 输入文件路径
            task_name: 任务名称
        
        Returns:
            输出目录的绝对路径，如果任务类型未知则返回None
        """
        def get_dir_name(file_name: str, extensions: List[str], dir_suffix: str) -> str:
            """生成输出目录名"""
            dir_name = file_name
            for ext in extensions:
                if dir_name.lower().endswith(ext.lower()):
                    dir_name = dir_name.replace(ext, dir_suffix)
                    break
            return os.path.abspath(os.path.join(self.output_dir, dir_name))
        
        filename = os.path.basename(file_path)
        task_suffix_map = {
            "png_to_dxf": "_dxf",
            "ocr_image": "_ocr",
            "pdf_to_images": "_images"
        }
        
        if task_name in task_suffix_map:
            return get_dir_name(filename, file_extensions, task_suffix_map[task_name])
        return None
    
    def clean_old_files(self, max_age_days: int = 7):
        """
        清理超过指定天数的旧文件
        
        Args:
            max_age_days: 文件最大保留天数
        """
        import time
        from datetime import datetime, timedelta
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        # 清理上传目录
        self._clean_directory(self.upload_dir, cutoff_time)
        
        # 清理输出目录
        self._clean_directory(self.output_dir, cutoff_time)
    
    def _clean_directory(self, directory: str, cutoff_time: float):
        """
        清理指定目录中的旧文件
        
        Args:
            directory: 要清理的目录
            cutoff_time: 截止时间戳
        """
        for root, dirs, files in os.walk(directory):
            # 清理文件
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        try:
                            os.remove(file_path)
                            print(f"Removed old file: {file_path}")
                        except Exception as e:
                            print(f"Error removing file {file_path}: {e}")
            
            # 清理空目录
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if os.path.isdir(dir_path) and not os.listdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                        print(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        print(f"Error removing directory {dir_path}: {e}")

# 创建全局文件工具实例
file_utils = FileUtils()