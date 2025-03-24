# -*- coding: utf-8 -*-
import os
import platform
import ctypes
import numpy as np
import cv2
import psutil
import shutil
from pathlib import Path
from src.common.errors import ProcessingError, InputError, ResourceError, TimeoutError
from typing import Union

class Util:
    @staticmethod
    def get_disk_space(path='/'):
        """获取磁盘剩余空间（跨平台）"""
        if platform.system() == 'Windows':
            return Util._clswindows_disk_space(path)
        else:
            return Util._unix_disk_space(path)

    @staticmethod
    def _unix_disk_space(path):
        """Unix/Linux/MacOS实现"""
        stat = os.statvfs(path)
        free = stat.f_bavail * stat.f_frsize
        total = stat.f_blocks * stat.f_frsize
        return free, total

    @staticmethod
    def _windows_disk_space(path):
        """Windows实现"""
        # 确保路径是驱动器根目录
        drive = os.path.splitdrive(os.path.abspath(path))[0]
        if not drive.endswith('\\'):
            drive += '\\'

        # 使用ctypes调用Win32 API
        free_bytes = ctypes.c_ulonglong()
        total_bytes = ctypes.c_ulonglong()
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(drive),
            None,
            ctypes.byref(total_bytes),
            ctypes.byref(free_bytes)
        )
        return free_bytes.value, total_bytes.value

    @staticmethod
    def get_disk_space_psutil(path='/'):
        """使用psutil获取磁盘空间"""
        usage = psutil.disk_usage(path)
        return usage.free, usage.total
    
    @staticmethod
    def has_valid_files(path, extensions):
        """递归检查目录或文件是否包含指定扩展名的文件"""
        path = Path(path)
        if not path.exists():
            return False
        return Util.validate_extname(path, extensions)
    
    @staticmethod
    def ensure_directory_exists(directory_path):
        """
        确保目录存在，如果不存在则创建。

        :param directory_path: 目录路径
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"创建目录: {directory_path}")
        else:
            print(f"目录已存在: {directory_path}")

    @staticmethod
    def ensure_file_exists(file_path):
        """
        确保文件存在。

        :param file_path: 文件路径
        """
        if not os.path.isfile(file_path):
            print(f"警告: 文件 {file_path} 不存在。")
        else:
            print(f"文件已存在: {file_path}")  
    
    @staticmethod        
    def validate_image_file(input_path: str) -> None:
        """验证输入文件有效性"""
        path = Path(input_path)
        if not path.exists():
            raise InputError(f"输入文件不存在: {input_path}")
        if not path.is_file():
            raise InputError(f"输入路径不是文件: {input_path}")
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tiff', '.tif'):
            raise InputError(f"不支持的文件格式: {path.suffix}")
        if path.stat().st_size > 100 * 1024 * 1024:  # 100MB限制
            raise InputError("文件大小超过100MB限制")
        
    @staticmethod
    def check_system_resources() -> None:
        """检查系统资源是否充足"""
        # 示例：检查磁盘空间
        free_space, _ = Util.get_disk_space_psutil('/')
        if free_space < 500 * 1024 * 1024:  # 500MB
            raise ResourceError("磁盘空间不足（需要至少500MB空闲空间）")
       
    @staticmethod
    def default_output_path(input_path, suffix):
        """生成默认输出路径"""
        base_dir = Path(input_path).parent / 'output'
        file_name = Path(input_path).stem.replace(" ", "_")
        return os.path.join(base_dir, f"{file_name}_{suffix}")    
    
    @staticmethod
    def remove_directory(dir_path):
        try:           
            shutil.rmtree(dir_path)           
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    
    @staticmethod
    def opencv_read(input_image_path):       
        with open(input_image_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)       
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)     
            return img
        return None
     
    @staticmethod        
    def opencv_write(img, output_image_path, ext='.png'):
        success, encoded_image = cv2.imencode(ext, img)
        if success:            
            image_bytes = encoded_image.tobytes()
            with open(output_image_path, 'wb') as f:
                f.write(image_bytes)                
        else:
            print("Failed to encode image")
                    
    @staticmethod
    def validate_extname(file_input, allowed_exts, is_file=True):
        """验证文件扩展名"""
        try:
            # 处理 UploadFile 对象
            if hasattr(file_input, 'filename'):
                filename = file_input.filename
            else:
                filename = str(file_input)

            # 统一路径分隔符并转换为绝对路径
            path = Path(filename).absolute()
            
            # 确保所有扩展名都是小写
            allowed_exts = [ext.lower() for ext in allowed_exts]
            
            if is_file:
                # 检查文件是否存在
                if not path.exists():
                    print(f"File does not exist: {path}")
                    return False
                
                # 获取文件扩展名并检查
                file_ext = path.suffix.lower()
                if file_ext not in allowed_exts:
                    print(f"Invalid extension: {file_ext}, allowed: {allowed_exts}")
                    return False
                return True
            
            # 仅验证扩展名
            file_ext = path.suffix.lower()
            if not file_ext:
                print(f"No extension found in filename: {filename}")
                return False
            
            return file_ext in allowed_exts

        except Exception as e:
            print(f"Error validating file extension: {e}")
            return False 