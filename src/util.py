# -*- coding: utf-8 -*-
import os
import platform
import ctypes
import psutil
import shutil
from pathlib import Path
from errors import ProcessingError, InputError, ResourceError, TimeoutError
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

        if path.is_file():
            return path.suffix.lower() in extensions

        for p in path.rglob('*'):
            if p.is_file() and p.suffix.lower() in extensions:
                return True
        return False
    
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