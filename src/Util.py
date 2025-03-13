# -*- coding: utf-8 -*-
import os
import platform
import ctypes
import psutil
from pathlib import Path

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