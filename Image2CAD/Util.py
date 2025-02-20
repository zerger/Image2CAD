# -*- coding: utf-8 -*-
import os
import platform
import ctypes
import psutil

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