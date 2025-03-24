# -*- coding: utf-8 -*-
import os
import platform
import ctypes
import numpy as np
import cv2
import psutil
import shutil
from pathlib import Path
from typing import Union
from src.common.errors import ProcessingError, InputError, ResourceError, TimeoutError

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
    def get_allow_imgExt():
        """从初始化配置中获取允许的图片扩展名"""
        return {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', ".webp"}
    
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
        try:
            # 1. path 变量在 else 分支中定义，但在 if 分支中未定义
            if hasattr(input_path, 'filename'):
                filename = input_path.filename
                path = Path(filename)
            else:
                filename = str(input_path)
                path = Path(filename)
            
            # 2. 统一获取绝对路径
            path = path.absolute()
            
            # 3. 添加错误信息的具体内容
            if not path.exists():
                raise InputError(f"输入文件不存在: {str(path)}")
            if not path.is_file():
                raise InputError(f"输入路径不是文件: {str(path)}")
            if not Util.validate_extname(str(path), Util.get_allow_imgExt()):
                raise InputError(f"不支持的文件格式: {path.suffix}，支持的格式：{Util.get_allow_imgExt()}")
            
            # 4. 文件大小检查（保持不变）
            if path.stat().st_size > 10 * 1024 * 1024:  # 10MB限制
                raise InputError(f"文件大小超过10MB限制：{path.stat().st_size / (1024*1024):.2f}MB")
            
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"文件验证失败: {str(e)}")
        
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
        # 获取输出路径的扩展名
        output_ext = Path(output_image_path).suffix.lower()    
        # 如果输出路径没有扩展名，使用传入的 ext
        if not output_ext:
            output_ext = ext.lower()
            output_image_path = str(Path(output_image_path).with_suffix(output_ext))
        if output_ext not in Util.get_allow_imgExt() and output_ext not in ['.pbm']: 
            raise InputError(f"不支持的文件格式: {ext}")
        if Util.validate_extname(output_image_path, Util.get_allow_imgExt()):
            success, encoded_image = cv2.imencode(ext, img)
            if success:    
                # 创建输出目录
                output_dir = Path(output_image_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)      
                  
                image_bytes = encoded_image.tobytes()
                with open(output_image_path, 'wb') as f:
                    f.write(image_bytes)                
            else:
                print("Failed to encode image")
        elif Util.validate_extname(output_image_path, ['.pbm']):            
           Util.save_as_pbm(img, output_image_path)
        else:
            print("Failed to validate image file")            
                    
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
            # 仅验证扩展名
            file_ext = path.suffix.lower()
            if not file_ext:
                print(f"No extension found in filename: {filename}")
                return False
            
            return file_ext in allowed_exts

        except Exception as e:
            print(f"Error validating file extension: {e}")
            return False 

    @staticmethod
    def save_as_pbm(img, output_path):
        try:
            # 确保图像是二值图像
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            # 转换为 0 和 1 (注意：在 PBM 中，1 表示黑色)
            binary = (binary == 0).astype(np.uint8)
            
            # 计算每行需要的字节数（8位一组）
            width_bytes = (binary.shape[1] + 7) // 8
            
            # 创建输出目录
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 写入 PBM 文件
            with open(output_path, 'wb') as f:
                # 写入 PBM 头部
                f.write(b'P4\n')  # P4 表示二进制 PBM
                f.write(f'{binary.shape[1]} {binary.shape[0]}\n'.encode())
                
                # 将位打包成字节并写入
                for row in binary:
                    packed_row = np.packbits(row[:width_bytes * 8])
                    f.write(packed_row.tobytes())
                    
            return True
        except Exception as e:
            print(f"保存 PBM 文件时出错: {str(e)}")
            return False 