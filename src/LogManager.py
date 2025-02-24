# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
import sys
import time

class LogManager:
    _instance = None
    _lock = Lock()  # 线程安全锁
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # 初始化日志系统
                cls._instance.logger = logging.getLogger('app')
                cls._instance.logger.setLevel(logging.DEBUG)
                cls._instance._configured = False
                cls._instance._initialized = False           
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._configure_logging()
            self._initialized = True
    
    @classmethod
    def get_instance(cls):
        """获取单例实例的推荐方法"""
        return cls()
            
    def _configure_logging(self, console=True, file_path=None):
        """确保只配置一次日志"""
        if hasattr(self, '_configured'):
            return    
        # 清除已有Handler
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # 控制台Handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)
        
        # 文件Handler
        if file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)
            
        self._configured = True
    
    def _get_formatter(self):
        return logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def log_error(self, message, print_console=False):
        """记录错误并可选打印到控制台"""
        self.logger.error(message)
        if print_console:
            print(f"错误: {message}")
            
    def log_warn(self, message, print_console=False):
        """记录警告并可选打印到控制台"""
        self.logger.warn(message)
        if print_console:
            print(f"警告: {message}")
            
    def log_info(self, message, print_console=False):
        """记录信息并可选打印到控制台"""
        self.logger.info(message)
        if print_console:
            print(f"信息: {message}")
            
    def log_processing_time(self, message, start_time: float):
        """记录处理耗时"""
        elapsed = time.time() - start_time
        self.log_info(f"{message} 处理完成，耗时 {elapsed:.2f} 秒")
        
    @classmethod
    def get_instance(cls):
        return cls()
        
def setup_logging(
    console: bool = True,
    file_path: str = None,
    max_bytes: int = 10*1024*1024,  # 10MB
    backup_count: int = 5
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 清除已有Handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台Handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件Handler
    if file_path:
        file_handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)