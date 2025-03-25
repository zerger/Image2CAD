# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from threading import Lock
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Union, Dict
from enum import Enum

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
                cls._instance._loggers: Dict[str, logging.Logger] = {}
            return cls._instance

    def __init__(self):
        """初始化日志管理器"""
        if not self._initialized:
            self._default_config = {
                'console_level': LogLevel.INFO,
                'file_level': LogLevel.DEBUG,
                'log_dir': Path('logs'),
                'max_bytes': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5,
                'encoding': 'utf-8',
                'date_format': '%Y-%m-%d %H:%M:%S'
            }
            
            # 确保日志目录存在
            self._default_config['log_dir'].mkdir(parents=True, exist_ok=True)
            
            # 初始化默认日志器
            self.setup_logger('app')
            self._initialized = True
            
    def setup_logger(self, 
                    name: str,
                    console: bool = True,
                    file: bool = True,
                    rotation: str = 'size',  # 'size' or 'time'
                    **kwargs) -> logging.Logger:
        """设置新的日志器
        
        Args:
            name: 日志器名称
            console: 是否输出到控制台
            file: 是否输出到文件
            rotation: 日志轮转方式 ('size' 或 'time')
            **kwargs: 其他配置参数
        
        Returns:
            配置好的日志器
        """
        try:
            with self._lock:
                # 如果已存在，先移除
                if name in self._loggers:
                    self.remove_logger(name)
                
                # 创建新的日志器
                logger = logging.getLogger(name)
                logger.setLevel(logging.DEBUG)
                
                # 合并配置
                config = self._default_config.copy()
                config.update(kwargs)
                
                # 添加控制台处理器
                if console:
                    console_handler = self._create_console_handler(config)
                    logger.addHandler(console_handler)
                
                # 添加文件处理器
                if file:
                    file_handler = self._create_file_handler(name, rotation, config)
                    logger.addHandler(file_handler)
                
                self._loggers[name] = logger
                return logger
                
        except Exception as e:
            sys.stderr.write(f"设置日志器失败: {str(e)}\n")
            raise

    def _create_console_handler(self, config: dict) -> logging.Handler:
        """创建控制台处理器"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(config['console_level'].value)
        handler.setFormatter(self._create_formatter(config, is_console=True))
        return handler

    def _create_file_handler(self, name: str, rotation: str, config: dict) -> logging.Handler:
        """创建文件处理器"""
        log_file = config['log_dir'] / f"{name}.log"
        
        if rotation == 'size':
            handler = RotatingFileHandler(
                filename=str(log_file),
                maxBytes=config['max_bytes'],
                backupCount=config['backup_count'],
                encoding=config['encoding']
            )
        else:  # time rotation
            handler = TimedRotatingFileHandler(
                filename=str(log_file),
                when='midnight',
                interval=1,
                backupCount=config['backup_count'],
                encoding=config['encoding']
            )
            
        handler.setLevel(config['file_level'].value)
        handler.setFormatter(self._create_formatter(config))
        return handler

    def _create_formatter(self, config: dict, is_console: bool = False) -> logging.Formatter:
        """创建日志格式器"""
        if is_console:
            return logging.Formatter(
                '%(levelname)s - %(message)s'
            )
        return logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt=config['date_format']
        )

    def get_logger(self, name: str = 'app') -> logging.Logger:
        """获取指定的日志器"""
        return self._loggers.get(name, self._loggers['app'])

    def remove_logger(self, name: str):
        """移除指定的日志器"""
        if name in self._loggers:
            logger = self._loggers[name]
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            del self._loggers[name]

    def log(self, 
            level: Union[LogLevel, str], 
            message: str, 
            logger_name: str = 'app',
            exc_info: bool = False,
            print_console: bool = False):
        """统一的日志记录方法
        
        Args:
            level: 日志级别
            message: 日志消息
            logger_name: 日志器名称
            exc_info: 是否包含异常信息
            print_console: 是否同时打印到控制台
        """
        try:
            if isinstance(level, str):
                level = LogLevel[level.upper()]
            
            logger = self.get_logger(logger_name)
            logger.log(level.value, message, exc_info=exc_info)
            
            if print_console:
                print(f"{level.name}: {message}")
                
        except Exception as e:
            sys.stderr.write(f"记录日志失败: {str(e)}\n")

    def log_exception(self, e: Exception, logger_name: str = 'app'):
        """记录异常信息
        
        Args:
            e: 异常对象
            logger_name: 日志器名称
        """
        func_name = sys._getframe(1).f_code.co_name
        tb_info = traceback.format_exc()
        self.log(
            LogLevel.ERROR,
            f"Exception in {func_name}: {str(e)}\nTraceback:\n{tb_info}",
            logger_name,
            exc_info=True
        )

    def log_processing_time(self, 
                          message: str, 
                          start_time: float,
                          logger_name: str = 'app'):
        """记录处理时间
        
        Args:
            message: 处理描述
            start_time: 开始时间戳
            logger_name: 日志器名称
        """
        elapsed = time.time() - start_time
        self.log(
            LogLevel.INFO,
            f"{message} 完成，耗时 {elapsed:.2f} 秒",
            logger_name
        )

    def log_debug(self, message: str, logger_name: str = 'app'):
        """记录调试信息"""
        self.log(LogLevel.DEBUG, message, logger_name)

    def log_info(self, message: str, logger_name: str = 'app'):
        """记录一般信息"""
        self.log(LogLevel.INFO, message, logger_name)

    def log_warn(self, message: str, logger_name: str = 'app'):
        """记录警告信息"""
        self.log(LogLevel.WARNING, message, logger_name)

    def log_error(self, message: str, logger_name: str = 'app', exc_info: bool = True):
        """记录错误信息，包含堆栈跟踪
        
        Args:
            message: 错误信息
            logger_name: 日志器名称
            exc_info: 是否包含异常信息，默认为True
        """
        func_name = sys._getframe(1).f_code.co_name
        tb_info = traceback.format_exc()
        if tb_info and tb_info != 'NoneType: None\n':  # 如果有异常堆栈
            self.log(
                LogLevel.ERROR,
                f"Error in {func_name}: {message}\nTraceback:\n{tb_info}",
                logger_name,
                exc_info=exc_info
            )
        else:  # 如果没有异常堆栈，至少记录调用栈
            stack = ''.join(traceback.format_stack()[:-1])
            self.log(
                LogLevel.ERROR,
                f"Error in {func_name}: {message}\nCall Stack:\n{stack}",
                logger_name,
                exc_info=exc_info
            )

    def log_critical(self, message: str, logger_name: str = 'app', exc_info: bool = True):
        """记录严重错误信息，包含堆栈跟踪
        
        Args:
            message: 严重错误信息
            logger_name: 日志器名称
            exc_info: 是否包含异常信息，默认为True
        """
        func_name = sys._getframe(1).f_code.co_name
        tb_info = traceback.format_exc()
        if tb_info and tb_info != 'NoneType: None\n':  # 如果有异常堆栈
            self.log(
                LogLevel.CRITICAL,
                f"Critical Error in {func_name}: {message}\nTraceback:\n{tb_info}",
                logger_name,
                exc_info=exc_info
            )
        else:  # 如果没有异常堆栈，至少记录调用栈
            stack = ''.join(traceback.format_stack()[:-1])
            self.log(
                LogLevel.CRITICAL,
                f"Critical Error in {func_name}: {message}\nCall Stack:\n{stack}",
                logger_name,
                exc_info=exc_info
            )

# 创建全局实例
log_mgr = LogManager()