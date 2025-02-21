
# -*- coding: utf-8 -*-

# 自定义异常
class ProcessingError(Exception):
    """处理流程基础异常"""
    
class InputError(ProcessingError):
    """输入验证失败"""

class ResourceError(ProcessingError):
    """系统资源不足"""

class TimeoutError(ProcessingError):
    """操作超时"""