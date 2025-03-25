# -*- coding: utf-8 -*-

from typing import Optional

# 自定义异常
class ProcessingError(Exception):
    """基础处理错误"""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause

class InputError(ProcessingError):
    """输入错误"""
    pass

class ResourceError(ProcessingError):
    """资源错误"""
    pass

class ValidationError(ProcessingError):
    """输入验证错误"""
    pass

class TimeoutError(ProcessingError):
    """超时错误"""
    pass