# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys
from threading import Lock
import configparser
import argparse
from cryptography.fernet import Fernet
from logManager import LogManager, setup_logging
from util import Util
from errors import ProcessingError, InputError, ResourceError, TimeoutError

log_mgr = LogManager().get_instance()
class ConfigManager:
    _instance = None
    _config = None
    _lock = Lock()  # 线程安全锁
    _config_path = str(Path(__file__).resolve().parent.parent / "config.ini")
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls.logger = LogManager().get_instance()
                cls._instance._initialized = False  
                cls._instance.__init__()
            return cls._instance
    
    @classmethod
    def get_instance(cls):
        """获取单例实例的推荐方法"""
        return cls()
    
    @staticmethod
    def _encrypt_value(value: str) -> str:
        return ConfigManager._cipher.encrypt(value.encode()).decode()
    
    @staticmethod
    def _decrypt_value(value: str) -> str:
        return ConfigManager._cipher.decrypt(value.encode()).decode()
    
    @staticmethod
    def get_allow_imgExt():
        """从初始化配置中获取允许的图片扩展名"""
        return {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    def __init__(self):
        """初始化配置"""
        if not self._initialized:
            self._config = configparser.ConfigParser()
            # 设置默认配置
            self._defaults = {
                'tesseract_path': '',
                'potrace_path': '',
                'log_level': 'INFO',
                'max_workers': max(1, os.cpu_count() // 2),
                'pdf_export_dpi': 200,
                'pdf_scale': 2.0,
                'pdf_grayscale': True,
                'pdf_output_dir': './pdf_images',
                'max_image_pixels': 256_000_000,
                'interpolation_distance': 3, 
            }
            self._config['DEFAULT'] = {
                k: str(v) if isinstance(v, (int, float)) else v 
                for k, v in self._defaults.items()
            }
            # 尝试加载现有配置
            if os.path.exists(self._config_path):
                self._config.read(self._config_path)
            else:
                self._save_config()  # 生成默认配置
            # 立即加载默认值，防止 apply_security_settings() 访问失败
            for k, v in self._defaults.items():
                if not self._config.has_option('DEFAULT', k):
                    self._config.set('DEFAULT', k, str(v))   

            self._initialized = True
                    
    def apply_security_settings(self):
        """应用图像处理安全限制"""
        try:
            from PIL import Image
            max_pixels = self.get_setting(key='max_image_pixels', section='DEFAULT', fallback=256_000_000)
            Image.MAX_IMAGE_PIXELS = max_pixels
            log_mgr.log_info(f"设置图像安全像素限制为: {max_pixels}")
        except ImportError:
            log_mgr.log_warn("未安装Pillow库，跳过安全设置")            
            pass
        
    
    def _get_default_fallback(self, key):
        """从初始化配置中获取统一默认值"""
        return self._defaults.get(key, None)
     
    def load_config(self, config_path: str) -> None:
        """加载指定配置文件"""
        self._config_path = config_path
        if not os.path.exists(config_path):
            self._init_config()
            return
        self._config.read(config_path)    
   
    def get_setting(self, key: str, section: str = 'DEFAULT', fallback=None):
        """带类型转换和安全回退的获取方法"""
        try:
            # 确保fallback不为None
            safe_fallback = fallback if fallback is not None else self._get_default_fallback(key)
            if section != 'DEFAULT' and not self._config.has_section(section):
                raise configparser.NoSectionError(section)
            value = self._config.get(section, key)
            
            # 处理空字符串情况
            if value.strip() == '':
                raise ValueError("空值")
                
            # 自动类型转换
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
                
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        except (configparser.NoSectionError, configparser.NoOptionError) as e:            
            log_mgr.log_warn(f"配置[{section}].{key} 不存在，使用回退值: {safe_fallback}")
            return safe_fallback    
   
    def set_setting(self, key: str, value, section: str = 'DEFAULT'):      
        """更新配置项并保存"""
        if section == "DEFAULT":  # 直接写入 DEFAULT
            self._config["DEFAULT"][key] = str(value)
        else:
            if not self._config.has_section(section):
                self._config.add_section(section)
            self._config.set(section, key, str(value))       
        self._save_config()                       
    
    def get_tesseract_path(self) -> str:
        """获取Tesseract路径"""
        path = self._config.get('DEFAULT', 'tesseract_path', fallback='')
        if not path:
            path = self._auto_detect_tesseract()
        return self.normalize_path(path)    
  
    def set_tesseract_path(self, path: str) -> None:
        """设置Tesseract路径"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tesseract路径无效: {path}")
        self._config['DEFAULT']['tesseract_path'] = path
        self._save_config()    
        
    def set_tesseract_mode(self, mode: str) -> None:
        if mode in ["fast", "normal", "best"]:           
            self._config['DEFAULT']['tesseract_mode'] = mode
            self._save_config()    
        else:
            raise ValueError("Invalid mode. Choose from 'fast', 'normal', 'best'.")

    def get_tesseract_mode(self) -> str:
        mode = self._config.get('DEFAULT', 'tesseract_mode', fallback='')
        if mode in ["fast", "normal", "best"]:  
            return mode 
        else:
            return "normal"
        
    def get_tesseract_data_path(self) -> str:       
        tesseract_mode = self.get_tesseract_mode()
        return self.set_tesseract_data_path_mode(tesseract_mode)      
   
    def set_tesseract_data_path_mode(self, mode) -> str:
        tesseract_exe = self.get_tesseract_path()       
        data_dir = Path(tesseract_exe).parent / 'tessdata'
        if mode == "fast":
            data_dir = Path(tesseract_exe).parent / 'tessdata_fast'
        elif mode == "best":
            data_dir = Path(tesseract_exe).parent / 'tessdata_best'
        else:
            data_dir = Path(tesseract_exe).parent / 'tessdata'  
        return data_dir
    
    def get_potrace_path(self) -> str:
        """获取Potrace路径"""
        path = self._config.get('DEFAULT', 'potrace_path', fallback='')
        if not path:
            path = self._auto_detect_potrace()
        return self.normalize_path(path)    
    
    @staticmethod
    def _auto_detect_tesseract() -> str:
        """自动检测Tesseract路径"""
        common_paths = [
            '/usr/bin/tesseract',  # Linux
            '/usr/local/bin/tesseract',  # Mac
            'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Windows
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("未找到Tesseract可执行文件")        
  
    def _auto_detect_potrace(self)->str:        
        # 尝试自动探测常见路径
        default_paths = [
            '/usr/local/bin/potrace',  # Linux/Mac
            '/usr/bin/potrace',
            os.path.join(os.getcwd(), 'potrace'),  # 当前目录
            os.path.join(os.getcwd(), 'src/potrace'),
            'C:/Program Files/potrace/potrace.exe',  # Windows
            os.path.join(os.getcwd(), 'potrace.exe'),  # 当前目录
            os.path.join(os.getcwd(), 'src/potrace.exe')
        ]
        for p in default_paths:
            if os.path.exists(p):
                self._potrace_path = p
                break
        else:
            raise FileNotFoundError("未找到potrace可执行文件，请通过--potrace参数指定")
        return self._potrace_path
        
    def set_tesseract_path(self, path):
        """设置并保存Tesseract路径"""
        valid_path = self._validate_path(path)      
        self.set_setting('tesseract_path', valid_path)
        with open(self._config_path, 'w') as f:
            self._config.write(f)
        return valid_path
    
    def _validate_path(self, path):
        """验证路径有效性"""
        exe_path = Path(path)
        if exe_path.is_dir():
            exe_path = exe_path / "tesseract.exe"
            
        if not exe_path.exists():
            raise FileNotFoundError(f"路径无效: {exe_path}")
            
        return str(exe_path.resolve())    
  
    def validate_pdf_settings(self):
        """验证PDF相关配置有效性"""
        output_dir = self.get_setting('pdf_output_dir')
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录不可写: {output_dir}")
        
        if int(self.get_setting('pdf_export_dpi')) < 72:
            self.set_setting('pdf_export_dpi', 72)
            log_mgr.log_warn("DPI过低，已自动设置为72")
    
    def _validate_numeric_setting(self, key, min_val, max_val):
        """验证数值型配置有效性"""
        raw_value = self._config['DEFAULT'].get(key, '')
        try:
            value = int(raw_value)
            if not (min_val <= value <= max_val):
                raise ValueError
        except ValueError:
            default = int(self._config['DEFAULT'][key])
            self._config['DEFAULT'][key] = str(default)
            log_mgr.log_warn(f"配置 {key} 值 {raw_value} 无效，已重置为默认值 {default}")    
            
    def _validate_centerline_config(self):
        """验证中心线配置有效性"""
        inter_dist = float(self.get_setting('interpolation_distance'))

        # 距离范围检查
        if not (0.1 <= inter_dist <= 5.0):
            raise ValueError(f"无效插值距离: {inter_dist}，应在0.1-5.0mm之间")      
   
    def _save_config(self) -> None:
        """保存配置到文件"""
        with open(self._config_path, 'w') as f:
            self._config.write(f)
            
    def normalize_path(self, path):
        """规范化路径，去除多余的换行符和空格，并使用 os.path.normpath 进行规范化。
        规范化路径，去除多余的换行符和空格，并使用 os.path.normpath 进行规范化。
        """
        # 去除换行符和空格
        cleaned_path = path.replace('\n', '').strip()
        # 使用 os.path.normpath 规范化路径
        normalized_path = os.path.normpath(cleaned_path)
        return normalized_path
    
    @staticmethod
    def check_system_requirements():
        """系统环境检查"""
        checks = [
            ('Tesseract OCR', config_manager.get_tesseract_path()),
            ('Potrace', config_manager.get_potrace_path()),
            ('Free Disk Space', Util.get_disk_space('/')[0] > 500*1024*1024),
            ('Memory', Util.get_memory_info()[1] > 2*1024*1024)  # 2GB以上
        ]
    
        print("\n系统环境检查报告:")
        for name, status in checks:
            status_str = "✓ OK" if status else "✗ 缺失"
            print(f"{name:15} {status_str}")

        if all(status for _, status in checks):
            print("\n环境检查通过")
        else:
            print("\n警告：存在缺失的依赖项")
    
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="配置工具")
    # 主命令参数
    parser.add_argument('action', 
                        choices=['set-tesseract', 'set-potrace', 'check-env'],
                        help="""操作选项:    
    set-tesseract: 设置Tesseract OCR路径
    set-potrace  : 设置Potrace矢量转换路径
    check-env    : 检查运行环境配置""")
    
    # 通用参数
    parser.add_argument('input_path', nargs='?', 
                       help="输入文件/目录路径（对set操作为工具路径）")  
    parser.add_argument('--config', default='config.ini',
                       help="指定配置文件路径（默认: ./config.ini）")   
    try:
        config_manager = ConfigManager.get_instance()
        setup_logging()  # 初始化日志
         
        args = parser.parse_args()     
        action = args.action.lower()  
        if action == 'set-tesseract' and not args.input_path:
            raise InputError("必须指定Tesseract路径")
            
        if action == 'set-potrace' and not args.input_path:
            raise InputError("必须指定Potrace路径")
            
        # 加载配置文件
        config_manager.load_config(args.config)      
      
        # 根据选择的 action 执行       
        if action == 'set-tesseract':
            config_manager.set_tesseract_path(args.input_path)
            log_mgr.log_info(f"Tesseract路径已设置为: {config_manager.get_tesseract_path()}")            
        elif action == 'set-potrace':
            config_manager.set_potrace_path(args.input_path)
            log_mgr.log_info(f"Potrace路径已设置为: {config_manager.get_potrace_path()}")            
        elif action == 'check-env':
            config_manager.check_system_requirements()
        else:
            print("请输入正确的命令")
            
    except argparse.ArgumentError as e:
        log_mgr.log_error(f"参数错误: {e}")
        parser.print_help()
        sys.exit(1)
    except Exception as e:
        log_mgr.log_error(f"运行错误: {str(e)}")
        sys.exit(2)