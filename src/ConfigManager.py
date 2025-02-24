from pathlib import Path
import os
import configparser
from cryptography.fernet import Fernet
from logManager import LogManager

log_mgr = LogManager().get_instance()
class ConfigManager:
    _instance = None
    _config = None
    _config_path = 'config.ini'
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance
    
    @staticmethod
    def _encrypt_value(self, value: str) -> str:
        return self._cipher.encrypt(value.encode()).decode()
    
    @staticmethod
    def _decrypt_value(self, value: str) -> str:
        return self._cipher.decrypt(value.encode()).decode()
    
    def _init_config(self):
        """初始化配置"""
        self._config = configparser.ConfigParser()
        # 设置默认配置
        self._defaults = {
            'tesseract_path': '',
            'potrace_path': '',
            'log_level': 'INFO',
            'max_workers': max(1, os.cpu_count() // 2),
            'pdf_export_dpi': 400,
            'pdf_scale': 2.0,
            'pdf_grayscale': True,
            'pdf_output_dir': './pdf_images'
        }
        self._config['DEFAULT'] = {
            k: str(v) if isinstance(v, (int, float)) else v 
            for k, v in self._defaults.items()
        }
        # 尝试加载现有配置
        if os.path.exists(self._config_path):
            self._config.read(self._config_path)
            
        # 验证数值型配置
        self._validate_numeric_setting('pdf_export_dpi', min_val=72, max_val=1200)
        self._validate_numeric_setting('max_workers', min_val=1, max_val=os.cpu_count())
    
    @classmethod
    def _get_default_fallback(cls, key):
        """从初始化配置中获取统一默认值"""
        return cls._instance._defaults.get(key, None)
     
    def load_config(self, config_path: str) -> None:
        """加载指定配置文件"""
        self._config_path = config_path
        if not os.path.exists(config_path):
            self._create_default_config()
            return
        self._config.read(config_path)
    
    @classmethod
    def get_setting(cls, key: str, section: str = 'DEFAULT', fallback=None):
        """带类型转换和安全回退的获取方法"""
        try:
            if not cls._instance._config.has_section(section):
                raise configparser.NoSectionError(section)
            value = cls._instance._config.get(section, key)
            
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
            # 确保fallback不为None
            safe_fallback = fallback if fallback is not None else cls._get_default_fallback(key)
            log_mgr.log_warn(f"配置[{section}].{key} 不存在，使用回退值: {safe_fallback}")
            return safe_fallback
    
    @classmethod
    def set_setting(cls, key: str, value, section: str = 'DEFAULT'):
        """更新配置项并保存"""
        if not cls._instance._config.has_section(section):
            cls._instance._config.add_section(section)
        cls._instance._config.set(section, key, str(value))
        cls._save_config()
                       
    @classmethod
    def get_tesseract_path(cls) -> str:
        """获取Tesseract路径"""
        path = cls._instance._config.get('DEFAULT', 'tesseract_path', fallback='')
        if not path:
            path = cls._auto_detect_tesseract()
        return path
    
    @classmethod
    def set_tesseract_path(cls, path: str) -> None:
        """设置Tesseract路径"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tesseract路径无效: {path}")
        cls._instance._config['DEFAULT']['tesseract_path'] = path
        cls._save_config()
    
    @classmethod
    def get_potrace_path(cls) -> str:
        """获取Potrace路径"""
        path = cls._instance._config.get('DEFAULT', 'potrace_path', fallback='')
        if not path:
            path = cls._auto_detect_potrace()
        return path
    
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
        
    @classmethod
    def _auto_detect_potrace(cls)->str:        
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
                cls._potrace_path = p
                break
        else:
            raise FileNotFoundError("未找到potrace可执行文件，请通过--potrace参数指定")
        return cls._potrace_path
        
    def set_tesseract_path(self, path):
        """设置并保存Tesseract路径"""
        valid_path = self._validate_path(path)
        self.config.set(self.SECTION, 'tesseract_path', valid_path)
        with open(self.CONFIG_FILE, 'w') as f:
            self.config.write(f)
        return valid_path
    
    def _validate_path(self, path):
        """验证路径有效性"""
        exe_path = Path(path)
        if exe_path.is_dir():
            exe_path = exe_path / "tesseract.exe"
            
        if not exe_path.exists():
            raise FileNotFoundError(f"路径无效: {exe_path}")
            
        return str(exe_path.resolve())
    
    @classmethod
    def validate_pdf_settings(cls):
        """验证PDF相关配置有效性"""
        output_dir = cls.get_setting('pdf_output_dir')
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录不可写: {output_dir}")
        
        if int(cls.get_setting('pdf_export_dpi')) < 72:
            cls.set_setting('pdf_export_dpi', 72)
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
                    
    @classmethod
    def _save_config(cls) -> None:
        """保存配置到文件"""
        with open(cls._instance._config_path, 'w') as f:
            cls._instance._config.write(f)
    