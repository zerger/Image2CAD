from pathlib import Path
import os
import configparser

class ConfigManager:
    _instance = None
    _config = None
    _config_path = 'config.ini'
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance
    
    def _init_config(self):
        """初始化配置"""
        self._config = configparser.ConfigParser()
        # 设置默认配置
        self._config['DEFAULT'] = {
            'tesseract_path': '',
            'potrace_path': '',
            'log_level': 'INFO',
            'max_workers': str(os.cpu_count() // 2)
        }
        # 尝试加载现有配置
        if os.path.exists(self._config_path):
            self._config.read(self._config_path)
     
    def load_config(self, config_path: str) -> None:
        """加载指定配置文件"""
        self._config_path = config_path
        if not os.path.exists(config_path):
            self._create_default_config()
            return
        self._config.read(config_path)
               
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
    def _save_config(cls) -> None:
        """保存配置到文件"""
        with open(cls._instance._config_path, 'w') as f:
            cls._instance._config.write(f)
    