from pathlib import Path
import os
import configparser

class ConfigManager:
    CONFIG_FILE = "app_config.ini"
    SECTION = "Tesseract-OCR"
    _potrace_path = None
    
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(self.CONFIG_FILE)
        
        if not self.config.has_section(self.SECTION):
            self.config.add_section(self.SECTION)
    
    def get_tesseract_path(self):
        """获取验证过的Tesseract路径"""
        # 优先级：环境变量 > 配置文件 > 默认路径
        env_path = os.getenv('TESSERACT_PATH')
        if env_path:
            return self._validate_path(env_path)
            
        if self.config.has_option(self.SECTION, 'tesseract_path'):
            config_path = self.config.get(self.SECTION, 'tesseract_path')
            return self._validate_path(config_path)
            
        default_path = Path("E:/Program Files/Tesseract-OCR/tesseract.exe")
        if default_path.exists():
            return str(default_path)
            
        raise EnvironmentError("未找到有效的Tesseract路径")
    
    @classmethod
    def set_potrace_path(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Potrace路径无效: {path}")
        cls._potrace_path = path
        
    @classmethod
    def get_potrace_path(cls):
        if cls._potrace_path is None:
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
    