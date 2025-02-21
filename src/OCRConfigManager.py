from pathlib import Path
import os
import configparser

class OCRConfigManager:
    CONFIG_FILE = "app_config.ini"
    SECTION = "Tesseract-OCR"
    
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
    