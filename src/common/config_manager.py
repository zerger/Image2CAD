# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys
from threading import Lock
import configparser
import argparse
from cryptography.fernet import Fernet
from src.common.log_manager import LogManager, log_mgr
from src.common.utils import Util
from src.common.errors import ProcessingError, InputError, ResourceError, TimeoutError
class ConfigManager:
    _instance = None
    _config = None
    _lock = Lock()  # 线程安全锁
    _config_path = str(Path(__file__).resolve().parent.parent / "config.ini")
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls.logger = log_mgr
                cls._instance._initialized = False  
                cls._instance.__init__()
            return cls._instance
    
    @classmethod
    def get_instance(cls):
        """获取单例实例的推荐方法"""
        return cls()
    @staticmethod
    def get_config():
        """获取配置对象"""
        if ConfigManager._config is None:
            instance = ConfigManager()  # 确保初始化
            if ConfigManager._config is None:  # 如果还是None，说明需要显式初始化
                instance.__init__()
        return ConfigManager._config        
    
    def get_max_workers(self):
        config = self.get_config()
        return config.getint('DEFAULT', 'max_workers', fallback=4)        
   
    def get_task_timeout(self):
        """获取任务超时时间（分钟）"""
        config = self.get_config()
        return config.getint('DEFAULT', 'task_timeout_minutes', fallback=30)    
   
    def _decrypt_value(self, value: str) -> str:
        """解密配置值
        Args:
            value: 待解密的字符串
        Returns:
            解密后的字符串，如果解密失败则返回原值
        """
        try:
            if not value:
                return value
                
            # 尝试解密
            decrypted = self._cipher.decrypt(value.encode())
            return decrypted.decode()
            
        except Exception as e:
            # 解密失败时记录错误并返回原值
            print(f"解密失败: {str(e)}")
            return value    
    
    def get_allow_imgExt(self):
        """从初始化配置中获取允许的图片扩展名"""
        return {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self):
        """初始化配置"""
        if not getattr(self, '_initialized', False):
            ConfigManager._config = configparser.ConfigParser()
            self._config = ConfigManager._config
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
                'ocr_mode': 'normal',
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
        
    
    def _get_default_fallback(self, key: str):
        """获取默认值，增加类型检查
        Args:
            key: 配置键名
        Returns:
            默认值
        """
        if not hasattr(self, '_defaults'):
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
                'ocr_mode': 'normal',
            }
        return self._defaults.get(key)
     
    def load_config(self, config_path: str) -> None:
        """加载指定配置文件"""
        self._config_path = config_path
        if not os.path.exists(config_path):
            self._init_config()
            return
        self._config.read(config_path)    
   
    def get_setting(self, key: str, section: str = 'DEFAULT', fallback=None):
        """获取配置项，增强错误处理和类型转换
        Args:
            key: 配置键名
            section: 配置区段名，默认为 DEFAULT
            fallback: 默认值
        Returns:
            转换后的配置值
        """
        try:
            # 参数验证
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"无效的配置键名: {key}")
            
            if not isinstance(section, str) or not section.strip():
                raise ValueError(f"无效的配置区段名: {section}")
            
            # 获取安全的默认值
            safe_fallback = fallback if fallback is not None else self._get_default_fallback(key)
            
            # 检查配置区段
            if section != 'DEFAULT' and not self._config.has_section(section):
                log_mgr.log_warn(f"配置区段[{section}]不存在，使用默认值: {safe_fallback}")
                return safe_fallback
            
            # 获取配置值
            value = self._config.get(section, key, fallback=None)
            if value is None:
                return safe_fallback
            
            # 空值处理
            if not value.strip():
                return safe_fallback
            
            # 类型转换
            return self._convert_value_type(value)
            
        except Exception as e:
            log_mgr.log_error(f"获取配置[{section}].{key}失败: {str(e)}")
            return safe_fallback
    
    def set_setting(self, key: str, value, section: str = 'DEFAULT'):
        """线程安全的配置更新"""
        with self._lock:
            try:
                if section != "DEFAULT":
                    if not self._config.has_section(section):
                        self._config.add_section(section)
                self._config[section][key] = str(value)
                self._save_config()
            except Exception as e:
                log_mgr.log_error(f"更新配置失败: {str(e)}")
                raise               
    
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
        
    def set_ocr_mode(self, mode: str) -> None:
        if mode in ["fast", "normal", "best"]:           
            self._config['DEFAULT']['ocr_mode'] = mode
            self._save_config()    
        else:
            raise ValueError("Invalid mode. Choose from 'fast', 'normal', 'best'.")

    def get_ocr_mode(self) -> str:
        mode = self._config.get('DEFAULT', 'ocr_mode', fallback='')
        if mode in ["fast", "normal", "best"]:  
            return mode 
        else:
            return "normal"
        
    def get_tesseract_data_path(self) -> str:       
        ocr_mode = self.get_ocr_mode()
        return self.set_tesseract_data_path_mode(ocr_mode)      
   
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
    
    def _auto_detect_tesseract(self) -> str:
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
    
    def get_server_host(self) -> str:
        """获取服务器主机地址"""
        return self.get_setting('server_host', fallback='127.0.0.1')
    
    def get_server_port(self) -> int:
        """获取服务器端口"""
        return int(self.get_setting('server_port', fallback=8000))
    
    def get_file_retention_days(self) -> int:
        """获取文件保留天数"""
        return int(self.get_setting('file_retention_days', fallback=7))    
    
    def check_system_requirements(self):
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
    
    def get_rapidocr_params(self, mode='normal'):
        """获取OCR参数配置
        Args:
            mode (str): 'fast', 'normal', 或 'best'
        Returns:
            dict: OCR参数配置
        """
        # 基础参数配置
        base_params = {
            "Global.use_det": True,
            "Global.use_cls": True,
            "Global.use_rec": True,
        }
        
        # 快速模式 - 优先速度
        fast_params = {
            **base_params,
            "Global.text_score": 0.3,
            "Global.max_side_len": 2048,
            "Global.min_side_len": 8,
            
            "Det.box_thresh": 0.3,
            "Det.unclip_ratio": 1.6,
            "Det.det_db_thresh": 0.3,
            "Det.det_db_box_thresh": 0.3,
            "Det.det_limit_side_len": 2048,
            
            "Rec.rec_batch_num": 6,
            "Rec.rec_thresh": 0.3,
            "Rec.rec_image_shape": "3, 32, 320",
            
            "Cls.cls_thresh": 0.9,
            "Cls.cls_batch_num": 6,
            
            "EngineConfig.use_fp16": True,
            "EngineConfig.enable_mkldnn": True,
            "EngineConfig.cpu_math_library_num_threads": 4
        }
        
        # 普通模式 - 平衡速度和精度
        normal_params = {
            **base_params,
            "Global.text_score": 0.2,
            "Global.max_side_len": 4096,
            "Global.min_side_len": 4,
            
            "Det.box_thresh": 0.25,
            "Det.unclip_ratio": 2.0,
            "Det.det_db_thresh": 0.2,
            "Det.det_db_box_thresh": 0.2,
            "Det.det_limit_side_len": 4096,
            
            "Rec.rec_batch_num": 3,
            "Rec.rec_thresh": 0.2,
            "Rec.rec_image_shape": "3, 48, 640",
            
            "Cls.cls_thresh": 0.9,
            "Cls.cls_batch_num": 3,
            
            "EngineConfig.use_fp16": False,
            "EngineConfig.enable_mkldnn": True,
            "EngineConfig.cpu_math_library_num_threads": 4
        }
        
        # 精细模式 - 优先精度
        best_params = {
            **base_params,
            "Global.text_score": 0.05,
            "Global.max_side_len": 16384,
            "Global.min_side_len": 2,
            
            "Det.box_thresh": 0.15,
            "Det.unclip_ratio": 3.0,
            "Det.det_db_thresh": 0.05,
            "Det.det_db_box_thresh": 0.05,
            "Det.det_limit_side_len": 16384,
            
            "Rec.rec_batch_num": 1,
            "Rec.rec_thresh": 0.05,
            "Rec.rec_image_shape": "3, 96, 960",
            "Rec.rec_algorithm": "SVTR_LCNet",
            
            "Cls.cls_thresh": 0.95,
            "Cls.cls_batch_num": 1,
            
            "EngineConfig.use_fp16": False,
            "EngineConfig.enable_mkldnn": True,
            "EngineConfig.cpu_math_library_num_threads": 4
        }
        
        # 根据模式选择参数
        params_map = {
            'fast': fast_params,
            'normal': normal_params,
            'best': best_params
        }
        
        return params_map.get(mode, normal_params)
   
    def get_rapidocr_runtime_params(self, mode='normal'):
        """获取OCR运行时参数
        Args:
            mode (str): 'fast', 'normal', 或 'best'
        Returns:
            tuple: (text_score, box_thresh, unclip_ratio)
        """
        mode_thresholds = {
            'fast': (0.3, 0.3, 1.6),
            'normal': (0.2, 0.25, 2.0),
            'best': (0.05, 0.15, 3.0)
        }
        return mode_thresholds.get(mode, mode_thresholds['normal'])

    def _convert_value_type(self, value: str):
        """智能类型转换配置值
        Args:
            value: 要转换的字符串值
        Returns:
            转换后的值
        """
        if not isinstance(value, str):
            return value
            
        # 去除首尾空格
        value = value.strip()
        
        # 布尔值处理
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        if value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # 数值处理
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # 列表处理
        if value.startswith('[') and value.endswith(']'):
            try:
                items = [item.strip() for item in value[1:-1].split(',')]
                return [self._convert_value_type(item) for item in items if item]
            except:
                pass
        
        # 默认返回字符串
        return value

    def validate_config(self):
        """验证所有配置项的有效性"""
        try:
            # 验证必要路径
            self._validate_paths()
            
            # 验证数值范围
            self._validate_numeric_ranges()
            
            # 验证目录权限
            self._validate_directories()
            
            # 验证系统资源
            self._validate_system_resources()
            
            return True
        except Exception as e:
            log_mgr.log_error(f"配置验证失败: {str(e)}")
            return False

    def _validate_paths(self):
        """验证路径配置"""
        required_paths = {
            'tesseract_path': '未找到Tesseract执行文件',
            'potrace_path': '未找到Potrace执行文件'
        }
        
        for key, error_msg in required_paths.items():
            path = self.get_setting(key)
            if not path or not Path(path).exists():
                raise ValueError(f"{error_msg}: {path}")

    def _encrypt_sensitive_value(self, value: str) -> str:
        """加密敏感配置值"""
        try:
            if not value or not isinstance(value, str):
                return value
            
            if not hasattr(self, '_cipher'):
                key = Fernet.generate_key()
                self._cipher = Fernet(key)
            
            return self._cipher.encrypt(value.encode()).decode()
        except Exception as e:
            log_mgr.log_error(f"加密失败: {str(e)}")
            return value

    def set_secure_setting(self, key: str, value: str, section: str = 'DEFAULT'):
        """设置加密的配置项"""
        encrypted_value = self._encrypt_sensitive_value(value)
        self.set_setting(key, encrypted_value, section)

    def backup_config(self):
        """备份当前配置"""
        try:
            backup_path = Path(self._config_path).with_suffix('.bak')
            with open(backup_path, 'w') as f:
                self._config.write(f)
            return str(backup_path)
        except Exception as e:
            log_mgr.log_error(f"配置备份失败: {str(e)}")
            return None

    def restore_config(self, backup_path: str = None):
        """从备份恢复配置"""
        try:
            if not backup_path:
                backup_path = Path(self._config_path).with_suffix('.bak')
            if not Path(backup_path).exists():
                raise FileNotFoundError(f"备份文件不存在: {backup_path}")
            
            self._config.read(backup_path)
            self._save_config()
            return True
        except Exception as e:
            log_mgr.log_error(f"配置恢复失败: {str(e)}")
            return False

# 全局单例
config_manager = ConfigManager.get_instance()

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
