# -*- coding: utf-8 -*-
import os
import cv2
import subprocess
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from configManager import ConfigManager
from util import Util

config_manager = ConfigManager.get_instance()

class TrainSHX_data:
    def __init__(self, shx_font=None, ttf_font=None, output_dir=None, train_model=None  ):
        self.shx_font = shx_font
        self.ttf_font = ttf_font       
        # 检查 output_dir 是否为空
        if not output_dir:
            # 设置默认输出目录为相对路径
            self.output_dir  = TrainSHX_data.get_default_output_dir()
        else:
            self.output_dir = output_dir
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)       
        self.train_model = train_model

    @staticmethod
    def get_default_output_dir():
        return Path(__file__).parent.parent / 'TestData' / 'ground-truth'
    
    @staticmethod  
    def is_valid_text_file(file_path):
        """
        检查文件是否存在并且是有效的文本文件。

        :param file_path: 文件路径
        :return: 如果文件有效且为文本文件，返回 True；否则返回 False
        """
        if not os.path.isfile(file_path):
            print(f"错误：文件 {file_path} 不存在。")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # 尝试读取前 1024 字节
            return True
        except (UnicodeDecodeError, IOError) as e:
            print(f"错误：无法读取文件 {file_path}，可能不是有效的文本文件。")
            return False
    
    @staticmethod
    def create_training_textdata(input, output):
        if os.path.isdir(input) and os.path.isdir(output):      
            filenames = [filename for filename in os.listdir(input) if filename.endswith(".txt")]
            for filename in filenames:
                TrainSHX_data.convert_mixed_text_to_grouped_text(os.path.join(input, filename), os.path.join(output, filename))
        elif os.path.isfile(input) and os.path.isfile(output):
           TrainSHX_data.convert_mixed_text_to_grouped_text(input, output)
        else:
            print(f"文件 {input} 无效或不是文本文件。")
            return          
              
    @staticmethod  
    def convert_mixed_text_to_grouped_text(input_file, output_file):
        """
        将输入文件中的 Unicode 编号和汉字格式转换为每行 10 组，每组 5 个汉字的格式。
        在输出文件的最前面添加指定的字符集，并将特殊字符放在单独的一行。
    
        :param input_file: 输入文件名，包含 Unicode 编号和汉字格式
        :param output_file: 输出文件名，保存转换后的格式
        """
        
        # 检查文件是否有效
        if not TrainSHX_data.is_valid_text_file(input_file):
            print(f"文件 {input_file} 无效或不是文本文件。")
            return        
    
        # 指定要添加的字符集
        prefix_chars_numbers_letters = "0123456789\nABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz"
        prefix_chars_special = "Ø ∠ ° ± × ÷ ∑ ∆ ∇ ㎡ ㎥"
    
        # 打开输入文件读取内容
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    
        # 打开输出文件准备写入
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入数字和字母
            f.write(prefix_chars_numbers_letters + '\n')
            # 写入特殊字符
            f.write(prefix_chars_special + '\n\n')
    
            # 用于存储每行的汉字
            line_content = []
            # 计数器，用于控制每行的汉字数量
            count = 0
    
            # 遍历每一行
            for line in lines:
                # 跳过注释和空行
                if line.strip().startswith('#') or not line.strip():
                    continue
    
                # 提取汉字
                # 判断行的格式
                if '##' in line:
                    # 处理 `19968:##一##` 形式
                    char = line.split('##')[1].strip()
                else:
                    # 处理 `丏` 形式
                    char = line.strip()
                line_content.append(char)
    
                # 每行10组，每组5个汉字
                if len(line_content) == 5:
                    f.write(' '.join(line_content) + ' ')
                    line_content = []
                    count += 1
    
                    # 每4组换行
                    if count == 4:
                        f.write('\n')
                        count = 0
    
            # 如果最后一行不足10组，仍然写入
            if line_content:
                f.write(' '.join(line_content) + '\n')
    
        print("转换完成，结果已保存到", output_file)
    
    def convert_shx_to_ttf(self) -> bool:
        """通过子进程调用 FontForge 进行 SHX 到 TTF 的转换"""
        try:
            # 生成临时 Python 脚本
            script_content = (
            "import fontforge\n"
            # f"font = fontforge.open(r'{self.shx_font.replace("'", r"\'")}')\n"
            # f"font.generate(r'{self.ttf_font.replace("'", r"\'")}')\n"
            )
            script_path = Path(__file__).parent / "_temp_convert.py"    
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            # 根据系统环境选择执行方式
            if sys.platform == "win32":
                # Windows 需要指定 MSYS2 环境路径
                python_path = "/mingw64/bin/python"
                cmd = [
                    r"E:/msys64/usr/bin/bash.exe",
                    "-l",
                    "-c",
                    f"{python_path} {script_path.as_posix()}"
                ]
            else:
                # Linux/macOS
                cmd = ["python3", script_path]

            # 执行转换命令
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # 清理临时脚本
            os.remove(script_path)

            print(f"✅ TTF 转换完成: {self.ttf_font}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 转换失败: {e.stderr}")
            return False
        except Exception as e:
            print(f"⚠️ 发生意外错误: {str(e)}")
            return False

    def generate_training_data(self):
        """生成 OCR 训练数据"""
        os.makedirs(self.output_dir, exist_ok=True)
        train_texts = [
            "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "简体中文 繁体字 形字体 OCR 训练",
            "Ø ∠ ° ± × ÷ ∑ ∆ ∇ ㎡ ㎥"
        ]

        print("🚀 正在生成训练数据...")
        for i, text in enumerate(train_texts):
            img = Image.new("L", (600, 100), 255)  # 白底黑字
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(self.ttf_font, 32)

            draw.text((10, 30), text, font=font, fill=0)
            img.save(f"{self.output_dir}/sample-{i}.png")

            with open(f"{self.output_dir}/sample-{i}.gt.txt", "w", encoding="utf-8") as f:
                f.write(text)

        print("✅ 训练数据生成完成！")

    def train_tesseract(self):
        """训练 Tesseract"""
        print("🚀 开始训练 Tesseract...")       
        
        # 配置路径
        TESSERACT_PATH = str(Path(config_manager.get_tesseract_path()).parent)
        TESSDATA_PATH = config_manager.set_tesseract_data_path_mode("best")       
        TRAIN_DIR = r"D:/Image2CADPy/tesseract_train"  # 训练数据存储目录
        FONTS_DIR = os.path.join(TRAIN_DIR, "fonts") # 字体目录
        TRAIN_TEXT = os.path.join(TRAIN_DIR, "train_text.txt")
        MAX_ITERATIONS = 400  # 训练轮数

        # 确保训练目录存在
        os.makedirs(TRAIN_DIR, exist_ok=True)
        
        # 校验和保护
        Util.ensure_directory_exists(TRAIN_DIR)
        Util.ensure_directory_exists(FONTS_DIR)
        Util.ensure_file_exists(TRAIN_TEXT)        
        
         # 提取基础模型 LSTM
        base_lstm = os.path.join(TRAIN_DIR, "chi_sim.lstm")
        subprocess.run([
            os.path.join(TESSERACT_PATH, "combine_tessdata"),
            "-e", os.path.join(TESSDATA_PATH, "chi_sim.traineddata"), base_lstm
        ])
        
        # 获取所有 TTF 字体
        fonts = [f for f in os.listdir(FONTS_DIR) if f.endswith(".ttf")]

        for font_file in fonts:
            font_name = os.path.splitext(font_file)[0]  # 获取字体名称
            font_path = os.path.join(FONTS_DIR, font_file)

            print(f"开始训练字体: {font_name}...")

            output_prefix = os.path.join(TRAIN_DIR, font_name)

            # 1. 生成训练图片
            subprocess.run([
                os.path.join(TESSERACT_PATH, "text2image"),
                "--text", TRAIN_TEXT,
                "--outputbase", output_prefix,
                "--font", font_name,
                "--fonts_dir", FONTS_DIR,
                "--ptsize", "32"
            ])

            # 2. 生成 .box 文件
            subprocess.run([
                os.path.join(TESSERACT_PATH, "tesseract"),
                f"{output_prefix}.tif", output_prefix, "box.train"
            ])

            # 3. 生成 LSTMF 文件
            subprocess.run([
                os.path.join(TESSERACT_PATH, "tesseract"),
                f"{output_prefix}.tif", output_prefix, "lstm.train"
            ])
           
            # 4. 训练 LSTM
            trained_output = os.path.join(TRAIN_DIR, f"{font_name}.traineddata")
            subprocess.run([
                os.path.join(TESSERACT_PATH, "lstmtraining"),
                "--model_output", trained_output,
                "--traineddata", os.path.join(TESSDATA_PATH, "chi_sim.traineddata"),
                "--train_listfile", f"{output_prefix}.lstmf",
                "--continue_from", base_lstm,
                "--max_iterations", str(MAX_ITERATIONS)
            ])

            print(f"字体 {font_name} 训练完成，生成 {trained_output}")

        print("所有字体训练完成！")
    
    @staticmethod
    def generate_box_file(image_path, box_file_path, min_width=5, min_height=5):
        # 读取图像
        image = Util.opencv_read(image_path)
        image_height = image.shape[0]
        # 二值化图像
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        with open(box_file_path, 'w') as f:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # 过滤掉小边框
                if w >= min_width and h >= min_height:
                    # 转换 Y 坐标
                    y = image_height - (y + h)

                    # 假设每个轮廓是一个字符，您需要根据实际情况调整
                    char = 'A'  # 这里需要替换为实际的字符
                    f.write(f"{char} {x} {y} {x+w} {y+h} 0\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrainSHX_data 工具")
    subparsers = parser.add_subparsers(dest='command')

    # 添加 convert 子命令
    convert_parser = subparsers.add_parser('convert_mixed_text_to_grouped_text', help='转换 Unicode 编号和汉字格式')
    convert_parser.add_argument('input_file', type=str, help='输入文件路径')
    convert_parser.add_argument('output_file', type=str, help='输出文件路径')

    create_parser = subparsers.add_parser('create_training_textdata', help='创建训练数据文本文件')
    create_parser.add_argument('input_dir', type=str, help='输入目录路径')
    create_parser.add_argument('output_dir', type=str, help='输出文件路径')
    
    create_parser = subparsers.add_parser('generate_box_file', help='创建box文件')
    create_parser.add_argument('input_file', type=str, help='输入文件路径')
    create_parser.add_argument('output_file', type=str, help='输出文件路径')
    
    create_parser = subparsers.add_parser('train_tesseract', help='创建训练数据文本文件')
    
    # 解析命令行参数
    args = parser.parse_args()

    if args.command == 'convert_mixed_text_to_grouped_text':
        TrainSHX_data.convert_mixed_text_to_grouped_text(args.input_file, args.output_file)
    elif args.command == 'create_training_textdata':
        TrainSHX_data.create_training_textdata(args.input_dir, args.output_dir)
    elif args.command == 'generate_box_file':
        TrainSHX_data.generate_box_file(args.input_file, args.output_file)
    elif args.command == 'train_tesseract':
        train_shx = TrainSHX_data()
        train_shx.train_tesseract()
    else:
        print("请输入正确的命令")
                
                       

       

