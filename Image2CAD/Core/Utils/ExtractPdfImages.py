import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import argparse
import subprocess
import cv2

def pdf_to_images(pdf_path, output_dir=None):  
    pdf_Dir = os.path.abspath(pdf_path)

    dir = os.path.dirname(pdf_Dir)
    # 如果没有指定输出文件夹，则默认创建 output_svg 文件夹
    if output_dir is None:
        output_dir = os.path.join(dir, "pdfImages")
    
      # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
     # 打开 PDF 文件   
    with fitz.open(pdf_path) as doc:    
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # 检查页面是否包含图像资源
            image_list = page.get_images(full=True)

            if image_list:  # 如果页面包含图像
                print(f"页面 {page_num + 1} 是图像，直接提取并保存")
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # 将图像保存为文件
                    image_filename = f"page_{page_num + 1}_image_{img_index + 1}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    print(f"保存图像：{image_path}")
            else:  # 如果页面没有图像，则使用 pdf2image 转换为图像
                print(f"页面 {page_num + 1} 不是图像，转换为图像并保存")
                pages = convert_from_path(pdf_path, 300, first_page=page_num + 1, last_page=page_num + 1)
                image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
                pages[0].save(image_path, 'PNG')
                print(f"保存图像：{image_path}")
            
def png_to_svg(input_folder, output_folder=None):
    # 如果没有指定输出文件夹，则默认创建 output_svg 文件夹
    if output_folder is None:
        output_folder = os.path.join(input_folder, "output_svg")
    
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历 input_folder 文件夹中的所有 PNG 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".pbm"
            output_pbmPath = os.path.join(output_folder, output_filename)      
           
            with cv2.imread(input_path) as img:  
                # 将图像转换为灰度图
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 使用阈值操作将图像转换为黑白图像
                _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                # 保存二值化图像
                cv2.imwrite(output_pbmPath, binary_img)
                svg_filename = os.path.splitext(filename)[0] + ".svg"
                output_svgPath = os.path.join(output_folder, svg_filename)          
                # 执行 potrace 命令
                subprocess.run(['D:/Image2CAD/Image2CAD/potrace', output_pbmPath, '-s', '-o', output_svgPath])
                os.remove(output_pbmPath)
                print(f"Converted {input_path} to {output_svgPath}")
    print(f"完成png转svg转换，输出到 {output_folder}")
            
def main():    
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process PDF and PNG files.")
    parser.add_argument('action', choices=['pdf2images', 'png2svg'], help="Choose the action to perform: 'pdf2images' or 'png2svg'")
    parser.add_argument('input_path', help="Input file or folder path.")
    parser.add_argument('output_path', nargs='?', help="Output file or folder path. If not provided, it will be auto-generated based on input_path.")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据选择的 action 执行相应的函数
    if args.action == 'pdf2images':
        pdf_to_images(args.input_path, args.output_path)
    elif args.action == 'png2svg':
        png_to_svg(args.input_path, args.output_path)

if __name__ == "__main__":
    main()

