import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import argparse

def pdf_to_images(pdf_path, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开 PDF 文件
    doc = fitz.open(pdf_path)
    
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
            
def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Convert PDF to images")
    parser.add_argument("pdf_path", help="The path to the PDF file")
    parser.add_argument("output_dir", help="The directory to save images")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 调用 pdf_to_images 函数
    pdf_to_images(args.pdf_path, args.output_dir)

if __name__ == "__main__":
    main()

