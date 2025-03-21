# -*- coding: utf-8 -*-
from celery import Celery
import time
from image2CAD import pdf_to_images, png_to_dxf
from ocrProcess import OCRProcess

app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def process_cad_image(input_path, output_path=None):
    print(f"Processing {input_path} -> {output_path}")
    png_to_dxf(input_path, output_path)  
    
@app.task
def convert_pdf_to_images(pdf_path, output_dir=None, dpi=None):
    print(f"Converting pdf to images {pdf_path} to {output_dir}")
    pdf_to_images(pdf_path, output_dir)
    
@app.task
def ocr_image(image_path, scale_factor=5, max_block_size=512, overlap=20, output_path=None):
    print(f"ocr Processing {image_path} -> {output_path}")
    ocr_process = OCRProcess()
    parsed_results, original_height = ocr_process.get_file_rapidOCR(image_path)
    print(parsed_results)
    print("ocr_image finished")
    return parsed_results, original_height