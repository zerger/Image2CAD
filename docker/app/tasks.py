from celery import Celery
import time
from image2CAD import pdf_to_images, png_to_dxf

app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def process_cad_image(input_path, output_path=None):
    print(f"Processing {input_path} -> {output_path}")
    png_to_dxf(input_path, output_path)  
    
@app.task
def convert_pdf_to_images(pdf_path, output_dir=None, dpi=None):
    print(f"Converting {pdf_path} to images in {output_dir}")
    pdf_to_images(pdf_path, output_dir)
