from celery import Celery
import time
from Image2CAD import pdf_to_images, png_to_svg

app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def process_cad_image(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    png_to_svg(input_path, output_path)  
