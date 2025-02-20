from celery import Celery
import time

app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@app.task
def process_cad_image(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    time.sleep(5)  # 模拟 CAD 矢量化过程
    with open(output_path, "w") as f:
        f.write("DXF data")  # 模拟 DXF 生成
    return f"Saved {output_path}"
