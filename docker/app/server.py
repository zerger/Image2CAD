from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil
from tasks import process_cad_image, convert_pdf_to_images  # Celery 任务

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload/image/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_dxf = os.path.join(OUTPUT_DIR, file.filename.replace(".png", ".dxf"))

    # 异步执行 CAD 处理任务
    task = process_cad_image.delay(file_path, output_dxf)

    return {"task_id": task.id, "message": "Processing started"}

@app.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_images_dir = os.path.join(OUTPUT_DIR, file.filename.replace(".pdf", "_images"))
    # 异步执行 PDF 转换任务
    task = convert_pdf_to_images.delay(file_path, output_images_dir)

    return {"task_id": task.id, "message": "PDF to images processing started"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = process_cad_image.AsyncResult(task_id)
    return {"status": task.status, "result": task.result if task.ready() else "Processing"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(file_path, media_type='application/octet-stream', filename=filename)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
