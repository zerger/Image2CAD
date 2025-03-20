# -*- coding: utf-8 -*-
from ctypes import util
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import shutil
from util import Util
from configManager import ConfigManager
from tasks import process_cad_image, convert_pdf_to_images, ocr_image  # Celery 任务

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

allow_imgExt = ConfigManager.get_allow_imgExt()
    
@app.post("/upload/image/ocr/")
async def upload_ocr_image(file: UploadFile = File(...), task_name: str = Form(...)):
    if task_name != "ocr_image":
        return {"error": "Unknown task type"}
    file_path = os.path.abspath(os.path.join(UPLOAD_DIR, file.filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_images_dir = os.path.abspath(os.path.join(OUTPUT_DIR, file.filename.replace(".png", "_images")))
    print(file_path)
    print(output_images_dir)

    # 异步执行 OCR 识别任务
    task = ocr_image.delay(file_path, output_images_dir)

    return {"task_id": task.id, "message": "ocr_image processing started"}

@app.post("/upload/image/")
async def upload_file(file: UploadFile = File(...), task_name: str = Form(...)):
    if task_name != "png_to_dxf" and task_name != "ocr_image":
        return {"error": "Unknown task type"}
    if not util.validate_input_path(file.filename, allow_imgExt):
        return {"error": "Invalid file type"}
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if task_name == "png_to_dxf":
        output_dxf = os.path.join(OUTPUT_DIR, file.filename.replace(".png", "_dxf"))
        # 异步执行 CAD 处理任务
        task = process_cad_image.delay(file_path, output_dxf)    
    elif task_name == "ocr_image":
        output_ocr = os.path.join(OUTPUT_DIR, file.filename.replace(".png", "_ocr"))
        task = ocr_image.delay(file_path, output_ocr)
    else:
        return {"error": "Unknown task type"}

    return {"task_id": task.id, "message": "Processing started"}

"""
    Upload and process PDF file to convert into images.

    Args:
        file (UploadFile): The PDF file to be uploaded and processed
        task_name (str): Task type identifier, must be "pdf_to_images"

    Returns:
        dict: Contains task_id and status message
            - task_id: ID of the async conversion task
            - message: Processing status message
            - error: Error message if validation fails

    Raises:
        None

    Notes:
        - Validates file extension and task type
        - Saves uploaded PDF to UPLOAD_DIR
        - Creates output directory for images
        - Triggers async PDF to images conversion task
"""
@app.post("/upload/pdf/")
async def upload_pdf(file: UploadFile = File(...), task_name: str = Form(...)):
    if task_name != "pdf_to_images":
        return {"error": "Unknown task type"}
    if not Util.validate_input_path(file.filename, [".pdf"]):
        return {"error": "Invalid file type"}
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
