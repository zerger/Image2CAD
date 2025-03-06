# -*- coding: utf-8 -*-
import requests
import time
import sys

def upload_file(file_path, upload_url):
    with open(file_path, "rb") as f:
        response = requests.post(upload_url, files={"file": f})
    return response.json()["task_id"]

def poll_task_status(task_id):
    while True:
        task_status = requests.get(f"http://127.0.0.1:8000/task/{task_id}").json()
        print(f"Task status: {task_status['status']}")
        if task_status["status"] == "SUCCESS":
            return
        time.sleep(2)

def download_file(download_url, output_path):
    response = requests.get(download_url)
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"文件已下载到 {output_path}！")

if __name__ == "__main__":
    task_type = sys.argv[1]  # 任务类型，例如 "png_to_dxf" 或 "pdf_to_images"
    file_path = sys.argv[2]  # 上传的文件路径

    if task_type == "png_to_dxf":
        upload_url = "http://127.0.0.1:8000/upload/image/"
        download_filename = "test.dxf"
    elif task_type == "pdf_to_images":
        upload_url = "http://127.0.0.1:8000/upload/pdf/"
        download_filename = "test_images.zip"
    else:
        print("未知的任务类型")
        sys.exit(1)

    task_id = upload_file(file_path, upload_url)
    print(f"Task ID: {task_id}")

    poll_task_status(task_id)

    download_url = f"http://127.0.0.1:8000/download/{download_filename}"
    download_file(download_url, download_filename)
