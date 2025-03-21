# -*- coding: utf-8 -*-
import requests
import time
import sys
import websockets, asyncio

async def ws_client():
    async with websockets.connect('ws://127.0.0.1:8000/test') as ws:
        await ws.send('this is a test')
        response = await ws.recv()
        print(response)
        
def upload_file(file_path, upload_url, task_name):
    with open(file_path, "rb") as f:
        response = requests.post(upload_url, files={"file": f}, data={"task_name": task_name})
        
        # 打印响应状态码和内容以进行调试
        print("Response status code:", response.status_code)
        print("Response content:", response.text)        
        # 检查 HTTP 错误
        response.raise_for_status()
        # 检查响应的内容类型是否为 JSON
        if response.headers.get('Content-Type') == 'application/json':
            json_response = response.json()
            if json_response.get("error"):
                raise ValueError(json_response.get("error"))
            return json_response.get("task_id")
        else:
            print("Unexpected content type:", response.headers.get('Content-Type'))
            raise ValueError("Expected JSON response")

def poll_task_status(server_ip, task_id):
    while True:
        task_status = requests.get(f"http://{server_ip}:8000/task/{task_id}").json()
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

    # 修改为服务器的实际 IP 地址
    server_ip = "10.0.101.60"

    if task_type == "png_to_dxf":
        upload_url = f"http://{server_ip}:8000/upload/image/"
        download_filename = "test.dxf"
    elif task_type == "pdf_to_images":
        upload_url = f"http://{server_ip}:8000/upload/pdf/"
        download_filename = "test_images.zip"
    elif task_type == "ocr_image":
        upload_url = f"http://{server_ip}:8000/upload/image/"
        download_filename = "test_ocr.json"
    elif task_type == "ws_client":
        asyncio.run(ws_client())
        sys.exit(1)
    else:
        print("未知的任务类型")
        sys.exit(1)

    task_id = upload_file(file_path, upload_url, task_type)
    print(f"Task ID: {task_id}")

    poll_task_status(server_ip, task_id)

    download_url = f"http://{server_ip}:8000/download/{download_filename}"
    download_file(download_url, download_filename)
