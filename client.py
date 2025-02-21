# -*- coding: utf-8 -*-
import requests
import time

# 上传图片
file_path = "test.png"
upload_url = "http://127.0.0.1:8000/upload/"

with open(file_path, "rb") as f:
    response = requests.post(upload_url, files={"file": f})

task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")

# 轮询任务状态
while True:
    task_status = requests.get(f"http://127.0.0.1:8000/task/{task_id}").json()
    print(f"Task status: {task_status['status']}")
    if task_status["status"] == "SUCCESS":
        break
    time.sleep(2)

# 下载 DXF
download_url = "http://127.0.0.1:8000/download/test.dxf"
response = requests.get(download_url)

with open("test.dxf", "wb") as f:
    f.write(response.content)

print("DXF 文件已下载！")
