# -*- coding: utf-8 -*-
import requests
import base64
import time

def test_ocr(image_path, url='http://10.0.101.60:9003/ocr'):
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    start = time.time()
    response = requests.post(url, json={'image': img_b64})
    latency = time.time() - start
    
    if response.status_code == 200:
        print(f"识别成功 | 耗时: {latency:.2f}s")
        return response.json()
    else:
        print(f"识别失败 | 状态码: {response.status_code}")
        return response.text

# 并发测试示例
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":  
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(test_ocr, 'D:/Image2CADPy/TestData/1.png') for _ in range(10)]
        results = [f.result() for f in futures]