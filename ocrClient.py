# -*- coding: utf-8 -*-
import requests
import base64
import time

def test_ocr(image_path, scale_factor=5, max_block_size=512, overlap=20, url='http://localhost:9003/ocr'):
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    start = time.time()
    response = requests.post(url, json={'image': img_b64, 'scale_factor': scale_factor, 
                                        'max_block_size': max_block_size, 'overlap': overlap})
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
    scale_factor=5
    max_block_size=512
    overlap=50
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(test_ocr, 'D:/Image2CADPy/TestData/pdfImages/pdf_page_1.png', scale_factor, max_block_size, overlap) for _ in range(10)]
        results = [f.result() for f in futures]
        print(results)