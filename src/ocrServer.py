# -*- coding: utf-8 -*-
import time
from flask import Flask, request, jsonify
import base64
from queue import Queue
from ocrProcess import OCRProcess
import sys
import logging
from io import BytesIO
from PIL import Image
from logManager import LogManager, setup_logging

log_mgr = LogManager().get_instance()

# 使用示例： python ocr_server.py 5

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# 初始化OCR引擎池
def init_engine_pool(pool_size):
    engine_queue = Queue(maxsize=pool_size)
    for _ in range(pool_size):
        engine = OCRProcess()
        engine_queue.put(engine)
    return engine_queue

# 命令行参数处理
if len(sys.argv) != 2:
    print("Usage: python ocr_server.py N")
    sys.exit(1)

try:
    POOL_SIZE = int(sys.argv[1])
except ValueError:
    print("Error: N must be an integer")
    sys.exit(1)

engine_pool = init_engine_pool(POOL_SIZE)

@app.route('/ocr', methods=['POST'])
def ocr_service():
    # 参数校验
    if 'image' not in request.json:
        logging.error("Missing image parameter")
        return jsonify({"error": "Missing image parameter"}), 400
    scale_factor=5
    max_block_size=512
    overlap=20
    if 'scale_factor' in request.json:
        scale_factor = request.json['scale_factor']
    if 'max_block_size' in request.json:
        max_block_size = request.json['max_block_size']
    if 'overlap' in request.json:
        overlap = request.json['overlap']
        
    # Base64解码
    try:
        img_b64 = request.json['image'].split(',')[-1]
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        logging.error(f"Base64解码失败: {str(e)}")
        return jsonify({"error": f"无效的图片数据: {str(e)}"}), 400  
    ocr_process = engine_pool.get()
    try:
        # 开始计时
        start_time = time.time()   
        # output_path = ocr_process.get_temp_directory() + "/output_image.png" 
        # with open(output_path, 'wb') as f:
        #     f.write(img_bytes)
        image = Image.open(BytesIO(img_bytes))
        # 直接处理二进制数据      
        text_positions = ocr_process.get_image_rapidOCR(image, scale_factor, max_block_size, overlap, None)    
        # 结束计时
        elapse = time.time() - start_time
        
        # 结果格式化（根据实际数据结构调整）
        formatted = []
        words, page_height = text_positions  
        if page_height is not None and words is not None:
            for text, x, y, width, height, angle in words:  
                box = [x, y, x+width, y+height]
                formatted.append({
                    "coordinates": box,  # 保持原始坐标结构
                    "text": text                        
                })        
    except Exception as e:        
        log_mgr.log_exception(f"OCR处理失败: {e}")
        return jsonify({"error": f"OCR处理失败: {str(e)}"}), 500
    finally:
        engine_pool.put(ocr_process)

    return jsonify({
        "result": formatted,
        "processing_time": elapse,
        "engine_count": POOL_SIZE
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9003, threaded=True)