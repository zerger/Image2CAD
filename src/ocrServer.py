# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import base64
from queue import Queue
from rapidocr import RapidOCR
import sys
import logging

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
        engine = RapidOCR()
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
        return jsonify({"error": "Missing image parameter"}), 400
    
    # Base64解码
    try:
        img_b64 = request.json['image'].split(',')[-1]
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        logging.error(f"Base64解码失败: {str(e)}")
        return jsonify({"error": f"无效的图片数据: {str(e)}"}), 400

    # 获取OCR引擎
    engine = engine_pool.get()
    try:
        # 直接处理二进制数据
        result, elapse = engine(img_bytes)
    except Exception as e:
        logging.error(f"OCR处理失败: {str(e)}")
        return jsonify({"error": f"OCR处理失败: {str(e)}"}), 500
    finally:
        engine_pool.put(engine)

    # 结果格式化（根据实际数据结构调整）
    formatted = []
    for item in result:
        formatted.append({
            "coordinates": item[0],  # 保持原始坐标结构
            "text": item[1],
            "confidence": float(item[2])
        })

    return jsonify({
        "result": formatted,
        "processing_time": elapse,
        "engine_count": POOL_SIZE
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9003, threaded=True)